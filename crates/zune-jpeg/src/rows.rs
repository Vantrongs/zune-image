//! Baseline grayscale JPEG decoded one MCU row (eight pixel rows) at a time, with
//! checkpoints to resume from at any MCU row.
//!
//! Only a single complete baseline scan of one component with 1x1 sampling is
//! accepted. Nothing image-sized is allocated: a reader holds one row of coefficients
//! and writes each band into the caller's buffer. The encoded input is shared, never
//! copied, by readers and checkpoints.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

#[cfg(not(feature = "std"))]
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;

use crate::bitstream::BitStream;
use crate::errors::DecodeErrors;
use crate::mcu::McuContinuation;
use crate::mcu_prog::get_marker;
use crate::misc::setup_component_params;
#[cfg(not(feature = "std"))]
use crate::progress::ProgressReader;
use crate::progress::{stopped, Progress};
use crate::JpegDecoder;

/// Why a row reader could not be created, read, or restored.
#[derive(Debug)]
pub enum RowDecodeError {
    /// The image is not a single baseline grayscale scan with 1x1 sampling; decode
    /// it with [`JpegDecoder`].
    Unsupported,
    /// The checkpoint belongs to a reader of another source.
    ForeignCheckpoint,
    /// An earlier read failed; restore a checkpoint before reading again.
    Failed,
    /// The JPEG data is invalid, or the output buffer is too small.
    Decode(DecodeErrors),
    /// The reader's [`Progress`] stopped it; a read that stopped leaves the reader
    /// [`RowDecodeError::Failed`].
    Stopped,
}

impl fmt::Display for RowDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str(
                "Row decoding requires a single baseline grayscale scan with 1x1 sampling",
            ),
            Self::ForeignCheckpoint => f.write_str("Checkpoint belongs to another JPEG source"),
            Self::Failed => f.write_str("Restore a checkpoint before reusing a failed row decoder"),
            Self::Decode(e) => fmt::Display::fmt(e, f),
            Self::Stopped => f.write_str("Row decoding stopped"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RowDecodeError {}

impl From<DecodeErrors> for RowDecodeError {
    fn from(error: DecodeErrors) -> Self {
        Self::Decode(error)
    }
}

struct Source<S> {
    bytes: S,
    options: DecoderOptions,
}

/// The shared source as the decoder's byte input.
struct SharedSource<S>(Arc<Source<S>>);

impl<S: AsRef<[u8]>> AsRef<[u8]> for SharedSource<S> {
    fn as_ref(&self) -> &[u8] {
        self.0.bytes.as_ref()
    }
}

/// The input, its reads told to a progress.
#[cfg(not(feature = "std"))]
type RowInput<'p, S> = ProgressReader<'p, ZCursor<SharedSource<S>>>;

#[cfg(not(feature = "std"))]
fn row_input<S: AsRef<[u8]>>(source: SharedSource<S>, progress: &dyn Progress) -> RowInput<'_, S> {
    ProgressReader::new(ZCursor::new(source), progress)
}

/// The input read through an 8 KiB buffer. Without it every refill of the entropy
/// decoder calls `S::as_ref` again, which for a source that resolves its bytes through
/// shared owners costs more than the decoding. Each refill of the buffer is told to
/// the progress, rather than each of the decoder's reads.
#[cfg(feature = "std")]
struct RowInput<'p, S>(std::io::BufReader<Refills<'p, std::io::Cursor<SharedSource<S>>>>);

#[cfg(feature = "std")]
fn row_input<S: AsRef<[u8]>>(source: SharedSource<S>, progress: &dyn Progress) -> RowInput<'_, S> {
    RowInput(std::io::BufReader::with_capacity(
        8192,
        Refills {
            inner: std::io::Cursor::new(source),
            progress,
            stopped: false,
        },
    ))
}

/// A reader that tells `progress` the bytes of each read, at least one for a read
/// that found no more, and reads as ended once it stops.
#[cfg(feature = "std")]
struct Refills<'p, R> {
    inner: R,
    progress: &'p dyn Progress,
    stopped: bool,
}

#[cfg(feature = "std")]
impl<R: std::io::Read> std::io::Read for Refills<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.stopped {
            return Ok(0);
        }
        let read = self.inner.read(buf)?;
        self.stopped = self.progress.advance(read.max(1));
        Ok(read)
    }
}

#[cfg(feature = "std")]
impl<R: std::io::Seek> std::io::Seek for Refills<'_, R> {
    fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(position)
    }

    fn stream_position(&mut self) -> std::io::Result<u64> {
        self.inner.stream_position()
    }
}

#[cfg(feature = "std")]
impl<S: AsRef<[u8]>> std::io::Read for RowInput<'_, S> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

#[cfg(feature = "std")]
impl<S: AsRef<[u8]>> std::io::BufRead for RowInput<'_, S> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.0.fill_buf()
    }

    fn consume(&mut self, amount: usize) {
        self.0.consume(amount);
    }
}

#[cfg(feature = "std")]
impl<S: AsRef<[u8]>> std::io::Seek for RowInput<'_, S> {
    fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
        match position {
            // The marker lookahead steps back often; `BufReader::seek` would drop the
            // buffer each time.
            std::io::SeekFrom::Current(offset) => {
                self.0.seek_relative(offset)?;
                self.0.stream_position()
            }
            _ => self.0.seek(position),
        }
    }

    fn stream_position(&mut self) -> std::io::Result<u64> {
        self.0.stream_position()
    }
}

/// A reader of 8-bit luma bands of at most eight rows, tightly packed.
///
/// `S` must return the same bytes for as long as it lives. It is kept in an `Arc`
/// shared with the reader's checkpoints, so it need not be `Clone`. Output is luma
/// whatever output colour space `options` asks for. The reader tells its
/// [`Progress`] the bytes it reads and each band's size before decoding it, so that
/// the work between reports is bounded whatever the data, and fails with
/// [`RowDecodeError::Stopped`] once it stops.
pub struct GrayscaleRows<'p, S> {
    source: Arc<Source<S>>,
    progress: &'p dyn Progress,
    decoder: JpegDecoder<RowInput<'p, S>>,
    bits: BitStream,
    next_row: usize,
    /// The data ended (or broke off, when not strict); the remaining rows are 128.
    fill_rest: bool,
    failed: bool,
}

/// A reader's state at an MCU row boundary, independent of the reader. It shares the
/// encoded input and holds one row of coefficients, which a band cut short by the
/// end of the data reuses.
pub struct GrayscaleCheckpoint<S> {
    source: Arc<Source<S>>,
    position: usize,
    bits: BitStream,
    next_row: usize,
    dc_pred: i32,
    todo: usize,
    raw: Vec<i16>,
    fill_rest: bool,
}

impl<S> Clone for GrayscaleCheckpoint<S> {
    fn clone(&self) -> Self {
        Self {
            source: Arc::clone(&self.source),
            position: self.position,
            bits: self.bits.clone(),
            next_row: self.next_row,
            dc_pred: self.dc_pred,
            todo: self.todo,
            raw: self.raw.clone(),
            fill_rest: self.fill_rest,
        }
    }
}

impl<'p, S: AsRef<[u8]>> GrayscaleRows<'p, S> {
    /// Parses the headers and checks, in one pass over the entropy-coded bytes
    /// without decoding them, that no other scan or table follows: such an image is
    /// [`RowDecodeError::Unsupported`] before any row is returned.
    pub fn new(
        bytes: S, options: DecoderOptions, progress: &'p dyn Progress,
    ) -> Result<Self, RowDecodeError> {
        let source = Arc::new(Source {
            bytes,
            options: options.jpeg_set_out_colorspace(ColorSpace::Luma),
        });
        let mut decoder = parse_headers(&source, progress)?;
        let entropy = source
            .bytes
            .as_ref()
            .get(input_position(&mut decoder)?..)
            .ok_or(DecodeErrors::ExhaustedData)?;
        validate_single_scan(entropy, progress)?;
        Self::at_first_row(source, decoder, progress)
    }

    fn at_first_row(
        source: Arc<Source<S>>, mut decoder: JpegDecoder<RowInput<'p, S>>,
        progress: &'p dyn Progress,
    ) -> Result<Self, RowDecodeError> {
        setup_component_params(&mut decoder)?;
        decoder.check_tables()?;
        let component = &mut decoder.components[0];
        component.needed = true;
        component.raw_coeff = vec![0; component.width_stride * 8];
        Ok(Self {
            source,
            progress,
            decoder,
            bits: BitStream::new(),
            next_row: 0,
            fill_rest: false,
            failed: false,
        })
    }

    /// Width and height in pixels.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (
            usize::from(self.decoder.width()),
            usize::from(self.decoder.height()),
        )
    }

    /// The first row the next read returns.
    #[must_use]
    pub fn next_row(&self) -> usize {
        self.next_row
    }

    /// The bytes the next read writes; zero after the last row.
    #[must_use]
    pub fn output_buffer_size(&self) -> usize {
        let (width, height) = self.dimensions();
        width * (height - self.next_row).min(8)
    }

    /// Decodes the next band into the first `output_buffer_size()` bytes of `output`
    /// and returns its rows; zero after the last row.
    ///
    /// A buffer too small is an error that leaves the reader as it was. A decoding
    /// error makes the reader [`RowDecodeError::Failed`] until a checkpoint is
    /// restored. Data that ends early gives the same rows as [`JpegDecoder::decode`].
    pub fn read_mcu_row(&mut self, output: &mut [u8]) -> Result<usize, RowDecodeError> {
        if self.failed {
            return Err(RowDecodeError::Failed);
        }
        let size = self.output_buffer_size();
        if output.len() < size {
            return Err(DecodeErrors::TooSmallOutput(size, output.len()).into());
        }
        if size == 0 {
            return Ok(0);
        }
        if self.progress.advance(size) {
            return Err(RowDecodeError::Stopped);
        }
        let result = self.read_band(&mut output[..size]);
        // A stop ends the input, which the band may have read as data cut short.
        if stopped(self.progress) {
            self.failed = true;
            return Err(RowDecodeError::Stopped);
        }
        self.failed = result.is_err();
        Ok(result?)
    }

    fn read_band(&mut self, output: &mut [u8]) -> Result<usize, DecodeErrors> {
        let (width, height) = self.dimensions();
        if self.bits.overread_by > 0 && !self.fill_rest {
            if self.decoder.options.strict_mode() {
                return Err(DecodeErrors::FormatStatic("Premature end of buffer"));
            }
            self.fill_rest = true;
        }
        if self.fill_rest {
            output.fill(128);
        } else {
            let mcu_row = self.next_row / 8;
            let continuation = self.decoder.inner_decode_mcu_width::<false, false>(
                width.div_ceil(8),
                mcu_row,
                &mut [0; 64],
                &mut self.bits,
                &mut core::array::from_fn(|_| Vec::new()),
            )?;
            self.decoder.post_process(
                output,
                mcu_row,
                height.div_ceil(8),
                width,
                width.div_ceil(8) * 8,
                &mut 0,
                &mut [],
            )?;
            match continuation {
                McuContinuation::Ok => {}
                McuContinuation::Terminate => self.fill_rest = true,
                // `new` rejected sources with these markers.
                McuContinuation::AnotherSos | McuContinuation::InterScanMarker(_) => {
                    return Err(DecodeErrors::FormatStatic(
                        "Row decoding cannot continue across scans",
                    ));
                }
            }
        }
        let rows = output.len() / width;
        self.next_row += rows;
        if self.next_row == height && !self.fill_rest && !self.bits.seen_eoi {
            // Consume the final marker as `JpegDecoder::decode` does, on a best-effort basis.
            let _ = get_marker(&mut self.decoder.stream, &mut self.bits);
        }
        Ok(rows)
    }

    /// The reader's state at its next row. No input bytes are copied.
    pub fn checkpoint(&mut self) -> Result<GrayscaleCheckpoint<S>, RowDecodeError> {
        if self.failed {
            return Err(RowDecodeError::Failed);
        }
        Ok(GrayscaleCheckpoint {
            source: Arc::clone(&self.source),
            position: input_position(&mut self.decoder)?,
            bits: self.bits.clone(),
            next_row: self.next_row,
            dc_pred: self.decoder.components[0].dc_pred,
            todo: self.decoder.todo,
            raw: self.decoder.components[0].raw_coeff.clone(),
            fill_rest: self.fill_rest,
        })
    }

    /// Continues from `checkpoint`, which must come from this reader or a reader of
    /// the same source; a foreign one is rejected and the reader is left as it was.
    pub fn restore(&mut self, checkpoint: &GrayscaleCheckpoint<S>) -> Result<(), RowDecodeError> {
        if !Arc::ptr_eq(&self.source, &checkpoint.source) {
            return Err(RowDecodeError::ForeignCheckpoint);
        }
        self.decoder
            .stream
            .set_position(checkpoint.position)
            .map_err(DecodeErrors::from)?;
        self.bits = checkpoint.bits.clone();
        self.next_row = checkpoint.next_row;
        let component = &mut self.decoder.components[0];
        component.dc_pred = checkpoint.dc_pred;
        component.raw_coeff.copy_from_slice(&checkpoint.raw);
        self.decoder.todo = checkpoint.todo;
        self.fill_rest = checkpoint.fill_rest;
        self.failed = false;
        Ok(())
    }
}

impl<S: AsRef<[u8]>> GrayscaleCheckpoint<S> {
    /// The first row a reader resumed from here returns.
    #[must_use]
    pub fn next_row(&self) -> usize {
        self.next_row
    }

    /// The bytes this checkpoint takes: the value itself and its row of coefficients.
    /// The shared source is not included.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        core::mem::size_of::<Self>() + self.raw.capacity() * core::mem::size_of::<i16>()
    }

    /// A new reader at this checkpoint, from parsing the headers again, telling
    /// `progress` its work; the single-scan check of [`GrayscaleRows::new`] is not
    /// repeated.
    pub fn resume<'p>(
        &self, progress: &'p dyn Progress,
    ) -> Result<GrayscaleRows<'p, S>, RowDecodeError> {
        let source = Arc::clone(&self.source);
        let decoder = parse_headers(&source, progress)?;
        let mut rows = GrayscaleRows::at_first_row(source, decoder, progress)?;
        rows.restore(self)?;
        Ok(rows)
    }
}

/// A decoder past the headers of `source`, if its image is one rows can be read from;
/// the bytes it reads are told to `progress`.
fn parse_headers<'p, S: AsRef<[u8]>>(
    source: &Arc<Source<S>>, progress: &'p dyn Progress,
) -> Result<JpegDecoder<RowInput<'p, S>>, RowDecodeError> {
    let mut decoder = JpegDecoder::new_with_options(
        row_input(SharedSource(Arc::clone(source)), progress),
        source.options,
    );
    let parsed = decoder.decode_headers();
    if stopped(progress) {
        return Err(RowDecodeError::Stopped);
    }
    parsed?;
    if decoder.is_progressive
        || decoder.input_colorspace != ColorSpace::Luma
        || decoder.components.len() != 1
        || decoder.num_scans != 1
        || decoder.components[0].horizontal_sample != 1
        || decoder.components[0].vertical_sample != 1
        || decoder.spec_start != 0
        || decoder.spec_end != 63
        || decoder.succ_high != 0
        || decoder.succ_low != 0
    {
        return Err(RowDecodeError::Unsupported);
    }
    Ok(decoder)
}

fn input_position<T: zune_core::bytestream::ZByteReaderTrait>(
    decoder: &mut JpegDecoder<T>,
) -> Result<usize, DecodeErrors> {
    let position = decoder.stream.position()?;
    usize::try_from(position)
        .map_err(|_| DecodeErrors::FormatStatic("Input position does not fit usize"))
}

/// The bytes the single-scan check scans between reports to its progress.
const SCAN_PIECE: usize = 1 << 16;

/// Accepts entropy-coded data whose markers up to the first EOI are all restarts. A
/// missing EOI or a dangling 0xFF is left to the entropy decoder, which treats it as
/// data that ends early. Each piece scanned is told to `progress`.
fn validate_single_scan(bytes: &[u8], progress: &dyn Progress) -> Result<(), RowDecodeError> {
    let mut position = 0;
    let mut reported = 0;
    loop {
        if position - reported >= SCAN_PIECE {
            if progress.advance(position - reported) {
                return Err(RowDecodeError::Stopped);
            }
            reported = position;
        }
        let end = bytes.len().min(position + SCAN_PIECE);
        let Some(offset) = memchr::memchr(0xff, &bytes[position..end]) else {
            if end == bytes.len() {
                break;
            }
            position = end;
            continue;
        };
        position += offset + 1;
        let fill = bytes[position..bytes.len().min(position + SCAN_PIECE)]
            .iter()
            .take_while(|&&byte| byte == 0xff)
            .count();
        position += fill;
        let Some(&marker) = bytes.get(position) else {
            break;
        };
        // A run of fill bytes longer than a piece goes on in the next round.
        if marker == 0xff {
            continue;
        }
        position += 1;
        match marker {
            0 | 0xd0..=0xd7 => {}
            0xd9 => break,
            _ => return Err(RowDecodeError::Unsupported),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::cell::Cell;

    use super::{validate_single_scan, Progress, RowDecodeError, SCAN_PIECE};

    /// Every marker code, at offsets around the search's vector widths and after
    /// runs of fill bytes.
    #[test]
    fn only_restarts_and_the_end_pass_the_single_scan_check() {
        for offset in 0..=129 {
            for fill in [1, 2, 17] {
                for marker in 0..=u8::MAX {
                    let mut bytes = vec![0x42; offset];
                    bytes.resize(offset + fill, 0xff);
                    bytes.extend_from_slice(&[marker, 0, 0x42, 0xff, 0xd9]);
                    let admitted = matches!(marker, 0 | 0xd0..=0xd7 | 0xd9 | 0xff);
                    let result = validate_single_scan(&bytes, &());
                    assert_eq!(
                        result.is_ok(),
                        admitted,
                        "offset={offset} fill={fill} marker={marker:02x}"
                    );
                    if !admitted {
                        assert!(matches!(result, Err(RowDecodeError::Unsupported)));
                    }
                }
            }
        }
    }

    #[test]
    fn data_cut_anywhere_passes_the_single_scan_check() {
        let mut bytes = vec![0x42; 129];
        bytes.extend_from_slice(&[0xff, 0, 0x42]);
        for restart in 0xd0..=0xd7 {
            bytes.extend_from_slice(&[0xff, 0xff, restart, 0x42]);
        }
        bytes.extend_from_slice(&[0xff, 0xff, 0xff, 0xd9]);
        for end in 0..=bytes.len() {
            assert!(
                validate_single_scan(&bytes[..end], &()).is_ok(),
                "end={end}"
            );
        }
    }

    #[test]
    fn markers_after_the_first_end_are_not_checked() {
        for marker in 0..=u8::MAX {
            let bytes = [0x42, 0xff, 0xff, 0xd9, 0xff, 0xd8, 0xff, marker];
            assert!(
                validate_single_scan(&bytes, &()).is_ok(),
                "marker={marker:02x}"
            );
        }
    }

    /// Markers and runs of fill bytes across the pieces scanned between reports are
    /// checked as within one.
    #[test]
    fn markers_across_pieces_are_checked() {
        for (fill, data) in [
            (3 * SCAN_PIECE, 0),
            (1, 3 * SCAN_PIECE - 1),
            (2, 2 * SCAN_PIECE),
        ] {
            // After a restart the table marker is checked; after the end it is not.
            for marker in [0xc4, 0xd3, 0xd9] {
                let mut bytes = vec![0x42; data];
                bytes.resize(data + fill, 0xff);
                bytes.extend_from_slice(&[marker, 0x42, 0xff, 0xc4]);
                let result = validate_single_scan(&bytes, &());
                assert_eq!(
                    result.is_ok(),
                    marker == 0xd9,
                    "fill={fill} data={data} marker={marker:02x}"
                );
            }
        }
    }

    /// The scan reports each piece and stops within one more once its progress does.
    #[test]
    fn the_scan_stops_within_a_piece_of_its_progress() {
        struct Stop(Cell<usize>);
        impl Progress for Stop {
            fn advance(&self, work: usize) -> bool {
                self.0.set(self.0.get() + work);
                self.0.get() >= SCAN_PIECE
            }
        }
        for byte in [0x42, 0xff] {
            let bytes = vec![byte; 64 * SCAN_PIECE];
            let progress = Stop(Cell::new(0));
            assert!(matches!(
                validate_single_scan(&bytes, &progress),
                Err(RowDecodeError::Stopped)
            ));
            assert!(
                progress.0.get() < 3 * SCAN_PIECE,
                "{byte:02x}: {}",
                progress.0.get()
            );
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod input_tests {
    use alloc::sync::Arc;
    use alloc::vec::Vec;

    use zune_core::bytestream::{ZByteReaderTrait, ZSeekFrom};
    use zune_core::options::DecoderOptions;

    use super::{row_input, RowInput, SharedSource, Source};

    fn input(bytes: Vec<u8>) -> RowInput<'static, Vec<u8>> {
        let source = SharedSource(Arc::new(Source {
            bytes,
            options: DecoderOptions::default(),
        }));
        row_input(source, &())
    }

    /// The buffered input keeps the positions, peeks and failed reads of a plain
    /// cursor across its buffer boundaries.
    #[test]
    fn the_buffered_input_reads_as_a_cursor_does() {
        let bytes: Vec<u8> = (0..=u8::MAX).cycle().take(16403).collect();
        let mut reader = input(bytes.clone());
        assert_eq!(reader.read_byte_no_error(), bytes[0]);
        assert_eq!(reader.z_seek(ZSeekFrom::Start(8190)).unwrap(), 8190);
        let mut peeked = [0; 9];
        reader.peek_exact_bytes(&mut peeked).unwrap();
        assert_eq!(peeked, bytes[8190..8199]);
        assert_eq!(reader.z_position().unwrap(), 8190);
        assert_eq!(reader.z_seek(ZSeekFrom::Current(9)).unwrap(), 8199);
        assert_eq!(reader.z_seek(ZSeekFrom::Current(-10)).unwrap(), 8189);
        assert_eq!(reader.read_byte_no_error(), bytes[8189]);
        assert_eq!(reader.z_seek(ZSeekFrom::End(-3)).unwrap(), 16400);
        let mut too_long = [0; 9];
        assert!(reader.read_exact_bytes(&mut too_long).is_err());
        assert!(reader.peek_exact_bytes(&mut too_long).is_err());
        assert_eq!(reader.z_position().unwrap(), 16400);
        assert_eq!(reader.peek_bytes(&mut too_long).unwrap(), 3);
        assert_eq!(&too_long[..3], &bytes[16400..]);
        assert_eq!(reader.read_bytes(&mut too_long).unwrap(), 3);
        assert!(reader.is_eof().unwrap());
        assert_eq!(reader.read_byte_no_error(), 0);
        assert!(reader.z_seek(ZSeekFrom::Current(-16404)).is_err());
        reader.z_seek(ZSeekFrom::Start(4)).unwrap();
        assert_eq!(reader.read_byte_no_error(), bytes[4]);
    }
}
