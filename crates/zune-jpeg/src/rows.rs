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

#[cfg(not(feature = "std"))]
type RowInput<S> = ZCursor<SharedSource<S>>;

/// The input read through an 8 KiB buffer. Without it every refill of the entropy
/// decoder calls `S::as_ref` again, which for a source that resolves its bytes through
/// shared owners costs more than the decoding.
#[cfg(feature = "std")]
struct RowInput<S>(std::io::BufReader<std::io::Cursor<SharedSource<S>>>);

#[cfg(feature = "std")]
impl<S: AsRef<[u8]>> RowInput<S> {
    fn new(source: SharedSource<S>) -> Self {
        Self(std::io::BufReader::with_capacity(
            8192,
            std::io::Cursor::new(source),
        ))
    }
}

#[cfg(feature = "std")]
impl<S: AsRef<[u8]>> std::io::Read for RowInput<S> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

#[cfg(feature = "std")]
impl<S: AsRef<[u8]>> std::io::BufRead for RowInput<S> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.0.fill_buf()
    }

    fn consume(&mut self, amount: usize) {
        self.0.consume(amount);
    }
}

#[cfg(feature = "std")]
impl<S: AsRef<[u8]>> std::io::Seek for RowInput<S> {
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
/// whatever output colour space `options` asks for.
pub struct GrayscaleRows<S> {
    source: Arc<Source<S>>,
    decoder: JpegDecoder<RowInput<S>>,
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

impl<S: AsRef<[u8]>> GrayscaleRows<S> {
    /// Parses the headers and checks, in one pass over the entropy-coded bytes
    /// without decoding them, that no other scan or table follows: such an image is
    /// [`RowDecodeError::Unsupported`] before any row is returned.
    pub fn new(bytes: S, options: DecoderOptions) -> Result<Self, RowDecodeError> {
        let source = Arc::new(Source {
            bytes,
            options: options.jpeg_set_out_colorspace(ColorSpace::Luma),
        });
        let mut decoder = parse_headers(&source)?;
        let entropy = source
            .bytes
            .as_ref()
            .get(input_position(&mut decoder)?..)
            .ok_or(DecodeErrors::ExhaustedData)?;
        validate_single_scan(entropy)?;
        Self::at_first_row(source, decoder)
    }

    fn at_first_row(
        source: Arc<Source<S>>, mut decoder: JpegDecoder<RowInput<S>>,
    ) -> Result<Self, RowDecodeError> {
        setup_component_params(&mut decoder)?;
        decoder.check_tables()?;
        let component = &mut decoder.components[0];
        component.needed = true;
        component.raw_coeff = vec![0; component.width_stride * 8];
        Ok(Self {
            source,
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
        let result = self.read_band(&mut output[..size]);
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

    /// A new reader at this checkpoint, from parsing the headers again; the single-scan
    /// check of [`GrayscaleRows::new`] is not repeated.
    pub fn resume(&self) -> Result<GrayscaleRows<S>, RowDecodeError> {
        let source = Arc::clone(&self.source);
        let decoder = parse_headers(&source)?;
        let mut rows = GrayscaleRows::at_first_row(source, decoder)?;
        rows.restore(self)?;
        Ok(rows)
    }
}

/// A decoder past the headers of `source`, if its image is one rows can be read from.
fn parse_headers<S: AsRef<[u8]>>(
    source: &Arc<Source<S>>,
) -> Result<JpegDecoder<RowInput<S>>, RowDecodeError> {
    let mut decoder = JpegDecoder::new_with_options(
        RowInput::new(SharedSource(Arc::clone(source))),
        source.options,
    );
    decoder.decode_headers()?;
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

/// Accepts entropy-coded data whose markers up to the first EOI are all restarts. A
/// missing EOI or a dangling 0xFF is left to the entropy decoder, which treats it as
/// data that ends early.
fn validate_single_scan(bytes: &[u8]) -> Result<(), RowDecodeError> {
    let mut position = 0;
    while let Some(offset) = memchr::memchr(0xff, &bytes[position..]) {
        position += offset;
        while bytes.get(position) == Some(&0xff) {
            position += 1;
        }
        let Some(&marker) = bytes.get(position) else {
            break;
        };
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

    use super::{validate_single_scan, RowDecodeError};

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
                    let result = validate_single_scan(&bytes);
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
            assert!(validate_single_scan(&bytes[..end]).is_ok(), "end={end}");
        }
    }

    #[test]
    fn markers_after_the_first_end_are_not_checked() {
        for marker in 0..=u8::MAX {
            let bytes = [0x42, 0xff, 0xff, 0xd9, 0xff, 0xd8, 0xff, marker];
            assert!(validate_single_scan(&bytes).is_ok(), "marker={marker:02x}");
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod input_tests {
    use alloc::sync::Arc;
    use alloc::vec::Vec;

    use zune_core::bytestream::{ZByteReaderTrait, ZSeekFrom};
    use zune_core::options::DecoderOptions;

    use super::{RowInput, SharedSource, Source};

    fn input(bytes: Vec<u8>) -> RowInput<Vec<u8>> {
        RowInput::new(SharedSource(Arc::new(Source {
            bytes,
            options: DecoderOptions::default(),
        })))
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
