//! Exact, full-resolution grayscale decoding one MCU row at a time.
//!
//! Only single-component, complete first scans with 1x1 sampling are accepted.
//! No full-image pixel or coefficient buffer is allocated. The input is immutable
//! and shared by readers and checkpoints; a checkpoint retains one previous MCU
//! band because permissive decoding of a truncated band can reuse its samples.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;

use crate::bitstream::BitStream;
use crate::errors::DecodeErrors;
use crate::mcu::McuContinuation;
use crate::mcu_prog::get_marker;
use crate::misc::setup_component_params;
use crate::JpegDecoder;

/// Failure to construct, advance, or restore a row decoder.
#[derive(Debug)]
pub enum RowDecodeError {
    /// This image requires the existing full-image decoder.
    Unsupported,
    /// The checkpoint belongs to another independently constructed source.
    ForeignCheckpoint,
    /// A failed reader must be restored to a valid checkpoint before reuse.
    Failed,
    /// A JPEG parsing or output-buffer error.
    Decode(DecodeErrors),
}

impl fmt::Display for RowDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str(
                "Row decoding requires a baseline grayscale complete first scan with 1x1 sampling",
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

struct SharedSource<S>(Arc<Source<S>>);

impl<S: AsRef<[u8]>> AsRef<[u8]> for SharedSource<S> {
    fn as_ref(&self) -> &[u8] {
        self.0.bytes.as_ref()
    }
}

/// A reader of tightly packed 8-bit Luma bands, at most eight rows per call.
///
/// The source must return the same bytes for its entire lifetime. Ownership is
/// retained in an `Arc`; it need not implement `Clone`. `Vec<u8>`, `Arc<[u8]>`,
/// and immutable source leases can be passed without copying the encoded bytes.
///
/// Dropping the reader early stops entropy decoding immediately. Checkpoints
/// keep the immutable encoded source alive and can create independent readers.
/// Output is always Luma, regardless of the output colorspace in `options`.
pub struct GrayscaleRows<S> {
    source: Arc<Source<S>>,
    decoder: JpegDecoder<ZCursor<SharedSource<S>>>,
    bits: BitStream,
    next_row: usize,
    fill_rest: bool,
    failed: bool,
}

/// An opaque, independently owned entropy checkpoint at an MCU row boundary.
///
/// There is no automatic index: callers must budget retained checkpoints using
/// [`Self::storage_bytes`]. Each snapshot shares (does not copy) the input.
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
    /// Parse headers and validate support before allocating the MCU row buffer.
    /// Header metadata and Huffman tables may allocate during parsing. A single
    /// structural pass over entropy bytes rejects later scans/table changes
    /// before any pixels can be returned. It does not entropy-decode the image;
    /// checkpoint resumes do not repeat this pass.
    pub fn new(bytes: S, options: DecoderOptions) -> Result<Self, RowDecodeError> {
        Self::from_source(
            Arc::new(Source {
                bytes,
                options: options.jpeg_set_out_colorspace(ColorSpace::Luma),
            }),
            true,
        )
    }

    fn from_source(source: Arc<Source<S>>, preflight: bool) -> Result<Self, RowDecodeError> {
        let mut decoder = JpegDecoder::new_with_options(
            ZCursor::new(SharedSource(Arc::clone(&source))),
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
        if preflight {
            let position = decoder.stream.position().map_err(DecodeErrors::from)?;
            let position = usize::try_from(position)
                .map_err(|_| DecodeErrors::FormatStatic("Input position does not fit usize"))?;
            let entropy = source
                .bytes
                .as_ref()
                .get(position..)
                .ok_or(DecodeErrors::ExhaustedData)?;
            validate_single_scan(entropy)?;
        }
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

    /// Source width and height in pixels.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (
            usize::from(self.decoder.width()),
            usize::from(self.decoder.height()),
        )
    }

    /// Index of the first pixel row returned by the next successful read.
    #[must_use]
    pub fn next_row(&self) -> usize {
        self.next_row
    }

    /// Bytes needed by the next read, or zero at the image end.
    #[must_use]
    pub fn output_buffer_size(&self) -> usize {
        let (width, height) = self.dimensions();
        width * (height - self.next_row).min(8)
    }

    /// Decode the next band and return its number of pixel rows (zero at EOF).
    ///
    /// Only the first `output_buffer_size()` bytes are written. Too-small buffers
    /// leave the reader unchanged. JPEG errors poison the reader until restore;
    /// bytes from a failed call must not be used. Non-strict truncation uses the
    /// same samples and 128-filled remainder as `JpegDecoder::decode`.
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
        let result = self.read_inner(&mut output[..size]);
        if result.is_err() {
            self.failed = true;
        }
        result.map_err(Into::into)
    }

    fn read_inner(&mut self, output: &mut [u8]) -> Result<usize, DecodeErrors> {
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
            let mut tmp = [0; 64];
            let continuation = self.decoder.inner_decode_mcu_width::<false, false, true>(
                width.div_ceil(8),
                self.next_row / 8,
                &mut tmp,
                &mut self.bits,
                &mut core::array::from_fn(|_| Vec::new()),
            )?;
            self.decoder.post_process(
                output,
                self.next_row / 8,
                height.div_ceil(8),
                width,
                width.div_ceil(8) * 8,
                &mut 0,
                &mut [],
            )?;
            match continuation {
                McuContinuation::Ok => {}
                McuContinuation::Terminate => self.fill_rest = true,
                McuContinuation::AnotherSos | McuContinuation::InterScanMarker(_) => {
                    return Err(DecodeErrors::FormatStatic(
                        "Row decoding cannot continue across scans",
                    ));
                }
            }
        }
        let count = output.len() / width;
        self.next_row += count;
        if self.next_row == height && !self.fill_rest && !self.bits.seen_eoi {
            // Match the full decoder's best-effort final marker consumption.
            let _ = get_marker(&mut self.decoder.stream, &mut self.bits);
        }
        Ok(count)
    }

    /// Save the complete row-boundary state. No input bytes are copied.
    pub fn checkpoint(&mut self) -> Result<GrayscaleCheckpoint<S>, RowDecodeError> {
        if self.failed {
            return Err(RowDecodeError::Failed);
        }
        let position = self.decoder.stream.position().map_err(DecodeErrors::from)?;
        let position = usize::try_from(position)
            .map_err(|_| DecodeErrors::FormatStatic("Input position does not fit usize"))?;
        Ok(GrayscaleCheckpoint {
            source: Arc::clone(&self.source),
            position,
            bits: self.bits.clone(),
            next_row: self.next_row,
            dc_pred: self.decoder.components[0].dc_pred,
            todo: self.decoder.todo,
            raw: self.decoder.components[0].raw_coeff.clone(),
            fill_rest: self.fill_rest,
        })
    }

    /// Restore a checkpoint from this source or from one of its resumed readers.
    /// A foreign checkpoint is rejected without changing the decoder.
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
        self.decoder.components[0].dc_pred = checkpoint.dc_pred;
        self.decoder.todo = checkpoint.todo;
        self.decoder.components[0]
            .raw_coeff
            .copy_from_slice(&checkpoint.raw);
        self.fill_rest = checkpoint.fill_rest;
        self.failed = false;
        Ok(())
    }
}

impl<S: AsRef<[u8]>> GrayscaleCheckpoint<S> {
    /// First pixel row produced when resumed.
    #[must_use]
    pub fn next_row(&self) -> usize {
        self.next_row
    }

    /// Owned snapshot bytes, excluding allocator overhead and shared input/config.
    /// Retaining N checkpoints requires N times this amount, not a full image.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        core::mem::size_of::<Self>() + self.raw.capacity() * core::mem::size_of::<i16>()
    }

    /// Shared source/config allocation charged once per source owner, including
    /// its two Arc counters and alignment, excluding any memory owned by `S`
    /// (such as encoded bytes) and allocator bookkeeping.
    #[must_use]
    pub fn source_storage_bytes(&self) -> usize {
        let counters = core::alloc::Layout::new::<[usize; 2]>();
        let (layout, _) = counters
            .extend(core::alloc::Layout::new::<Source<S>>())
            .expect("Source layout must fit its Arc allocation");
        layout.pad_to_align().size()
    }

    /// Reparse the immutable headers and restore this snapshot into a new reader.
    /// The original reader may have been dropped; all mutable buffers are separate.
    pub fn resume(&self) -> Result<GrayscaleRows<S>, RowDecodeError> {
        let mut rows = GrayscaleRows::from_source(Arc::clone(&self.source), false)?;
        rows.restore(self)?;
        Ok(rows)
    }
}

// Stop at the first EOI, matching the full decoder's concatenated-image boundary.
// A missing EOI or dangling FF remains a truncation for the entropy decoder,
// whose strict/non-strict behavior must stay unchanged.
fn validate_single_scan(bytes: &[u8]) -> Result<(), RowDecodeError> {
    let mut position = 0;
    while position < bytes.len() {
        if bytes[position] != 0xff {
            position += 1;
            continue;
        }
        while position < bytes.len() && bytes[position] == 0xff {
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
