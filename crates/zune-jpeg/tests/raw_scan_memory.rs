//! The most a raw row pull of a progressive image holds at once, with and without
//! scan checkpoints: a caller with the whole file bounds its memory by this.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::{JpegDecoder, RawImcuRowStatus};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct PeakAllocator;

fn grow(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: forwards to `System`, only counting.
unsafe impl GlobalAlloc for PeakAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        grow(new_size);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `ptr` came from this allocator, which takes it from `System`.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: PeakAllocator = PeakAllocator;

const SIDE: u16 = 512;

fn progressive_444() -> Vec<u8> {
    let pixels: Vec<u8> = (0..usize::from(SIDE) * usize::from(SIDE) * 3)
        .map(|i| (i * 37 % 251) as u8)
        .collect();
    let mut out = Vec::new();
    let mut encoder = Encoder::new(&mut out, 90);
    encoder.set_sampling_factor(SamplingFactor::F_1_1);
    encoder.set_progressive(true);
    encoder.encode(&pixels, SIDE, SIDE, ColorType::Rgb).unwrap();
    out
}

/// The most bytes live during a pull of every row of `jpeg`, above those live
/// before it, and the rows' bytes.
fn pull_peak(jpeg: &[u8], scan_checkpoints: bool) -> (usize, Vec<Vec<u8>>) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let mut decoder = JpegDecoder::new(ZCursor::new(jpeg));
    decoder.decode_headers().unwrap();
    let mut raw = decoder.raw_output();
    raw.set_scan_checkpoints(scan_checkpoints);
    let layout = raw.layout().unwrap();
    let count = raw.num_components().unwrap();
    let strides: Vec<usize> = layout[..count].iter().map(|plane| plane.width).collect();
    let mut rows: Vec<Vec<u8>> = layout[..count]
        .iter()
        .map(|plane| vec![0; plane.width * plane.vertical_sampling_factor * 8])
        .collect();
    let mut image: Vec<Vec<u8>> = layout[..count]
        .iter()
        .map(|plane| Vec::with_capacity(plane.width * plane.height))
        .collect();
    loop {
        let mut planes: Vec<&mut [u8]> = rows.iter_mut().map(Vec::as_mut_slice).collect();
        match raw.decode_next_imcu_row(&mut planes, &strides).unwrap() {
            RawImcuRowStatus::RowReady { rows_written } => {
                for index in 0..count {
                    let written = rows_written[index] * strides[index];
                    image[index].extend_from_slice(&rows[index][..written]);
                }
            }
            RawImcuRowStatus::Complete => break,
            status => panic!("complete input gave {status:?}"),
        }
    }
    drop(raw);
    drop(decoder);
    let image_bytes: usize = image.iter().map(Vec::capacity).sum();
    (PEAK.load(Ordering::Relaxed) - before - image_bytes, image)
}

#[test]
fn a_pull_without_scan_checkpoints_holds_one_copy_of_the_coefficients() {
    let jpeg = progressive_444();
    let coefficients = usize::from(SIDE) * usize::from(SIDE) * 3 * 2;
    let (with, rows_with) = pull_peak(&jpeg, true);
    let (without, rows_without) = pull_peak(&jpeg, false);
    assert_eq!(rows_with, rows_without);
    eprintln!("coefficients {coefficients}, with checkpoints {with}, without {without}");
    assert!(with >= 2 * coefficients, "with checkpoints: {with}");
    assert!(without < coefficients + coefficients / 4, "without checkpoints: {without}");
}
