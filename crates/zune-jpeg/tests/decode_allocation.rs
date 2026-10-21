//! What decoding one small image allocates. A PDF page can hold thousands of
//! small JPEGs, each decoded by a new decoder, so every allocation a decode makes
//! is paid per image; these numbers are exact so that a change which allocates
//! more, or less, says so here. Update them when that is intended.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::JpegDecoder;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

// SAFETY: forwards to `System`, only counting.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator, which takes it from `System`.
        unsafe { System.dealloc(ptr, layout) };
    }
}

fn count(size: usize) {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        BYTES.set(BYTES.get() + size);
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// A 24x16 colour image with 4:2:0 sampling, as most JPEGs in PDFs are.
fn jpeg(progressive: bool) -> Vec<u8> {
    let (width, height) = (24_u16, 16_u16);
    let pixels: Vec<u8> = (0..usize::from(width) * usize::from(height) * 3)
        .map(|i| (i * 37 % 251) as u8)
        .collect();
    let mut out = Vec::new();
    let mut encoder = Encoder::new(&mut out, 90);
    encoder.set_sampling_factor(SamplingFactor::F_2_2);
    encoder.set_progressive(progressive);
    encoder.encode(&pixels, width, height, ColorType::Rgb).unwrap();
    out
}

/// Allocations and bytes allocated (reallocations counted as allocations of the
/// new size) by a decode of `jpeg`, the decoded pixels included.
fn decode(jpeg: &[u8]) -> (usize, usize) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    COUNTING.set(true);
    let mut decoder = JpegDecoder::new(ZCursor::new(jpeg));
    decoder.decode_headers().unwrap();
    let pixels = decoder.decode().unwrap();
    drop((pixels, decoder));
    COUNTING.set(false);
    (ALLOCATIONS.get(), BYTES.get())
}

/// With `arith`, a decoder also holds the arithmetic coding statistics.
fn plus_arith((allocations, bytes): (usize, usize)) -> (usize, usize) {
    if cfg!(feature = "arith") {
        (allocations + 2, bytes + 2416)
    } else {
        (allocations, bytes)
    }
}

#[test]
fn a_baseline_decode_allocates() {
    assert_eq!(decode(&jpeg(false)), plus_arith((20, 9312)));
}

#[test]
fn a_progressive_decode_allocates() {
    assert_eq!(decode(&jpeg(true)), plus_arith((23, 10848)));
}
