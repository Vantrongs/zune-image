//! A row reader's memory does not grow with the image height.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use zune_jpeg::rows::GrayscaleRows;
use zune_jpeg::zune_core::options::DecoderOptions;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

// SAFETY: forwards to `System`, only counting.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is `System`'s.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && COUNTING.try_with(Cell::get).unwrap_or(false) {
            let live = LIVE.get() + layout.size();
            LIVE.set(live);
            PEAK.set(PEAK.get().max(live));
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING.try_with(Cell::get).unwrap_or(false) {
            LIVE.set(LIVE.get().wrapping_sub(layout.size()));
        }
        // SAFETY: `ptr` came from `alloc` above.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// The peak bytes of opening a 257-pixel-wide image declared `height` rows high,
/// reading a band and taking a checkpoint; and the bytes the checkpoint allocated.
fn measure(height: u16) -> (usize, usize) {
    let mut bytes = include_bytes!("row-fixtures/257x17-r7.jpg").to_vec();
    let sof = bytes.windows(2).position(|p| p == [255, 192]).unwrap();
    bytes[sof + 5..sof + 7].copy_from_slice(&height.to_be_bytes());
    let input: Arc<[u8]> = bytes.into();
    LIVE.set(0);
    PEAK.set(0);
    COUNTING.set(true);
    let options = DecoderOptions::default().set_max_height(65535);
    let mut rows = GrayscaleRows::new(Arc::clone(&input), options, &()).unwrap();
    let mut band = vec![0; rows.output_buffer_size()];
    rows.read_mcu_row(&mut band).unwrap();
    let before = LIVE.get();
    let checkpoint = rows.checkpoint().unwrap();
    let checkpoint_bytes = LIVE.get() - before;
    drop((checkpoint, band, rows));
    COUNTING.set(false);
    assert_eq!(LIVE.get(), 0);
    (PEAK.get(), checkpoint_bytes)
}

#[test]
fn memory_does_not_grow_with_the_height() {
    let (peak, checkpoint) = measure(17);
    assert_eq!(measure(65000), (peak, checkpoint));
    // A checkpoint holds one row of coefficients: 33 blocks of 64 `i16`.
    assert_eq!(checkpoint, 264 * 8 * 2);
    // That row in the reader and the checkpoint, the output band, the headers, with
    // `std` the input buffer, the token by which the state the decoder keeps to retry
    // a scan holds the Huffman tables (which stay in the decoder), and with `arith`
    // the arithmetic coding statistics, about 2.4 KiB.
    let input_buffer = if cfg!(feature = "std") { 8192 } else { 0 };
    let header_tables = 16 + if cfg!(feature = "arith") { 2560 } else { 0 };
    assert!(
        peak <= 40 * 264 + 2048 + input_buffer + header_tables,
        "peak {peak}"
    );
}
