use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use zune_jpeg::{rows::GrayscaleRows, zune_core::options::DecoderOptions};

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}
struct TrackingGuard;
impl Drop for TrackingGuard {
    fn drop(&mut self) {
        TRACK.set(false);
    }
}
struct CountingAllocator;
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() && TRACK.try_with(Cell::get).unwrap_or(false) {
            LIVE.with(|v| {
                let n = v.get() + layout.size();
                v.set(n);
                PEAK.with(|p| p.set(p.get().max(n)));
            });
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            LIVE.with(|v| v.set(v.get().wrapping_sub(layout.size())));
        }
        System.dealloc(ptr, layout);
    }
}
#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

fn measured(height: u16) -> (usize, usize) {
    let mut bytes = include_bytes!("row-fixtures/257x17-r7.jpg").to_vec();
    let sof = bytes.windows(2).position(|p| p == [255, 192]).unwrap();
    bytes[sof + 5..sof + 7].copy_from_slice(&height.to_be_bytes());
    let input: Arc<[u8]> = bytes.into();
    LIVE.set(0);
    PEAK.set(0);
    TRACK.set(true);
    let _guard = TrackingGuard;
    let mut rows = GrayscaleRows::new(
        Arc::clone(&input),
        DecoderOptions::default().set_max_height(65535),
    )
    .unwrap();
    let mut band = vec![0; rows.output_buffer_size()];
    rows.read_mcu_row(&mut band).unwrap();
    let before_checkpoint = LIVE.get();
    let checkpoint = rows.checkpoint().unwrap();
    let checkpoint_allocation = LIVE.get() - before_checkpoint;
    assert_eq!(checkpoint_allocation, 264 * 8 * 2);
    let clone = checkpoint.clone();
    assert_eq!(LIVE.get() - before_checkpoint, 2 * checkpoint_allocation);
    drop(clone);
    let result = (PEAK.get(), checkpoint.storage_bytes());
    drop(checkpoint);
    drop(band);
    drop(rows);
    assert_eq!(LIVE.get(), 0);
    TRACK.set(false);
    result
}

#[test]
fn height_does_not_grow_row_or_checkpoint_allocation() {
    let short = measured(17);
    let tall = measured(65000);
    assert_eq!(short, tall);
    // One i16 band + one u8 output band + one i16 checkpoint band,
    // plus one cloned checkpoint, bounded headers, and std input staging.
    let staging = if cfg!(feature = "std") { 8192 } else { 0 };
    assert!(short.0 <= 56 * 264 + 2048 + staging, "peak={}", short.0);
    assert!(short.1 <= 16 * 264 + 256, "checkpoint={}", short.1);
    println!(
        "width=257 heights=17,65000 peak={} checkpoint={}",
        short.0, short.1
    );
}
