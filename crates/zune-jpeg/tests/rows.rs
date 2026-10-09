use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use zune_jpeg::zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
use zune_jpeg::{
    rows::{GrayscaleRows, RowDecodeError},
    JpegDecoder,
};

fn options(strict: bool) -> DecoderOptions {
    DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::Luma)
        .set_strict_mode(strict)
}
fn full(bytes: &[u8], opts: DecoderOptions) -> Result<Vec<u8>, String> {
    JpegDecoder::new_with_options(ZCursor::new(bytes), opts)
        .decode()
        .map_err(|e| format!("{e}"))
}
fn drain<S: AsRef<[u8]>>(rows: &mut GrayscaleRows<S>) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let mut band = vec![0; rows.output_buffer_size()];
        if rows.read_mcu_row(&mut band).map_err(|e| format!("{e}"))? == 0 {
            break;
        }
        out.extend_from_slice(&band);
    }
    Ok(out)
}
fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/row-fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

#[test]
fn exact_edges_and_restart_intervals() {
    let mut names = Vec::new();
    for w in [1, 7, 8, 9] {
        for h in [1, 7, 8, 9] {
            names.push(format!("{w}x{h}.jpg"));
        }
    }
    names.extend(["33x41-r3.jpg".into(), "257x17-r7.jpg".into()]);
    for name in names {
        for unsafe_simd in [false, true] {
            let bytes = fixture(&name);
            let opts = options(false).set_use_unsafe(unsafe_simd);
            let expected = full(&bytes, opts).unwrap();
            let mut rows = GrayscaleRows::new(bytes, opts).unwrap();
            assert_eq!(
                drain(&mut rows).unwrap(),
                expected,
                "{name} SIMD={unsafe_simd}"
            );
            assert_eq!(rows.next_row(), rows.dimensions().1);
        }
    }
}

#[test]
fn native_baseline_fingerprints_do_not_change() {
    for (name, hash) in [
        ("1x1.jpg", 0xaf63bd4c8601b7df),
        ("7x9.jpg", 0x8acaccf437b7b2a8),
        ("8x8.jpg", 0x9d190eb5949928fd),
        ("9x7.jpg", 0xd6aaa62c523e4448),
        ("33x41-r3.jpg", 0xa976393c9d8d937b),
        ("257x17-r7.jpg", 0x958b6ee30460c0ab),
    ] {
        let actual = full(&fixture(name), options(false))
            .unwrap()
            .iter()
            .fold(0xcbf29ce484222325u64, |h, b| {
                (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
            });
        assert_eq!(actual, hash, "{name}");
    }
}

#[test]
fn checkpoints_restore_prefetch_dc_restart_and_prior_band() {
    let bytes = fixture("33x41-r3.jpg"); // Five MCUs/row, restart every three.
    let expected = full(&bytes, options(false)).unwrap();
    let mut rows = GrayscaleRows::new(bytes, options(false)).unwrap();
    let mut checkpoints = vec![rows.checkpoint().unwrap()];
    while rows.output_buffer_size() != 0 {
        rows.read_mcu_row(&mut vec![0; rows.output_buffer_size()])
            .unwrap();
        checkpoints.push(rows.checkpoint().unwrap());
    }
    for &index in &[3, 1, 5, 0, 2, 6, 4, 1] {
        let checkpoint = &checkpoints[index];
        rows.restore(checkpoint).unwrap();
        assert_eq!(
            drain(&mut rows).unwrap(),
            expected[checkpoint.next_row() * 33..]
        );
    }
    drop(rows);
    let checkpoint = checkpoints[2].clone();
    drop(checkpoints);
    let mut a = checkpoint.resume().unwrap();
    let mut b = checkpoint.resume().unwrap();
    drop(checkpoint);
    a.read_mcu_row(&mut vec![0; a.output_buffer_size()])
        .unwrap();
    assert_eq!(drain(&mut b).unwrap(), expected[16 * 33..]);
    assert_eq!(drain(&mut a).unwrap(), expected[24 * 33..]);
}

#[test]
fn truncation_matches_full_decoder_and_resumes() {
    let bytes = fixture("33x41-r3.jpg");
    let sos = bytes.windows(2).position(|p| p == [255, 218]).unwrap();
    let first = sos + 2 + usize::from(u16::from_be_bytes([bytes[sos + 2], bytes[sos + 3]]));
    for end in first..=bytes.len() {
        for strict in [false, true] {
            let input = &bytes[..end];
            let expected = full(input, options(strict));
            let mut rows = GrayscaleRows::new(input, options(strict)).unwrap();
            let actual = drain(&mut rows);
            assert_eq!(actual, expected, "end={end} strict={strict}");
            if !strict {
                let mut rows = GrayscaleRows::new(input, options(false)).unwrap();
                while rows.output_buffer_size() != 0 {
                    let checkpoint = rows.checkpoint().unwrap();
                    let mut resumed = checkpoint.resume().unwrap();
                    let mut a = vec![0; rows.output_buffer_size()];
                    let mut b = vec![0; resumed.output_buffer_size()];
                    let ar = rows.read_mcu_row(&mut a).map_err(|e| format!("{e}"));
                    let br = resumed.read_mcu_row(&mut b).map_err(|e| format!("{e}"));
                    assert_eq!(ar, br);
                    assert_eq!(a, b);
                    if ar.is_err() {
                        break;
                    }
                }
            }
        }
    }
}

#[test]
fn reject_unsupported_foreign_and_failed_states() {
    for name in ["color.jpg", "progressive.jpg"] {
        assert!(matches!(
            GrayscaleRows::new(fixture(name), options(false)),
            Err(RowDecodeError::Unsupported)
        ));
    }
    let bytes = fixture("33x41-r3.jpg");
    let mut a = GrayscaleRows::new(bytes.clone(), options(false)).unwrap();
    let mut b = GrayscaleRows::new(bytes.clone(), options(false)).unwrap();
    let checkpoint = a.checkpoint().unwrap();
    assert!(matches!(
        b.restore(&checkpoint),
        Err(RowDecodeError::ForeignCheckpoint)
    ));
    let mut too_small = [0; 1];
    assert!(a.read_mcu_row(&mut too_small).is_err());
    assert_eq!(a.next_row(), 0);
    assert_eq!(
        drain(&mut a).unwrap(),
        full(&bytes, options(false)).unwrap()
    );
    let truncated = &bytes[..bytes.len() / 2];
    let mut failed = GrayscaleRows::new(truncated, options(true)).unwrap();
    let checkpoint = failed.checkpoint().unwrap();
    assert!(drain(&mut failed).is_err());
    assert!(matches!(failed.checkpoint(), Err(RowDecodeError::Failed)));
    assert!(matches!(
        failed.read_mcu_row(&mut []),
        Err(RowDecodeError::Failed)
    ));
    failed.restore(&checkpoint).unwrap();
    assert_eq!(failed.next_row(), 0);
    assert!(drain(&mut failed).is_err());
}

#[test]
fn source_lease_is_not_cloned_or_dropped_while_snapshots_exist() {
    struct Lease {
        bytes: Vec<u8>,
        drops: Arc<AtomicUsize>,
    }
    impl AsRef<[u8]> for Lease {
        fn as_ref(&self) -> &[u8] {
            &self.bytes
        }
    }
    impl Drop for Lease {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let mut rows = GrayscaleRows::new(
        Lease {
            bytes: fixture("33x41-r3.jpg"),
            drops: drops.clone(),
        },
        options(false),
    )
    .unwrap();
    let checkpoint = rows.checkpoint().unwrap();
    drop(rows);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let mut resumed = checkpoint.resume().unwrap();
    drop(checkpoint);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(drain(&mut resumed).unwrap().len(), 33 * 41);
    drop(resumed);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn scan_and_table_markers_are_rejected_before_any_pixels() {
    let original = fixture("33x41-r3.jpg");
    let restarts: Vec<usize> = original
        .windows(2)
        .enumerate()
        .filter(|(_, b)| b[0] == 255 && (208..=215).contains(&b[1]))
        .map(|(i, _)| i)
        .collect();
    for restart in restarts {
        for marker in [0xda, 0xdb, 0xc4, 0xdd, 0xfe, 0xe0] {
            let mut bytes = original.clone();
            bytes[restart + 1] = marker;
            assert!(matches!(
                GrayscaleRows::new(bytes, options(false)),
                Err(RowDecodeError::Unsupported)
            ));
        }
    }
    let expected = full(&original, options(false)).unwrap();
    let mut concatenated = original.clone();
    concatenated.extend_from_slice(&fixture("progressive.jpg"));
    let mut rows = GrayscaleRows::new(concatenated, options(false)).unwrap();
    assert_eq!(drain(&mut rows).unwrap(), expected);
}

#[test]
fn independent_checkpoint_readers_run_on_different_threads() {
    let bytes = fixture("33x41-r3.jpg");
    let expected = full(&bytes, options(false)).unwrap();
    let mut rows = GrayscaleRows::new(bytes, options(false)).unwrap();
    rows.read_mcu_row(&mut vec![0; rows.output_buffer_size()])
        .unwrap();
    let checkpoint = rows.checkpoint().unwrap();
    let other = checkpoint.clone();
    drop(rows);
    let first = std::thread::spawn(move || drain(&mut checkpoint.resume().unwrap()).unwrap());
    let second = std::thread::spawn(move || drain(&mut other.resume().unwrap()).unwrap());
    assert_eq!(first.join().unwrap(), expected[8 * 33..]);
    assert_eq!(second.join().unwrap(), expected[8 * 33..]);
}
