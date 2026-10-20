//! A progress stops whole-image and row decoding within a bounded amount of work.

use std::cell::Cell;

use zune_jpeg::progress::{Progress, ProgressCursor};
use zune_jpeg::rows::{GrayscaleRows, RowDecodeError};
use zune_jpeg::zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
use zune_jpeg::{errors::DecodeErrors, JpegDecoder};

/// Stops once `limit` work is reported; records the work reported until then.
struct StopAt {
    work: Cell<usize>,
    limit: usize,
}

impl Progress for StopAt {
    fn advance(&self, work: usize) -> bool {
        if self.work.get() < self.limit {
            self.work.set(self.work.get() + work);
        }
        self.work.get() >= self.limit
    }
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/row-fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// `bytes` with the frame declared `width` × `height`.
fn resized(mut bytes: Vec<u8>, width: u16, height: u16) -> Vec<u8> {
    let sof = bytes
        .windows(2)
        .position(|p| p[0] == 0xff && matches!(p[1], 0xc0 | 0xc2))
        .unwrap();
    bytes[sof + 5..sof + 7].copy_from_slice(&height.to_be_bytes());
    bytes[sof + 7..sof + 9].copy_from_slice(&width.to_be_bytes());
    bytes
}

fn options() -> DecoderOptions {
    DecoderOptions::default()
        .set_max_width(65535)
        .set_max_height(65535)
}

/// Images whose data ends long before their declared rows, baseline (padded with
/// gray) and progressive (every row transformed), stop when their progress does, and
/// decode as without one when it does not stop.
#[test]
fn whole_decoding_stops_with_its_progress() {
    for name in ["33x41-r3.jpg", "progressive.jpg", "color.jpg"] {
        let bytes = resized(fixture(name), 4000, 4000);
        let limit = 1 << 20;
        let progress = StopAt {
            work: Cell::new(0),
            limit,
        };
        let mut decoder =
            JpegDecoder::new_with_options(ProgressCursor::new(&bytes, &progress), options());
        let result = decoder.decode_with_progress(&progress);
        assert!(
            matches!(result, Err(DecodeErrors::Cancelled)),
            "{name}: {result:?}"
        );
        // The last report is at most a row of MCUs of 4000 pixels in three components.
        assert!(progress.work.get() < limit + 4000 * 16 * 3 * 4, "{name}");

        let unstopped = JpegDecoder::new_with_options(ZCursor::new(&bytes), options()).decode();
        let none = StopAt {
            work: Cell::new(0),
            limit: usize::MAX,
        };
        let reported = JpegDecoder::new_with_options(ProgressCursor::new(&bytes, &none), options())
            .decode_with_progress(&none);
        assert_eq!(reported.ok(), unstopped.ok(), "{name}");
    }
}

/// A reader stops in the headers, in the single-scan check and in a band once its
/// progress does, then reads again only from a checkpoint.
#[test]
fn row_decoding_stops_with_its_progress() {
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::Luma);
    let bytes = fixture("257x17-r7.jpg");
    let mut whole = GrayscaleRows::new(bytes.as_slice(), options, &()).unwrap();
    let mut band = vec![0; whole.output_buffer_size()];
    whole.read_mcu_row(&mut band).unwrap();
    let base_work = {
        let counted = StopAt {
            work: Cell::new(0),
            limit: usize::MAX,
        };
        let mut rows = GrayscaleRows::new(bytes.as_slice(), options, &counted).unwrap();
        rows.read_mcu_row(&mut band).unwrap();
        counted.work.get()
    };
    assert!(base_work > 0);
    for limit in [1, base_work / 2, base_work] {
        let progress = StopAt {
            work: Cell::new(0),
            limit,
        };
        let opened = GrayscaleRows::new(bytes.as_slice(), options, &progress);
        let Ok(mut rows) = opened else {
            assert!(matches!(opened, Err(RowDecodeError::Stopped)), "{limit}");
            continue;
        };
        let checkpoint = rows.checkpoint().unwrap();
        assert!(
            matches!(rows.read_mcu_row(&mut band), Err(RowDecodeError::Stopped)),
            "{limit}"
        );
        assert!(matches!(
            rows.read_mcu_row(&mut band),
            Err(RowDecodeError::Stopped)
        ));
        let mut resumed = checkpoint.resume(&()).unwrap();
        let mut again = vec![0; resumed.output_buffer_size()];
        resumed.read_mcu_row(&mut again).unwrap();
        let mut expected = vec![0; again.len()];
        GrayscaleRows::new(bytes.as_slice(), options, &())
            .unwrap()
            .read_mcu_row(&mut expected)
            .unwrap();
        assert_eq!(again, expected);
    }
}
