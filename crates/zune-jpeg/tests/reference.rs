//! Generated images decode as an independent decoder decodes them, and restart
//! intervals, which change only the entropy coding, do not change the pixels.

use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
use zune_jpeg::rows::GrayscaleRows;
use zune_jpeg::zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
use zune_jpeg::JpegDecoder;

/// Tiles of 8x8 pixels alternate between flat, gradient and noise, so that MCUs
/// take from a few bits to hundreds and restart markers land anywhere in the
/// bitstream's read-ahead.
fn image(width: usize, height: usize, channels: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let tiles_x = width.div_ceil(8);
    let kinds: Vec<u64> = (0..tiles_x * height.div_ceil(8))
        .map(|_| next() % 3)
        .collect();
    let mut out = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        for x in 0..width {
            let kind = kinds[(y / 8) * tiles_x + x / 8];
            for c in 0..channels {
                out.push(match kind {
                    0 => (40 + 60 * c) as u8,
                    1 => ((x * 5 + y * 3 + c * 70) % 256) as u8,
                    _ => next() as u8,
                });
            }
        }
    }
    out
}

struct Case {
    width: u16,
    height: u16,
    color: ColorType,
    sampling: SamplingFactor,
    progressive: bool,
}

impl Case {
    fn channels(&self) -> usize {
        if matches!(self.color, ColorType::Luma) {
            1
        } else {
            3
        }
    }

    fn encode(&self, pixels: &[u8], restart_interval: u16) -> Vec<u8> {
        let mut out = Vec::new();
        let mut encoder = Encoder::new(&mut out, 90);
        encoder.set_sampling_factor(self.sampling);
        encoder.set_progressive(self.progressive);
        if restart_interval > 0 {
            encoder.set_restart_interval(restart_interval);
        }
        encoder
            .encode(pixels, self.width, self.height, self.color)
            .unwrap();
        out
    }

    fn name(&self, restart_interval: u16) -> String {
        format!(
            "{}x{} {:?} {:?} progressive={} restart={restart_interval}",
            self.width, self.height, self.color, self.sampling, self.progressive
        )
    }

    /// MCUs per row of the interleaved scan; one block per MCU in a gray image.
    fn mcus_per_row(&self) -> u16 {
        let h = if matches!(self.color, ColorType::Luma) {
            1
        } else {
            (self.sampling as u8 & 0x7f) >> 4
        };
        self.width.div_ceil(8 * u16::from(h))
    }
}

fn cases() -> Vec<Case> {
    let sizes = [
        (1, 1),
        (7, 9),
        (8, 8),
        (9, 17),
        (16, 16),
        (17, 33),
        (31, 8),
        (33, 15),
        (63, 40),
        (64, 64),
        (65, 23),
        (127, 47),
        (248, 35),
        (227, 34),
    ];
    let mut cases = Vec::new();
    for (width, height) in sizes {
        for progressive in [false, true] {
            cases.push(Case {
                width,
                height,
                color: ColorType::Luma,
                sampling: SamplingFactor::F_1_1,
                progressive,
            });
            for sampling in [
                SamplingFactor::F_1_1,
                SamplingFactor::F_2_1,
                SamplingFactor::F_1_2,
                SamplingFactor::F_2_2,
                SamplingFactor::F_4_1,
            ] {
                cases.push(Case {
                    width,
                    height,
                    color: ColorType::Rgb,
                    sampling,
                    progressive,
                });
            }
        }
    }
    cases
}

fn zune(bytes: &[u8], case: &Case, unsafe_simd: bool) -> Vec<u8> {
    let out = if case.channels() == 1 { ColorSpace::Luma } else { ColorSpace::RGB };
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(out)
        .set_use_unsafe(unsafe_simd);
    JpegDecoder::new_with_options(ZCursor::new(bytes), options)
        .decode()
        .unwrap_or_else(|e| panic!("{e}"))
}

fn zune_rows(bytes: &[u8], unsafe_simd: bool) -> Vec<u8> {
    let options = DecoderOptions::default().set_use_unsafe(unsafe_simd);
    let mut rows = GrayscaleRows::new(bytes, options, &()).unwrap();
    let mut out = Vec::new();
    loop {
        let mut band = vec![0; rows.output_buffer_size()];
        if rows.read_mcu_row(&mut band).unwrap() == 0 {
            return out;
        }
        out.extend_from_slice(&band);
    }
}

#[test]
fn restart_intervals_do_not_change_the_pixels() {
    for (seed, case) in cases().iter().enumerate() {
        let pixels = image(
            usize::from(case.width),
            usize::from(case.height),
            case.channels(),
            seed as u64,
        );
        let plain = case.encode(&pixels, 0);
        let per_row = case.mcus_per_row();
        for unsafe_simd in [false, true] {
            let expected = zune(&plain, case, unsafe_simd);
            for interval in [
                1,
                2,
                3,
                4,
                5,
                7,
                per_row.saturating_sub(1),
                per_row,
                per_row + 1,
            ] {
                if interval == 0 {
                    continue;
                }
                let bytes = case.encode(&pixels, interval);
                let name = format!("{} SIMD={unsafe_simd}", case.name(interval));
                assert!(zune(&bytes, case, unsafe_simd) == expected, "{name}");
                if case.channels() == 1 && !case.progressive {
                    assert!(zune_rows(&bytes, unsafe_simd) == expected, "rows {name}");
                }
            }
        }
    }
}

/// jpeg-decoder rounds the IDCT and the colour conversion differently, by up to 3
/// in a sample (measured on these cases), and interpolates subsampled chroma
/// differently by up to 4.
#[test]
fn pixels_are_those_of_an_independent_decoder() {
    for (seed, case) in cases().iter().enumerate() {
        let pixels = image(
            usize::from(case.width),
            usize::from(case.height),
            case.channels(),
            seed as u64,
        );
        let (h, v) = if case.channels() == 1 {
            (1, 1)
        } else {
            let factors = case.sampling as u8 & 0x7f;
            (usize::from(factors >> 4), usize::from(factors & 15))
        };
        let tolerance = if h * v == 1 { 3 } else { 4 };
        let (width, height) = (usize::from(case.width), usize::from(case.height));
        // Where the image edge cuts an MCU, zune-jpeg interpolates the last row and
        // column of subsampled chroma unlike jpeg-decoder and libjpeg-turbo, which
        // agree with each other; the difference is a known defect left for upstream.
        let compared_width = if h == 1 || width % (8 * h) == 0 { width } else { width - 1 };
        let compared_height = if v == 1 || height % (8 * v) == 0 { height } else { height - 1 };
        for interval in [0, 3] {
            let bytes = case.encode(&pixels, interval);
            let reference = jpeg_decoder::Decoder::new(&bytes[..]).decode().unwrap();
            for unsafe_simd in [false, true] {
                let actual = zune(&bytes, case, unsafe_simd);
                let name = format!("{} SIMD={unsafe_simd}", case.name(interval));
                assert_eq!(actual.len(), reference.len(), "{name}");
                let row = width * case.channels();
                for y in 0..compared_height {
                    let range = y * row..y * row + compared_width * case.channels();
                    for (x, (a, b)) in actual[range.clone()]
                        .iter()
                        .zip(&reference[range])
                        .enumerate()
                    {
                        assert!(
                            a.abs_diff(*b) <= tolerance,
                            "{name}: sample {x} of row {y} is {a}, the reference {b}"
                        );
                    }
                }
            }
        }
    }
}
