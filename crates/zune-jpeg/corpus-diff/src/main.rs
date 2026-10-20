//! `corpus-diff <jpeg dir> <dump dir>`: decodes each `*.jpg` with the `old` and `new`
//! zune-jpeg as hayro does (whole image, and grayscale rows), prints one line per file
//! whose output differs, and writes both outputs of those files to the dump directory.

use std::path::Path;

/// What one revision gives for a file: the colour space, dimensions and samples,
/// or the error.
type Decoded = Result<(String, usize, usize, Vec<u8>), String>;

macro_rules! decoder {
    ($name:ident, $zune:ident) => {
        mod $name {
            use $zune::progress::ProgressCursor;
            use $zune::zune_core::colorspace::ColorSpace;
            use $zune::zune_core::options::DecoderOptions;
            use $zune::JpegDecoder;

            /// hayro-syntax's `filter::dct::decode` output colour space choice.
            pub fn whole(bytes: &[u8]) -> super::Decoded {
                let options = DecoderOptions::default()
                    .set_max_width(usize::from(u16::MAX))
                    .set_max_height(usize::from(u16::MAX));
                let mut d = JpegDecoder::new_with_options(ProgressCursor::new(bytes, &()), options);
                d.decode_headers().map_err(|e| format!("{e:?}"))?;
                let input = d.input_colorspace().unwrap();
                let components = d.info().unwrap().components;
                // hayro asks for as many bands as the image's PDF colour space has
                // when that is not 1, 3 or 4; a valid PDF gives the JPEG's count.
                let mut out = if !matches!(components, 1 | 3 | 4) {
                    ColorSpace::MultiBand(core::num::NonZeroU32::new(components.into()).unwrap())
                } else {
                    match input {
                        ColorSpace::Luma | ColorSpace::LumaA => ColorSpace::Luma,
                        ColorSpace::CMYK => ColorSpace::CMYK,
                        ColorSpace::YCCK => ColorSpace::YCCK,
                        _ => ColorSpace::RGB,
                    }
                };
                if input == ColorSpace::CMYK && components == 3 {
                    out = ColorSpace::RGB;
                }
                d.set_options(options.jpeg_set_out_colorspace(out));
                let pixels = d.decode_with_progress(&()).map_err(|e| format!("{e:?}"))?;
                let (w, h) = d.dimensions().unwrap();
                Ok((format!("{out:?}"), w, h, pixels))
            }

            /// hayro's grayscale row reader, for the images it takes.
            pub fn rows(bytes: &[u8]) -> Option<super::Decoded> {
                let options = DecoderOptions::default()
                    .set_max_width(usize::from(u16::MAX))
                    .set_max_height(usize::from(u16::MAX));
                let mut rows = match $zune::rows::GrayscaleRows::new(bytes, options, &()) {
                    Ok(rows) => rows,
                    Err($zune::rows::RowDecodeError::Unsupported) => return None,
                    Err(e) => return Some(Err(format!("{e}"))),
                };
                let (w, h) = rows.dimensions();
                let mut out = Vec::with_capacity(w * h);
                loop {
                    let mut band = vec![0; rows.output_buffer_size()];
                    match rows.read_mcu_row(&mut band) {
                        Ok(0) => return Some(Ok(("Luma".into(), w, h, out))),
                        Ok(_) => out.extend_from_slice(&band),
                        Err(e) => return Some(Err(format!("{e}"))),
                    }
                }
            }
        }
    };
}
decoder!(old_rev, old);
decoder!(new_rev, new);

fn summary(d: &Decoded) -> String {
    match d {
        Ok((cs, w, h, _)) => format!("{cs} {w}x{h}"),
        Err(e) => format!("error {e}"),
    }
}

/// Prints the difference, if any, and dumps both outputs.
fn compare(name: &str, path: &str, old: &Decoded, new: &Decoded, dump: &Path) -> bool {
    if old == new {
        return false;
    }
    let detail = match (old, new) {
        (Ok((_, _, _, a)), Ok((_, _, _, b))) if a.len() == b.len() => {
            let diffs: Vec<u8> = a
                .iter()
                .zip(b)
                .map(|(x, y)| x.abs_diff(*y))
                .filter(|d| *d > 0)
                .collect();
            format!(
                "{} samples differ, max {}",
                diffs.len(),
                diffs.iter().max().unwrap()
            )
        }
        _ => format!("{} -> {}", summary(old), summary(new)),
    };
    println!("{name}\t{path}\t{}\t{detail}", summary(new));
    for (which, d) in [("old", old), ("new", new)] {
        if let Ok((_, _, _, px)) = d {
            std::fs::write(dump.join(format!("{name}.{path}.{which}")), px).unwrap();
        }
    }
    true
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dir, dump) = (Path::new(&args[1]), Path::new(&args[2]));
    std::fs::create_dir_all(dump).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "jpg"))
        .collect();
    files.sort();
    let (mut differ, mut rows_read) = (0, 0);
    for file in &files {
        let bytes = std::fs::read(file).unwrap();
        let name = file.file_stem().unwrap().to_str().unwrap();
        let (old, new) = (old_rev::whole(&bytes), new_rev::whole(&bytes));
        let mut changed = compare(name, "whole", &old, &new, dump);
        match (old_rev::rows(&bytes), new_rev::rows(&bytes)) {
            (None, None) => {}
            (old_rows, new_rows) => {
                rows_read += 1;
                let gone = || Err("not read as rows".to_string());
                let (old_rows, new_rows) =
                    (old_rows.unwrap_or_else(gone), new_rows.unwrap_or_else(gone));
                changed |= compare(name, "rows", &old_rows, &new_rows, dump);
                // Rows must equal the whole decode of the same revision.
                if let (Ok(r), Ok(w)) = (&new_rows, &new) {
                    if r.3 != w.3 {
                        println!(
                            "{name}\trows-vs-whole\t{}\tnew rows differ from new whole",
                            summary(&new)
                        );
                        changed = true;
                    }
                }
            }
        }
        differ += usize::from(changed);
    }
    eprintln!(
        "{} files, {rows_read} also read as rows, {differ} differ",
        files.len()
    );
}
