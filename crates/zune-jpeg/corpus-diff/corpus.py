# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf==1.28.2", "numpy"]
# ///
"""Checks a zune-jpeg change on the JPEGs of a PDF corpus (hayro-tests/downloads).

uv run corpus.py extract <pdf dir> <jpeg dir>
    Writes each distinct DCTDecode stream (filters before it decoded when they are
    FlateDecode) as <hash>.jpg, and index.tsv: hash, PDF, xref.
cargo run --release -- <jpeg dir> <dump dir> > diff.tsv
    Lists the streams the old and the new revision decode differently.
uv run corpus.py mupdf <jpeg dir> diff.tsv <dump dir> <pdf dir>
    For each whole-image difference, the mean distance of the old and the new samples
    to MuPDF's decode of the image, over the samples where old and new differ.
"""
import hashlib, pathlib, sys, zlib
import numpy as np, pymupdf

def filters(doc, xref):
    kind, value = doc.xref_get_key(xref, "Filter")
    if kind == "name":
        return [value]
    if kind == "array":
        return value.strip("[]").split()
    return []

def extract(pdfs, out):
    out.mkdir(parents=True, exist_ok=True)
    index, skipped, failed = {}, {}, []
    for pdf in sorted(pdfs.rglob("*.pdf")):
        try:
            doc = pymupdf.open(pdf)
            if doc.needs_pass:
                failed.append(f"{pdf}: password")
                continue
            for xref in range(1, doc.xref_length()):
                fs = filters(doc, xref) if doc.xref_is_stream(xref) else []
                if not fs or fs[-1] not in ("/DCTDecode", "/DCT"):
                    continue
                data = doc.xref_stream_raw(xref)
                for f in fs[:-1]:
                    if f not in ("/FlateDecode", "/Fl"):
                        skipped[f] = skipped.get(f, 0) + 1
                        data = None
                        break
                    data = zlib.decompressobj().decompress(data)
                if not data:
                    continue
                h = hashlib.sha1(data).hexdigest()[:16]
                if h not in index:
                    (out / f"{h}.jpg").write_bytes(data)
                    index[h] = (pdf.relative_to(pdfs), xref)
        except Exception as e:
            failed.append(f"{pdf}: {e}")
    (out / "index.tsv").write_text("".join(f"{h}\t{p}\t{x}\n" for h, (p, x) in index.items()))
    print(f"{len(index)} distinct streams; skipped for other filters {skipped}; {len(failed)} PDFs not read")
    for f in failed:
        print("  ", f)

def mupdf(jpegs, listing, dump, pdfs):
    index = {h: (p, int(x)) for h, p, x in (l.split("\t") for l in (jpegs / "index.tsv").read_text().splitlines())}
    for line in listing.read_text().splitlines():
        name, path, summary, detail = line.split("\t")
        if path != "whole":
            continue
        pdf, xref = index[name]
        old, new = dump / f"{name}.whole.old", dump / f"{name}.whole.new"
        verdict = ""
        if old.exists() and new.exists():
            o, n = (np.frombuffer(p.read_bytes(), np.uint8).astype(np.int16) for p in (old, new))
            pix = pymupdf.Pixmap(pymupdf.open(pdfs / pdf), xref)
            if pix.alpha:
                pix = pymupdf.Pixmap(pix, 0)
            m = np.frombuffer(pix.samples, np.uint8).astype(np.int16)
            if len(m) == len(o):
                ch = o != n
                do, dn = np.abs(o[ch] - m[ch]).mean(), np.abs(n[ch] - m[ch]).mean()
                verdict = f"to MuPDF old {do:.2f} new {dn:.2f}; whole image old {np.abs(o - m).mean():.3f} new {np.abs(n - m).mean():.3f}"
            else:
                verdict = f"MuPDF gives {pix.n} channels {pix.w}x{pix.h}"
        print(f"{name}\t{pdf}\txref {xref}\t{summary}\t{detail}\t{verdict}")

if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    else:
        mupdf(*(pathlib.Path(a) for a in sys.argv[2:6]))
