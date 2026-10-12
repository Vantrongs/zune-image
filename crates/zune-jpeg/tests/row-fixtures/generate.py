"""Regenerate deterministic synthetic fixtures with libjpeg-turbo cjpeg.

Usage: python3 generate.py [cjpeg executable]
The checked-in bytes were generated with libjpeg-turbo 3.1.4, quality 91.
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).parent
encoder = sys.argv[1] if len(sys.argv) > 1 else "cjpeg"
for width, height, restart in [
    *((w, h, 0) for w in (1, 7, 8, 9) for h in (1, 7, 8, 9)),
    (33, 41, 3),
    (257, 17, 7),
]:
    pixels = bytes(
        (x * 17 + y * 13 + (x * y) % 37) & 255
        for y in range(height)
        for x in range(width)
    )
    pgm = f"P5\n{width} {height}\n255\n".encode() + pixels
    flags = ["-quality", "91", "-grayscale"]
    suffix = f"-r{restart}" if restart else ""
    if restart:
        flags += ["-restart", f"{restart}B"]
    result = subprocess.run([encoder, *flags], input=pgm, capture_output=True, check=True)
    (root / f"{width}x{height}{suffix}.jpg").write_bytes(result.stdout)

for name, pgm, flags in [
    ("progressive", b"P5\n9 9\n255\n" + bytes(range(81)), ["-progressive"]),
    ("color", b"P6\n9 9\n255\n" + bytes(range(243)), []),
]:
    result = subprocess.run([encoder, *flags], input=pgm, capture_output=True, check=True)
    (root / f"{name}.jpg").write_bytes(result.stdout)
