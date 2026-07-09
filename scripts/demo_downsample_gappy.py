#!/usr/bin/env python
"""
Show that downsample_gappy() is EXACTLY equivalent to zero-padding a gappy
series and running downsample() on the whole padded array.

downsample_gappy downsamples each segment individually (padded with a handful
of zeros so its windows align with the global downsample grid) and returns the
per-segment pieces plus their global offsets, skipping the gaps entirely.
Adding the pieces into a zeros array of the downsampled length must reproduce
downsample(zero_padded) exactly. Every comparison below is checked with
np.array_equal, i.e. bitwise equality of the float32 outputs.

Run from anywhere (riptide must be importable):

    python scripts/demo_downsample_gappy.py
    python scripts/demo_downsample_gappy.py --trials 500 --seed 1
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "src")  # allow running from the repo root without install

from riptide.periodogram_py import F32, downsample, downsample_gappy


def zero_padded(data_list, gaps):
    """Assemble the full series: seg0 | zeros(gap0) | seg1 | ... as float32."""
    sizes = [d.size for d in data_list]
    size = sum(sizes) + sum(gaps)
    data = np.zeros(size, dtype=F32)
    for i, seg in enumerate(data_list):
        start = sum(sizes[:i]) + sum(gaps[:i])
        data[start:start + sizes[i]] = seg
    return data


def assemble(ds_segs, ds_starts, n):
    """Add the per-segment downsampled pieces into a zeros array of length n."""
    out = np.zeros(n, dtype=F32)
    for arr, k0 in zip(ds_segs, ds_starts):
        out[k0:k0 + arr.size] += arr
    return out


def compare(segs, gaps, f, label=""):
    """Run both paths; return True iff outputs are bitwise identical."""
    full = zero_padded(segs, gaps)

    t0 = time.perf_counter()
    ref = full if f == 1 else downsample(full, f)
    t1 = time.perf_counter()
    got = assemble(*downsample_gappy(segs, gaps, f), ref.size)
    t2 = time.perf_counter()

    equal = np.array_equal(ref, got)
    status = "EXACT (bitwise)" if equal else \
        f"DIFFER  max|diff| = {np.abs(ref - got).max():.3e}"
    print(f"  {label:<38s} f = {f:<12g} n = {ref.size:<8d} {status}"
          f"   [padded: {t1 - t0:6.2f} s, gappy: {t2 - t1:6.3f} s]")
    return equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200,
                        help="Number of randomized fuzz trials.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)

    ok = True

    print("Hand-picked cases (segment sizes, gap sizes, downsampling factor):")
    cases = [
        ([3000, 3000], [30000], 1, "f = 1 (direct assembly)"),
        ([3000, 3000], [30000], 1.25, "2 segments, big gap"),
        ([3000, 3000], [30000], 2.083333333, "fractional f"),
        ([3000, 3000], [30000], 4.7, "larger fractional f"),
        ([3000, 2999, 3001], [30001, 20007], 1.9375, "3 segments, odd sizes"),
        ([777, 1234, 555, 999], [4321, 8765, 2222], 3.3, "4 unequal segments"),
        ([100, 100], [50], 7.9, "gap only a few windows wide"),
        ([64, 64], [8], 6.5, "gap barely wider than f"),
        ([200, 200], [3], 6.5, "gap SMALLER than f (straddling window)"),
    ]
    for sizes, gaps, f, label in cases:
        segs = [rng.randn(s).astype(F32) for s in sizes]
        ok &= compare(segs, gaps, f, label)

    print(f"\nRandomized fuzz ({args.trials} trials: 2-4 segments, "
          "sizes 50-400, gaps 60-5000, f in [1.01, 12]):")
    tested = failures = 0
    for _ in range(args.trials):
        nseg = rng.randint(2, 5)
        sizes = list(rng.randint(50, 400, size=nseg))
        gaps = list(rng.randint(60, 5000, size=nseg - 1))
        f = float(rng.uniform(1.01, 12.0))
        segs = [rng.randn(s).astype(F32) for s in sizes]
        full = zero_padded(segs, gaps)
        if f > full.size:
            continue
        tested += 1
        ref = downsample(full, f)
        got = assemble(*downsample_gappy(segs, gaps, f), ref.size)
        if not np.array_equal(ref, got):
            failures += 1
            ok = False
            print(f"  FAIL: sizes={sizes} gaps={gaps} f={f} "
                  f"max|diff|={np.abs(ref - got).max():.3e}")
    print(f"  {tested} trials run, {failures} failures")

    print("\n=> downsample_gappy is bitwise identical to "
          "downsample(zero-padded array)" if ok else
          "\n=> MISMATCH FOUND: outputs are not identical")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
