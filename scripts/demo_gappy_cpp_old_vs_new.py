#!/usr/bin/env python
"""
Minimal comparison: gappy FFA search with the NEW segment-wise C++ kernel vs
the OLD zero-padding C++ kernel.

Both backends run ffa_search_gappy on the SAME TimeSeriesGappy, so the only
thing that differs is the periodogram kernel (libcpp.periodogram_gappy vs
libcpp.periodogram_gappy_old):

  - "cpp_old" FFA-transforms the whole zero-padded series as one block
    (skipping all-zero gap sub-blocks) -- the zero-pad ground truth.
  - "cpp" downsamples and FFA-transforms each segment on its own and merges
    the transforms across the gaps; the padded series is never materialised.

The two are NOT expected to agree bitwise: the FFA shift of each input row is
quantised through the merge tree, and the segment-wise tree differs from the
full binary tree. Both stay within +/- 1 phase bin of the ideal shift, so the
peaks should coincide, with off-peak S/N differing by a few tenths at most.
The new kernel should also be substantially faster, since it never touches
the gaps.

Run from anywhere (riptide must be importable):

    python scripts/demo_gappy_cpp_old_vs_new.py
    python scripts/demo_gappy_cpp_old_vs_new.py --plot gappy_cpp_old_vs_new.png
"""
import argparse
import sys
import time

import numpy as np

from riptide import TimeSeries, TimeSeriesGappy, ffa_search_gappy

sys.setrecursionlimit(10000)

# True period of the injected signal (seconds), and the +/- window searched
# around it.
TRUE_PERIOD = 4.0
PERIOD_HALF_WINDOW = 1.0


def timed(label, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), print how long it took, and return its result."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  [timing] {label}: {(time.perf_counter() - t0) * 1e3:.3f} ms")
    return result


def make_gappy(nseg=20, seg_length=30.0, tsamp=0.01, base_gap=300.0):
    """Build ONE long, phase-coherent series spanning the whole timeline, then
    cut the gap regions out so the surviving segments stay phase-coherent across
    the gaps. Returns a TimeSeriesGappy of `nseg` segments."""
    gap_lengths = base_gap + np.random.uniform(0.0, base_gap, size=nseg - 1)  # seconds
    seg_samples = int(round(seg_length / tsamp))
    gap_samples = [int(round(gl / tsamp)) for gl in gap_lengths]

    # Sample offset of each segment's start within the full continuous series.
    seg_starts = [0]
    for i in range(1, nseg):
        seg_starts.append(seg_starts[-1] + seg_samples + gap_samples[i - 1])
    total_length = (seg_starts[-1] + seg_samples) * tsamp

    # Scale amplitude by the on-fraction so the combined S/N stays comparable to
    # a single dense segment (same convention as demo_gappy_timeseries.py).
    fraction_on = (nseg * seg_length) / total_length
    snr = 20.0 * np.sqrt((300.0 / 745.0) / fraction_on)

    full = TimeSeries.generate(
        length=total_length, tsamp=tsamp, period=TRUE_PERIOD,
        amplitude=snr, stdnoise=1.0,
    )

    base_mjd = 58000.0
    segments = []
    for start in seg_starts:
        data = full.data[start:start + seg_samples]
        mjd = base_mjd + start * tsamp / 86400.0
        segments.append(TimeSeries(data, tsamp, copy=True, metadata={"mjd": mjd}))
    return TimeSeriesGappy(segments)


def peak(pgram):
    """Return (period, snr) at the maximum of the max-over-widths S/N curve."""
    periods = np.asarray(pgram.periods)
    snr = np.asarray(pgram.snrs).max(axis=1)
    i = int(np.nanargmax(snr))
    return periods[i], snr[i]

def normalise_snr(snr):
    snr = np.asarray(snr)
    snr -= np.nanmedian(snr)
    snr /= (np.nanpercentile(snr, 75) - np.nanpercentile(snr, 25))
    return snr

def make_plot(g, pg_new, pg_old, fname=None):
    import matplotlib.pyplot as plt

    fig, (ax_ts, ax_pg) = plt.subplots(2, 1, figsize=(10, 8))

    # Top: the gappy time series segments on a common time axis.
    ref = g[0].metadata["mjd"]
    for i, ts in enumerate(g):
        t0 = (ts.metadata["mjd"] - ref) * 86400.0
        t = t0 + np.arange(ts.nsamp) * ts.tsamp
        ax_ts.plot(t, ts.data, lw=0.6, label=f"segment {i}")
    ax_ts.set_xlabel("Time since first epoch (s)")
    ax_ts.set_ylabel("Amplitude")
    gaps = g.gap_samples
    ax_ts.set_title(
        f"TimeSeriesGappy  ({len(g)} segments, "
        f"gaps = {min(gaps)}-{max(gaps)} samples)"
    )
    ax_ts.legend(loc="upper right", fontsize="small")

    # Bottom: the two gappy periodograms overlaid, with the residual on a twin
    # axis. old (black solid) and new (orange dashed) should nearly coincide.
    p_o = np.asarray(pg_old.periods)
    s_o = np.asarray(pg_old.snrs).max(axis=1)
    o_o = np.argsort(p_o)
    p_n = np.asarray(pg_new.periods)
    s_n = np.asarray(pg_new.snrs).max(axis=1)
    o_n = np.argsort(p_n)

    s_n = normalise_snr(s_n)
    s_o = normalise_snr(s_o)
   

    ax_pg.plot(p_o[o_o], s_o[o_o], lw=2.0, color="k", label="backend='cpp_old'")
    ax_pg.plot(p_n[o_n], s_n[o_n], lw=1.0, color="tab:orange", ls="--",
               label="backend='cpp' (segment-wise)")
    ax_pg.set_xlabel("Trial period (s)")
    ax_pg.set_ylabel("max S/N")
    ax_pg.set_title("ffa_search_gappy: old zero-padding vs new segment-wise C++ kernel")
    ax_pg.legend(loc="upper left")

    ax_res = ax_pg.twinx()
    ax_res.plot(p_o[o_o], np.abs(s_o[o_o] - s_n[o_o]), lw=0.8,
                color="tab:green", alpha=0.6)
    ax_res.set_ylabel("|old - new|", color="tab:green")
    ax_res.tick_params(axis="y", labelcolor="tab:green")

    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=120)
        print(f"\nSaved plot to {fname}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", metavar="FILE", default=None,
                        help="Save the comparison plot to FILE (PNG).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    g = make_gappy()
    print(g)
    print(f"  {len(g)} segments (sizes {[s.nsamp for s in g]}), "
          f"gaps = {g.gap_samples}")

    # The search resolution must satisfy period_min >= tsamp * bins_min.
    # With tsamp = 0.01 s and bins_min = 240 the lower bound is 2.4 s.
    kw = dict(
        period_min=TRUE_PERIOD - PERIOD_HALF_WINDOW,
        period_max=TRUE_PERIOD + PERIOD_HALF_WINDOW,
        bins_min=240, bins_max=260, deredden=False,
    )

    print("\nRunning gappy search with both C++ kernels...")
    _, pg_new = timed("ffa_search_gappy [cpp, segment-wise]",
                      ffa_search_gappy, g, backend="cpp", **kw)
    _, pg_old = timed("ffa_search_gappy [cpp_old, zero-padding]",
                      ffa_search_gappy, g, backend="cpp_old", **kw)

    per_n, snr_n = peak(pg_new)
    per_o, snr_o = peak(pg_old)
    s_n = np.asarray(pg_new.snrs).max(axis=1)
    s_o = np.asarray(pg_old.snrs).max(axis=1)

    s_n = normalise_snr(s_n)
    s_o = normalise_snr(s_o)
    print("\nResults (injected period = "
          f"{TRUE_PERIOD:.4f} s):")
    print(f"  cpp (new): peak period = {per_n:.6f} s   S/N = {np.max(s_n):.3f}")
    print(f"  cpp_old  : peak period = {per_o:.6f} s   S/N = {np.max(s_o):.3f}")
    print(f"  max |S/N_new - S/N_old| = {(np.abs(np.nanmax(s_n) - np.nanmax(s_o))):.3e}")
    same_peak = np.isclose(per_n, per_o, rtol=1e-4) and np.isclose(snr_n, snr_o, rtol=0.05)
    print("  -> kernels find the same peak" if same_peak
          else "  -> kernels find DIFFERENT peaks")

    make_plot(g, pg_new, pg_old, fname=args.plot)


if __name__ == "__main__":
    main()
