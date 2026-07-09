#!/usr/bin/env python
"""
Minimal comparison: gappy FFA search with the C++ kernel vs the pure-Python
kernel.

Both backends run ffa_search_gappy on the SAME TimeSeriesGappy, so the only
thing that differs is the periodogram kernel (libcpp.periodogram_gappy vs
periodogram_py.periodogram_gappy). The two S/N curves should lie exactly on
top of each other (agreement to float32 precision).

Run from anywhere (riptide must be importable):

    python scripts/demo_gappy_cpp_vs_python.py
    python scripts/demo_gappy_cpp_vs_python.py --plot gappy_cpp_vs_python.png
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


def make_gappy(nseg=2, seg_length=30.0, tsamp=0.01, base_gap=30.0):
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


def make_plot(g, pg_cpp, pg_py, fname=None):
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
    # axis. cpp (black solid) and python (orange dashed) should coincide.
    p_c = np.asarray(pg_cpp.periods)
    s_c = np.asarray(pg_cpp.snrs).max(axis=1)
    o_c = np.argsort(p_c)
    p_p = np.asarray(pg_py.periods)
    s_p = np.asarray(pg_py.snrs).max(axis=1)
    o_p = np.argsort(p_p)

    ax_pg.plot(p_c[o_c], s_c[o_c], lw=2.0, color="k", label="backend='cpp'")
    ax_pg.plot(p_p[o_p], s_p[o_p], lw=1.0, color="tab:orange", ls="--",
               label="backend='python'")
    ax_pg.set_xlabel("Trial period (s)")
    ax_pg.set_ylabel("max S/N")
    ax_pg.set_title("ffa_search_gappy: C++ vs pure-Python kernel backend")
    ax_pg.legend(loc="upper left")

    ax_res = ax_pg.twinx()
    ax_res.plot(p_c[o_c], np.abs(s_c[o_c] - s_p[o_c]), lw=0.8,
                color="tab:green", alpha=0.6)
    ax_res.set_ylabel("|cpp - python|", color="tab:green")
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

    print("\nRunning gappy search with both kernel backends...")
    _, pg_cpp = timed("ffa_search_gappy [cpp]",
                      ffa_search_gappy, g, backend="cpp", **kw)
    _, pg_py = timed("ffa_search_gappy [python]",
                     ffa_search_gappy, g, backend="python", **kw)

    per_c, snr_c = peak(pg_cpp)
    per_p, snr_p = peak(pg_py)
    s_c = np.asarray(pg_cpp.snrs).max(axis=1)
    s_p = np.asarray(pg_py.snrs).max(axis=1)

    print("\nResults (injected period = "
          f"{TRUE_PERIOD:.4f} s):")
    print(f"  cpp    : peak period = {per_c:.6f} s   S/N = {snr_c:.3f}")
    print(f"  python : peak period = {per_p:.6f} s   S/N = {snr_p:.3f}")
    print(f"  max |S/N_cpp - S/N_py| = {np.nanmax(np.abs(s_c - s_p)):.3e}")
    print("  -> backends agree" if np.allclose(s_c, s_p, atol=1e-3, equal_nan=True)
          else "  -> backends DIFFER")

    make_plot(g, pg_cpp, pg_py, fname=args.plot)


if __name__ == "__main__":
    main()
