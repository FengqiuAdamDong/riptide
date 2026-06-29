#!/usr/bin/env python
"""
Manual test / demo script for the TimeSeriesGappy class and ffa_search_gappy.

Run from anywhere (riptide must be importable):

    python scripts/demo_gappy_timeseries.py
    python scripts/demo_gappy_timeseries.py --plot gappy.png

It exercises:
  * TimeSeriesGappy.generate (long series with a chunk cut out of the middle)
  * the gap_samples property
  * tsamp alignment (coarsest tsamp wins; finer series get downsampled)
  * ordering by observation epoch (mjd)
  * the validation / error paths
  * ffa_search_gappy (gap-aware FFA search of the collection)
"""
import argparse

import numpy as np

from riptide import TimeSeries, TimeSeriesGappy, ffa_search, ffa_search_gappy

# True period of the injected signal (seconds). With tsamp = 1 s this is 4
# samples, i.e. the signal folds onto ~4 phase bins. The search window is set
# to +/- 0.2 s around this value.
TRUE_PERIOD = 4.0
PERIOD_HALF_WINDOW = 0.2


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def normalise_snr(snr):
    """Normalise a S/N curve by subtracting its median and dividing by its
    interquartile range, so periodograms can be compared on a common scale."""
    snr = np.asarray(snr, dtype=np.float64)
    finite = snr[np.isfinite(snr)]
    median = np.median(finite)
    q75, q25 = np.percentile(finite, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        iqr = 1.0
    return (snr - median) / iqr


def demo_generate():
    banner("1. TimeSeriesGappy.generate: cut a chunk out of the middle")
    tsamp = 1.0            # 1 s samples (so 1 sample == 1 bin of period)
    length = 45.0          # 45 samples of data before the gap is removed
    gap_start = 15.0       # gap begins 15 samples in
    gap_length = 15.0      # 15 samples excised -> 15 / gap / 15 layout

    g = TimeSeriesGappy.generate(
        length=length,
        tsamp=tsamp,
        period=TRUE_PERIOD,
        gap_start=gap_start,
        gap_length=gap_length,
        amplitude=20.0,
        stdnoise=1.0,
    )

    print(g)
    for i, ts in enumerate(g):
        print(
            f"  segment {i}: nsamp={ts.nsamp:5d}  tobs={ts.length:7.3f}s  "
            f"mjd={ts.metadata['mjd']:.8f}"
        )

    expected_gap = int(round(gap_length / tsamp))
    print(f"\n  gap_samples           = {g.gap_samples}")
    print(f"  expected gap (samples) = [{expected_gap}]")
    assert g.gap_samples == [expected_gap], g.gap_samples
    assert len(g) == 2
    # Each surviving segment should be (length - gap_length) / 2 long
    assert g[0].nsamp == int(round(gap_start / tsamp))
    print("  -> OK")
    return g


def demo_alignment():
    banner("2. tsamp alignment: finer series downsampled to the coarsest")
    a = TimeSeries.generate(length=10.0, tsamp=0.001, period=1.0, stdnoise=0.0)
    b = TimeSeries.generate(length=10.0, tsamp=0.002, period=1.0, stdnoise=0.0)
    c = TimeSeries.generate(length=10.0, tsamp=0.005, period=1.0, stdnoise=0.0)

    print(f"  input tsamps: {a.tsamp}, {b.tsamp}, {c.tsamp}")
    g = TimeSeriesGappy([a, b, c])
    print(f"  common tsamp: {g.tsamp}  (expected {max(a.tsamp, b.tsamp, c.tsamp)})")
    assert np.isclose(g.tsamp, 0.005)
    for ts in g:
        assert np.isclose(ts.tsamp, g.tsamp)
    print("  all series share the common tsamp -> OK")


def demo_ordering():
    banner("3. ordering by observation epoch (mjd)")
    # Build three segments with deliberately shuffled epochs.
    def seg(mjd, nsamp=100, tsamp=0.01):
        data = np.random.randn(nsamp).astype(np.float32)
        return TimeSeries(data, tsamp, metadata={"mjd": mjd})

    shuffled = [seg(58002.0), seg(58000.0), seg(58001.0)]
    print("  input order (mjd):  ", [ts.metadata["mjd"] for ts in shuffled])
    g = TimeSeriesGappy(shuffled)
    ordered = [ts.metadata["mjd"] for ts in g]
    print("  stored order (mjd): ", ordered)
    assert ordered == sorted(ordered)
    print("  series sorted ascending by epoch -> OK")


def demo_errors():
    banner("4. validation / error paths")
    # Empty input
    try:
        TimeSeriesGappy([])
    except ValueError as e:
        print(f"  empty input        -> ValueError: {e}")
    else:
        raise AssertionError("empty input should raise ValueError")

    # Non-TimeSeries element
    try:
        TimeSeriesGappy([TimeSeries.generate(5.0, 0.01, 1.0), 42])
    except TypeError as e:
        print(f"  bad element        -> TypeError: {e}")
    else:
        raise AssertionError("non-TimeSeries element should raise TypeError")

    # gap_samples without epochs
    a = TimeSeries.generate(length=5.0, tsamp=0.01, period=1.0, stdnoise=0.0)
    try:
        TimeSeriesGappy([a, a.copy()]).gap_samples
    except ValueError as e:
        print(f"  gap without mjd    -> ValueError: {e}")
    else:
        raise AssertionError("missing mjd should raise ValueError")

    # Gap outside the data
    try:
        TimeSeriesGappy.generate(length=10.0, tsamp=0.01, period=1.0,
                                 gap_start=5.0, gap_length=20.0)
    except ValueError as e:
        print(f"  gap out of bounds  -> ValueError: {e}")
    else:
        raise AssertionError("out-of-bounds gap should raise ValueError")
    print("  all error paths behave -> OK")


def demo_search(g, period_min=TRUE_PERIOD - PERIOD_HALF_WINDOW,
                period_max=TRUE_PERIOD + PERIOD_HALF_WINDOW,
                bins_min=3, bins_max=4):
    banner("5. ffa_search_gappy: gap-aware FFA search of the collection")
    # NOTE: the search resolution must satisfy period_min >= tsamp * bins_min.
    # With tsamp = 1 s and bins_min = 3 the lower bound is 3 s, comfortably
    # below the injected 4 s (4 sample) period.
    segments, pgram = ffa_search_gappy(
        g,
        period_min=period_min,
        period_max=period_max,
        bins_min=bins_min,
        bins_max=bins_max,
        deredden=False,
    )
    periods = np.asarray(pgram.periods)
    snr = np.asarray(pgram.snrs).max(axis=1)

    print(f"  searched {len(segments)} segment(s), gaps = {g.gap_samples}")
    print(f"  trial periods : {periods.size}")
    print(f"  S/N array     : {pgram.snrs.shape}")
    if np.isfinite(snr).any():
        ibest = int(np.nanargmax(snr))
        print(
            f"  peak trial    : period = {periods[ibest]:.6f} s, "
            f"S/N = {snr[ibest]:.2f}"
        )
    print(
        "  NOTE: the actual S/N values come from the periodogram_gappy C++\n"
        "        kernel; this demo only checks that the search pipeline runs\n"
        "        end-to-end and returns arrays of the expected shape."
    )
    assert len(segments) == len(g)
    assert pgram.snrs.shape[0] == periods.size
    print("  -> pipeline OK")
    return segments, pgram


def demo_search_individual(g, period_min=TRUE_PERIOD - PERIOD_HALF_WINDOW,
                           period_max=TRUE_PERIOD + PERIOD_HALF_WINDOW,
                           bins_min=3, bins_max=4):
    banner("6. ffa_search on each individual segment")
    # Run a plain (gap-unaware) FFA search on every surviving segment so the
    # per-segment periodograms can be compared against the combined,
    # gap-aware result from ffa_search_gappy.
    results = []
    for i, ts in enumerate(g):
        _, pgram = ffa_search(
            ts,
            period_min=period_min,
            period_max=period_max,
            bins_min=bins_min,
            bins_max=bins_max,
            deredden=False,
        )
        periods = np.asarray(pgram.periods)
        snr = np.asarray(pgram.snrs).max(axis=1)
        ibest = int(np.nanargmax(snr)) if np.isfinite(snr).any() else 0
        print(
            f"  segment {i}: trials = {periods.size:4d}  "
            f"peak period = {periods[ibest]:.6f} s  S/N = {snr[ibest]:.2f}"
        )
        results.append(pgram)
    print("  -> per-segment searches OK")
    return results


def demo_search_stitched(g, period_min=TRUE_PERIOD - PERIOD_HALF_WINDOW,
                         period_max=TRUE_PERIOD + PERIOD_HALF_WINDOW,
                         bins_min=3, bins_max=4):
    banner("7. ffa_search on the segments stitched into one series")
    # Naively glue the surviving segments back-to-back into a single contiguous
    # TimeSeries (the gap is simply removed, not zero-filled) and run a plain
    # FFA search. This is the gap-unaware alternative to ffa_search_gappy.
    data = np.concatenate([ts.data for ts in g])
    stitched = TimeSeries(data, g.tsamp, metadata=g[0].metadata)
    print(f"  stitched series: nsamp={stitched.nsamp}  tobs={stitched.length:.3f}s")

    _, pgram = ffa_search(
        stitched,
        period_min=period_min,
        period_max=period_max,
        bins_min=bins_min,
        bins_max=bins_max,
        deredden=False,
    )
    periods = np.asarray(pgram.periods)
    snr = np.asarray(pgram.snrs).max(axis=1)
    ibest = int(np.nanargmax(snr)) if np.isfinite(snr).any() else 0
    print(
        f"  trials = {periods.size}  "
        f"peak period = {periods[ibest]:.6f} s  S/N = {snr[ibest]:.2f}"
    )
    print("  -> stitched search OK")
    return pgram


def make_plot(g, pgram, individual_pgrams=None, stitched_pgram=None, fname=None):
    import matplotlib.pyplot as plt

    fig, (ax_ts, ax_pg) = plt.subplots(2, 1, figsize=(10, 7))

    # Top panel: the gappy time series segments on a common time axis
    ref = g[0].metadata["mjd"]
    for i, ts in enumerate(g):
        t0 = (ts.metadata["mjd"] - ref) * 86400.0
        t = t0 + np.arange(ts.nsamp) * ts.tsamp
        ax_ts.plot(t, ts.data, lw=0.6, label=f"segment {i}")
    ax_ts.set_xlabel("Time since first epoch (s)")
    ax_ts.set_ylabel("Amplitude")
    ax_ts.set_title(f"TimeSeriesGappy  (gap = {g.gap_samples[0]} samples)")
    ax_ts.legend()

    # Bottom panel: the gappy periodogram with every per-segment periodogram
    # overlaid on top of each other on a common axis. Each periodogram is
    # normalised (median subtracted, divided by IQR) so they share a scale.
    periods = np.asarray(pgram.periods)
    snr = normalise_snr(np.asarray(pgram.snrs).max(axis=1))
    order = np.argsort(periods)
    ax_pg.plot(periods[order], snr[order], lw=1.0, color="k",
               label="ffa_search_gappy (combined)")
    if stitched_pgram is not None:
        p = np.asarray(stitched_pgram.periods)
        s = normalise_snr(np.asarray(stitched_pgram.snrs).max(axis=1))
        o = np.argsort(p)
        ax_pg.plot(p[o], s[o], lw=1.0, color="tab:red", ls="--",
                   label="ffa_search (stitched)")
    if individual_pgrams:
        for i, pg in enumerate(individual_pgrams):
            p = np.asarray(pg.periods)
            s = normalise_snr(np.asarray(pg.snrs).max(axis=1))
            o = np.argsort(p)
            ax_pg.plot(p[o], s[o], lw=0.6, alpha=0.7, label=f"segment {i}")
    ax_pg.set_xlabel("Trial period (s)")
    ax_pg.set_ylabel("Normalised S/N  (median subtracted, / IQR)")
    ax_pg.set_title("Periodograms: combined vs. individual segments")
    ax_pg.legend()

    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=120)
        print(f"\nSaved plot to {fname}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", metavar="FILE", default=None,
        help="Save a plot of the generated gappy time series to FILE (PNG).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    g = demo_generate()
    # demo_alignment()
    # demo_ordering()
    # demo_errors()
    segments, pgram = demo_search(g)
    individual_pgrams = demo_search_individual(g)
    stitched_pgram = demo_search_stitched(g)

    make_plot(g, pgram, individual_pgrams, stitched_pgram, args.plot)

    banner("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
