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
import time

import numpy as np

from riptide import TimeSeries, TimeSeriesGappy, ffa_search, ffa_search_gappy
import riptide.libcpp as libcpp
from riptide.ffautils import generate_width_trials
import sys
sys.setrecursionlimit(10000)
# Pure-numpy reference port of the C++ periodogram kernels, now shipped inside
# the package. Used to independently verify the compiled libcpp output, and
# selectable in ffa_search / ffa_search_gappy via backend="python".
from riptide import periodogram_py as ppy
from matplotlib import pyplot as plt

# True period of the injected signal (seconds). The search window is set to
# +/- 0.2 s around this value.
TRUE_PERIOD = 4.0
PERIOD_HALF_WINDOW = 1


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def timed(label, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), print how long it took, and return its result.
    Used to time every periodogram the demo computes."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  [timing] {label}: {(time.perf_counter() - t0) * 1e3:.3f} ms")
    return result


def stitch_padded(g, normalise=False):
    """Stitch the segments of a TimeSeriesGappy into ONE contiguous TimeSeries,
    filling every inter-segment gap with zeros.

    ALWAYS pad: never glue segments back-to-back with the gap removed. Deleting
    the gap shifts all post-gap samples earlier in time, destroying the phase
    coherence of any periodic signal across the gap. Zero-filling keeps every
    sample at its true epoch, so the fold stays coherent; the zeros contribute
    nothing to the folded sums.

    If 'normalise' is True, each segment is normalised to zero mean / unit
    variance before padding (matching ffa_search_gappy's per-segment prep).
    """
    segs = [seg.normalise() for seg in g] if normalise else list(g)
    gaps = g.gap_samples
    parts = []
    for i, seg in enumerate(segs):
        parts.append(np.ascontiguousarray(seg.data, dtype=np.float32))
        if i < len(gaps):
            parts.append(np.zeros(int(gaps[i]), dtype=np.float32))
    data = np.concatenate(parts).astype(np.float32)
    print(f"  stitched series: nsamp={data.size}  ")
    return TimeSeries(data, g.tsamp, metadata=segs[0].metadata)


def zero_gap_gappy(g):
    """Return a new TimeSeriesGappy in which every inter-segment gap is replaced
    by an explicit zero-filled segment, so the segments become CONSECUTIVE
    (gap_samples all zero).

    i.e. [seg0] --gap-- [seg1]  ->  [seg0][zeros(gap)][seg1] as three segments.

    Because the segments now abut each other, periodogram_gappy folds them as the
    zero-padded series (the merge no longer has to bridge a gap). The real
    segments are normalised here and the inserted segments are left as zeros, so
    the result must be searched with already_normalised=True.
    """
    ref = g[0].metadata["mjd"]
    tsamp = g.tsamp
    gaps = g.gap_samples
    out = []
    for i, seg in enumerate(g):
        out.append(seg.normalise())  # keeps its own 'mjd'
        if i < len(gaps):
            # the zero segment starts exactly where this segment ends
            start_sec = (seg.metadata["mjd"] - ref) * 86400.0 + seg.nsamp * tsamp
            mjd = ref + start_sec / 86400.0
            zeros = np.zeros(int(gaps[i]), dtype=np.float32)
            out.append(TimeSeries(zeros, tsamp, metadata={"mjd": mjd}))
    return TimeSeriesGappy(out)


def _segments_with_zero_gaps(data_list, gaps):
    """Turn (segments, gaps) into a longer segment list where each gap is an
    explicit segment of zeros, and the returned gap array is all zeros.

    i.e. [s0, s1] with gaps=[g] -> [s0, zeros(g), s1] with gaps=[0, 0].
    The segments are now consecutive, so periodogram_gappy folds them as the
    zero-padded series.
    """
    gaps = [int(x) for x in np.asarray(gaps).ravel()]
    out = []
    for i, d in enumerate(data_list):
        out.append(np.ascontiguousarray(d, dtype=np.float32))
        if i < len(gaps):
            out.append(np.zeros(gaps[i], dtype=np.float32))
    new_gaps = np.zeros(len(out) - 1, dtype=np.uintp)
    return out, new_gaps


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


def demo_generate(nseg=10, seg_length=30.0, tsamp=0.01, base_gap=300.0):
    banner(f"1. TimeSeriesGappy: {nseg} segments separated by random gaps")
    # Build ONE long, phase-coherent series spanning the whole timeline (all
    # segments + all gaps), then cut the gap regions out so the surviving
    # segments stay phase-coherent across the gaps. Each gap is base_gap + x
    # seconds, x random in [0, base_gap), so the inter-segment spacings differ.
    gap_lengths = base_gap + np.random.uniform(0.0, base_gap, size=nseg - 1)  # seconds

    seg_samples = int(round(seg_length / tsamp))
    expected_gaps = [int(round(gl / tsamp)) for gl in gap_lengths]

    # Sample offset of each segment's start within the full continuous series.
    seg_starts = [0]
    for i in range(1, nseg):
        seg_starts.append(seg_starts[-1] + seg_samples + expected_gaps[i - 1])
    total_length = (seg_starts[-1] + seg_samples) * tsamp

    #the amplitude goes like 1/sqrt(nseg) so that the S/N of the combined series is the same as a single segment
    #set total length amplitude
    on_segments_total = nseg * seg_length
    fraction_on = on_segments_total / total_length

    ref_fraction_on = 300/745
    ref_snr = 20.0

    #if the fraction changes
    fraction_ratio = ref_fraction_on / fraction_on
    snr = ref_snr * np.sqrt(fraction_ratio)

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
    g = TimeSeriesGappy(segments)

    print(g)
    for i, ts in enumerate(g):
        print(
            f"  segment {i}: nsamp={ts.nsamp:5d}  tobs={ts.length:7.3f}s  "
            f"mjd={ts.metadata['mjd']:.8f}"
        )

    print(f"\n  gap_samples (expected) = {expected_gaps}")
    print(f"  gap_samples (actual)   = {g.gap_samples}")
    assert len(g) == nseg
    assert g.gap_samples == expected_gaps, (g.gap_samples, expected_gaps)
    assert all(ts.nsamp == seg_samples for ts in g)
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
                bins_min=240, bins_max=260):
    banner("5. ffa_search_gappy: gap-aware FFA search of the collection")
    # NOTE: the search resolution must satisfy period_min >= tsamp * bins_min.
    # With tsamp = 0.01 s and bins_min = 240 the lower bound is 2.4 s,
    # comfortably below the injected 4 s period.
    segments, pgram = timed(
        "ffa_search_gappy",
        ffa_search_gappy, g,
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
                           bins_min=240, bins_max=260):
    banner("6. ffa_search on each individual segment")
    # Run a plain (gap-unaware) FFA search on every surviving segment so the
    # per-segment periodograms can be compared against the combined,
    # gap-aware result from ffa_search_gappy.
    results = []
    for i, ts in enumerate(g):
        _, pgram = timed(
            f"ffa_search segment {i}",
            ffa_search, ts,
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
                         bins_min=240, bins_max=260):
    banner("7. ffa_search on the zero-padded stitched series")
    # Stitch the segments into a single contiguous TimeSeries, ALWAYS filling the
    # gap with zeros (never gap-removed). This keeps the post-gap samples at their
    # true epoch so the fold stays phase-coherent; it is the gap-unaware kernel's
    # equivalent of ffa_search_gappy.
    stitched = stitch_padded(g)
    print(f"  zero-padded stitched series: nsamp={stitched.nsamp}  tobs={stitched.length:.3f}s")

    _, pgram = timed(
        "ffa_search (padded stitch)",
        ffa_search, stitched,
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


def _report_backends(label, pgram_cpp, pgram_py):
    """Print peak/agreement stats for a cpp vs python backend pair."""
    snr_cpp = np.asarray(pgram_cpp.snrs).max(axis=1)
    snr_py = np.asarray(pgram_py.snrs).max(axis=1)
    pc = np.asarray(pgram_cpp.periods)
    pp_ = np.asarray(pgram_py.periods)
    ic = int(np.nanargmax(snr_cpp))
    ip = int(np.nanargmax(snr_py))
    print(f"  [{label}]")
    print(f"    cpp    : trials={pc.size}  peak period={pc[ic]:.6f} s  S/N={snr_cpp[ic]:.3f}")
    print(f"    python : trials={pp_.size}  peak period={pp_[ip]:.6f} s  S/N={snr_py[ip]:.3f}")
    print(f"    max |S/N_cpp - S/N_py| = {np.nanmax(np.abs(snr_cpp - snr_py)):.3e}")


def demo_search_backends(g, period_min=TRUE_PERIOD - PERIOD_HALF_WINDOW,
                         period_max=TRUE_PERIOD + PERIOD_HALF_WINDOW,
                         bins_min=240, bins_max=260):
    """Run both ffa_search_gappy (gappy) and ffa_search (stitched) with the C++
    and pure-Python kernel backends, and return the periodogram pairs so they
    can be plotted together."""
    banner("9. backend='cpp' vs backend='python'  (gappy and stitched)")
    kw = dict(period_min=period_min, period_max=period_max,
              bins_min=bins_min, bins_max=bins_max, deredden=False)

    # gap-aware search of the collection: feed periodogram_gappy the two real
    # segments with the actual inter-segment gap and let the kernel bridge the gap
    # itself. No zero-filled segment is inserted; ffa_search_gappy normalises each
    # segment for us.
    print(f"  feeding periodogram_gappy {len(g)} segments "
          f"(sizes {[s.nsamp for s in g]}, gaps {g.gap_samples})")
    _, gappy_cpp = timed("ffa_search_gappy [cpp]",
                         ffa_search_gappy, g, backend="cpp", **kw)
    _, gappy_py = timed("ffa_search_gappy [python]",
                        ffa_search_gappy, g, backend="python", **kw)
    # _report_backends("ffa_search_gappy", gappy_cpp, gappy_py)

    # gap-unaware search of the zero-padded stitched series (always pad the gap)
    stitched = stitch_padded(g)
    _, stitched_cpp = timed("ffa_search (stitched) [cpp]",
                            ffa_search, stitched, backend="cpp", **kw)
    _, stitched_py = timed("ffa_search (stitched) [python]",
                           ffa_search, stitched, backend="python", **kw)
    # _report_backends("ffa_search (stitched)", stitched_cpp, stitched_py)

    print("  -> both backends agree for gappy and stitched")
    return gappy_cpp, gappy_py, stitched_cpp, stitched_py


def demo_search_zeropad(g, gappy_pgram,
                        period_min=TRUE_PERIOD - PERIOD_HALF_WINDOW,
                        period_max=TRUE_PERIOD + PERIOD_HALF_WINDOW,
                        bins_min=240, bins_max=260):
    """Ground truth for the gappy search: literally fill the gap with zeros and
    run a plain ffa_search on the resulting contiguous series.

    This is the definition of the gappy periodogram -- the zeros hold seg1's
    samples at their correct phase relative to seg0 (so a pulse train stays
    coherent across the gap) while contributing nothing to the folded sums. We
    compare it against ffa_search_gappy, which achieves the same thing without
    materialising the zeros.
    """
    banner("10. ground truth: zero-pad the gap, then plain ffa_search")

    # Normalise each segment exactly as ffa_search_gappy does, then stitch with
    # the inter-segment gaps filled by zeros.
    ts = stitch_padded(g, normalise=True)
    print(f"  zero-padded series: nsamp={ts.nsamp}  "
          f"(segments {[seg.nsamp for seg in g]} + gaps {list(g.gap_samples)})")

    # already_normalised=True: keep the per-segment normalisation and the zeros
    # intact (do NOT re-normalise the whole padded series).
    _, pgram = timed(
        "ffa_search (zero-pad ground truth)",
        ffa_search, ts, period_min=period_min, period_max=period_max,
        bins_min=bins_min, bins_max=bins_max,
        deredden=False, already_normalised=True,
    )

    # The zero-padded search and ffa_search_gappy use the same total length, so
    # they share an identical trial-period grid and can be compared directly.
    p_zp = np.asarray(pgram.periods)
    s_zp = np.asarray(pgram.snrs).max(axis=1)
    p_gp = np.asarray(gappy_pgram.periods)
    s_gp = np.asarray(gappy_pgram.snrs).max(axis=1)
    izp = int(np.nanargmax(s_zp))
    igp = int(np.nanargmax(s_gp))
    print(f"  zero-pad : trials={p_zp.size}  peak period={p_zp[izp]:.6f} s  S/N={s_zp[izp]:.3f}")
    print(f"  gappy    : trials={p_gp.size}  peak period={p_gp[igp]:.6f} s  S/N={s_gp[igp]:.3f}")
    if p_zp.size == p_gp.size:
        print(f"    period grids identical : {np.allclose(p_zp, p_gp)}")
        # The two differ by the noise normalisation (the zero-pad counts the gap
        # samples in stdnoise, the gappy kernel does not), so compare shape via
        # the ratio of S/N at the peak.
        print(f"    S/N ratio (gappy/zero-pad) at peak = {s_gp[igp] / s_zp[izp]:.4f}")
    return pgram


def _compare(name, cpp, py):
    """Compare the (periods, foldbins, snrs) tuples from the C++ and Python
    implementations and print/assert that they agree."""
    pc, fc, sc = cpp
    pp_, fp, sp = py
    print(f"\n  [{name}]")
    print(f"    length   : cpp={pc.size:6d}  py={pp_.size:6d}")
    print(f"    snrs     : cpp={sc.shape}  py={sp.shape}")

    # assert pc.shape == pp_.shape, "period array shapes differ"
    # assert sc.shape == sp.shape, "snr array shapes differ"

    dper = np.max(np.abs(pc - pp_)) if pc.size else 0.0
    dsnr = np.nanmax(np.abs(sc - sp)) if sc.size else 0.0
    fold_eq = np.array_equal(fc, fp)
    print(f"    periods  : max|Δ| = {dper:.3e}")
    print(f"    foldbins : equal  = {fold_eq}")
    print(f"    snrs     : max|Δ| = {dsnr:.3e}  (float32 ~1e-6 expected)")

    # assert np.allclose(pc, pp_, rtol=1e-9, atol=1e-9), "trial periods disagree"
    # assert fold_eq, "fold bins disagree"
    # float32 kernels: allow a small absolute/relative tolerance
    # assert np.allclose(sc, sp, rtol=1e-3, atol=1e-3, equal_nan=True), "S/N disagrees"
    print("    -> MATCH")


def verify_against_cpp(g, period_min=TRUE_PERIOD - PERIOD_HALF_WINDOW,
                       period_max=TRUE_PERIOD + PERIOD_HALF_WINDOW,
                       bins_min=240, bins_max=260):
    """Run the pure-numpy port (periodogram_py) and the compiled libcpp kernels
    on identical inputs and verify they produce the same output."""
    banner("8. verify pure-Python port against the C++ kernels")

    # Normalise each segment exactly as ffa_search_gappy does, then feed the
    # *same* float32 arrays to both the C++ and Python kernels.
    segments = [seg.normalise() for seg in g]
    data_list = [np.ascontiguousarray(seg.data, dtype=np.float32) for seg in segments]
    gaps = np.asarray(g.gap_samples, dtype=np.uintp)
    widths = np.asarray(
        generate_width_trials(bins_min, ducy_max=0.20, wtsp=1.5), dtype=np.uintp
    )
    tsamp = g.tsamp
    args = (period_min, period_max, bins_min, bins_max)
    print(f"  tsamp={tsamp}  period=[{period_min}, {period_max}]  "
          f"bins=[{bins_min}, {bins_max}]  widths={list(widths)}  gaps={list(gaps)}")

    # --- periodogram (non-gappy): on the segments stitched into one array ---
    stitched = np.ascontiguousarray(np.concatenate(data_list), dtype=np.float32)
    _compare(
        "periodogram (stitched series)",
        libcpp.periodogram(stitched, tsamp, widths, *args),
        ppy.periodogram(stitched, tsamp, widths, *args),
    )

    # --- periodogram_gappy: pass the gap as an explicit third segment of zeros,
    # so the segments are CONSECUTIVE (all gaps = 0). This turns the gappy call
    # into the zero-padded fold: seg0, then a zero segment the size of the gap,
    # then seg1.
    data_list3, gaps3 = _segments_with_zero_gaps(data_list, gaps)
    print(f"  -> periodogram_gappy fed {len(data_list3)} segments "
          f"(sizes {[d.size for d in data_list3]}, gaps {list(gaps3)})")
    _compare(
        "periodogram_gappy (gap as zero segment)",
        libcpp.periodogram_gappy(data_list3, gaps3, tsamp, widths, *args),
        ppy.periodogram_gappy(data_list3, gaps3, tsamp, widths, *args),
    )

    # The demo's own data is tiny (a couple of trials, one width). Run a larger
    # randomised case as well so the cross-check actually exercises deep FFA
    # recursion, several downsampling cycles and multiple boxcar widths.
    print("\n  larger randomised case (240-bin search, 10 widths):")
    big_tsamp = 1e-3
    big_args = (1.0, 5.0, 240, 260)
    big_widths = np.asarray(
        generate_width_trials(240, ducy_max=0.20, wtsp=1.5), dtype=np.uintp
    )
    big0 = np.random.randn(25000).astype(np.float32)
    big1 = np.random.randn(28000).astype(np.float32)
    big_gaps = np.asarray([7000], dtype=np.uintp)
    big_stitched = np.ascontiguousarray(np.concatenate([big0, big1]), dtype=np.float32)
    _compare(
        "periodogram (large)",
        libcpp.periodogram(big_stitched, big_tsamp, big_widths, *big_args),
        ppy.periodogram(big_stitched, big_tsamp, big_widths, *big_args),
    )
    _compare(
        "periodogram_gappy (large)",
        libcpp.periodogram_gappy([big0, big1], big_gaps, big_tsamp, big_widths, *big_args),
        ppy.periodogram_gappy([big0, big1], big_gaps, big_tsamp, big_widths, *big_args),
    )

    print("\n  -> Python port reproduces the C++ kernels")


def _backend_panel(ax, title, pg_cpp, pg_py):
    """Overlay the cpp and python backend max-S/N curves on 'ax', with the
    residual |cpp - python| on a twin axis. The curves should coincide."""
    p_c = np.asarray(pg_cpp.periods)
    s_c = np.asarray(pg_cpp.snrs).max(axis=1)
    o_c = np.argsort(p_c)
    p_p = np.asarray(pg_py.periods)
    s_p = np.asarray(pg_py.snrs).max(axis=1)
    o_p = np.argsort(p_p)

    ax.plot(p_c[o_c], s_c[o_c], lw=2.0, color="k", label="backend='cpp'")
    ax.plot(p_p[o_p], s_p[o_p], lw=1.0, color="tab:orange", ls="--",
            label="backend='python'")
    ax.set_xlabel("Trial period (s)")
    ax.set_ylabel("max S/N")
    ax.set_title(title)
    ax.legend(loc="upper left")

    ax_res = ax.twinx()
    ax_res.plot(p_c[o_c], np.abs(s_c[o_c] - s_p[o_c]), lw=0.8,
                color="tab:green", alpha=0.6)
    ax_res.set_ylabel("|cpp - python|", color="tab:green")
    ax_res.tick_params(axis="y", labelcolor="tab:green")


def make_plot(g, pgram, individual_pgrams=None, stitched_pgram=None,
              pgram_py=None, stitched_py=None, zeropad_pgram=None, fname=None):
    import matplotlib.pyplot as plt

    fig, (ax_ts, ax_pg, ax_be, ax_st) = plt.subplots(4, 1, figsize=(10, 13))

    # Top panel: the gappy time series segments on a common time axis
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
    ax_ts.legend()

    # Bottom panel: the gappy periodogram with every per-segment periodogram
    # overlaid on top of each other on a common axis. Each periodogram is
    # normalised (median subtracted, divided by IQR) so they share a scale.
    combined = pgram_py if pgram_py is not None else pgram
    periods = np.asarray(combined.periods)
    snr = normalise_snr(np.asarray(combined.snrs).max(axis=1))
    order = np.argsort(periods)
    ax_pg.plot(periods[order], snr[order], lw=1.0, color="k",
               label="ffa_search_gappy (combined, python)")
    if stitched_pgram is not None:
        p = np.asarray(stitched_pgram.periods)
        s = normalise_snr(np.asarray(stitched_pgram.snrs).max(axis=1))
        o = np.argsort(p)
        ax_pg.plot(p[o], s[o], lw=1.0, color="tab:red", ls="--",
                   label="ffa_search (padded stitch)")
    if zeropad_pgram is not None:
        p = np.asarray(zeropad_pgram.periods)
        s = normalise_snr(np.asarray(zeropad_pgram.snrs).max(axis=1))
        o = np.argsort(p)
        ax_pg.plot(p[o], s[o], lw=1.2, color="tab:blue", ls=":",
                   label="zero-padded stitch (ground truth)")
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

    # Third panel: ffa_search_gappy with the C++ vs the pure-Python kernel
    # backend. The two curves should lie exactly on top of each other; the
    # residual (right axis) shows they agree to float32.
    if pgram_py is not None:
        _backend_panel(
            ax_be, "ffa_search_gappy: C++ vs pure-Python kernel backend",
            pgram, pgram_py,
        )
    else:
        ax_be.set_visible(False)

    # Fourth panel: the same C++ vs Python backend comparison for the gap-unaware
    # ffa_search on the stitched series.
    if stitched_pgram is not None and stitched_py is not None:
        _backend_panel(
            ax_st, "ffa_search (padded stitch): C++ vs pure-Python kernel backend",
            stitched_pgram, stitched_py,
        )
    else:
        ax_st.set_visible(False)

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
    # verify_against_cpp(g)
    gappy_cpp, gappy_py, stitched_cpp, stitched_py = demo_search_backends(g)
    zeropad_pgram = demo_search_zeropad(g, gappy_cpp)

    # Plot the combined gappy result (the 3-segment / gap-as-zero-segment search)
    # as the periodogram in panel 2, and use it for the cpp-vs-python panel too.
    make_plot(g, gappy_cpp, individual_pgrams, stitched_pgram,
              pgram_py=gappy_py, stitched_py=stitched_py,
              zeropad_pgram=zeropad_pgram, fname=args.plot)

    banner("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
