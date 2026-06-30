import numpy as np

### Local module imports
import riptide.libcpp as libcpp
from . import periodogram_py
from .ffautils import generate_width_trials
from .periodogram import Periodogram
from .timing import timing


# Available periodogram kernel backends. "cpp" uses the compiled libcpp
# extension; "python" uses the pure-numpy reference port in periodogram_py
# (slower, but useful for testing/verification and for environments without the
# compiled extension).
_PERIODOGRAM_BACKENDS = {
    "cpp": libcpp.periodogram,
    "python": periodogram_py.periodogram,
}
_PERIODOGRAM_GAPPY_BACKENDS = {
    "cpp": libcpp.periodogram_gappy,
    "python": periodogram_py.periodogram_gappy,
}


def _resolve_backend(backends, backend):
    try:
        return backends[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; choose one of {sorted(backends)}"
        )


@timing
def ffa_search(
    tseries,
    period_min=1.0,
    period_max=30.0,
    fpmin=8,
    bins_min=240,
    bins_max=260,
    ducy_max=0.20,
    wtsp=1.5,
    deredden=True,
    rmed_width=4.0,
    rmed_minpts=101,
    already_normalised=False,
    backend="cpp",
):
    """
    Run a FFA search of a single TimeSeries object, producing its periodogram.

    Parameters
    ----------
    tseries : TimeSeries
        The time series object to search
    period_min : float
        Minimum period to search in seconds
    period_max : float
        Maximum period to search in seconds
    fpmin : int
        Minimum number of signal periods that must fit in the data. In other
        words, place a cap on period_max equal to DATA_LENGTH / fpmin
    bins_min : int
        Minimum number of phase bins in the folded data. Higher values
        provide better duty cycle resolution. As the code searches longer trial
        periods, the data are iteratively downsampled so that the number of
        phase bins remains between bins_min and bins_max
    bins_max : int
        Maximum number of phase bins in the folded data. Must be strictly
        larger than bins_min, approx. 10% larger is a good choice
    wtsp : float
        Multiplicative factor between consecutive boxcar width trials. The
        smallest width is always 1 phase bin, and the sequence of width
        trials is generated with the formula:
        W(n+1) = max( floor(wtsp x W(n)), W(n) + 1 )
        wtsp = 1.5 gives the following sequence of width trials (in number of
        phase bins): 1, 2, 3, 4, 6, 9, 13, 19 ...
    ducy_max : float
        Maximum duty cycle to optimally search. Limits the maximum width of the
        boxcar matched filters applied to any given profile.
        Example: on a 300 phase bin profile, ducy_max = 0.2 means that no
        boxcar filter of width > 60 bins will be applied
    deredden : bool
        Subtract red noise from the time series before searching
    rmed_width : float
        The width of the running median filter to subtract from the input data
        before processing, in seconds
    rmed_minpts : int
        The running median is calculated of a time scrunched version of the
        input data to save time: rmed_minpts is the minimum number of
        scrunched samples that must fit in the running median window
        Lower values make the running median calculation less accurate but
        faster, due to allowing a higher scrunching factor
    already_normalised : bool
        Assume that the data are already normalised to zero mean and unit
        standard deviation
    backend : str
        Which periodogram kernel implementation to use: "cpp" (default, the
        compiled libcpp extension) or "python" (the pure-numpy reference port in
        periodogram_py). Both produce the same result to float32 precision.

    Returns
    -------
    ts : TimeSeries
        The de-reddened and normalised time series that was actually searched
    pgram : Periodogram
        The output of the search, which contains among other things a 2D array
        representing S/N as a function of trial period and trial width.
    """
    periodogram = _resolve_backend(_PERIODOGRAM_BACKENDS, backend)

    ### Prepare data: deredden then normalise IN THAT ORDER
    if deredden:
        tseries = tseries.deredden(rmed_width, minpts=rmed_minpts)
    if not already_normalised:
        tseries = tseries.normalise()

    widths = generate_width_trials(bins_min, ducy_max=ducy_max, wtsp=wtsp)
    periods, foldbins, snrs = periodogram(
        tseries.data, tseries.tsamp, widths, period_min, period_max, bins_min, bins_max
    )
    pgram = Periodogram(widths, periods, foldbins, snrs, metadata=tseries.metadata)
    return tseries, pgram


@timing
def ffa_search_gappy(
    tsgappy,
    period_min=1.0,
    period_max=30.0,
    fpmin=8,
    bins_min=240,
    bins_max=260,
    ducy_max=0.20,
    wtsp=1.5,
    deredden=True,
    rmed_width=4.0,
    rmed_minpts=101,
    already_normalised=False,
    backend="cpp",
):
    """
    Run a FFA search of a TimeSeriesGappy object, producing its periodogram.

    This is the equivalent of :func:`ffa_search` for gappy data. Each
    individual TimeSeries that makes up the collection is de-reddened and
    normalised independently (in that order), so that segments coming from
    different observations are treated on their own merits. The prepared
    segments are then passed, *without* being stitched together, to
    ``libcpp.periodogram_gappy``, which is gap-aware: it FFA-transforms each
    segment separately and combines the transforms across the gaps.

    Parameters
    ----------
    tsgappy : TimeSeriesGappy
        The gappy time series to search. All its segments share a common
        ``tsamp`` and must carry an 'mjd' epoch so that the gap sizes can be
        computed.
    period_min : float
        Minimum period to search in seconds
    period_max : float
        Maximum period to search in seconds
    fpmin : int
        Minimum number of signal periods that must fit in the data. In other
        words, place a cap on period_max equal to DATA_LENGTH / fpmin
    bins_min : int
        Minimum number of phase bins in the folded data. Higher values
        provide better duty cycle resolution. As the code searches longer trial
        periods, the data are iteratively downsampled so that the number of
        phase bins remains between bins_min and bins_max
    bins_max : int
        Maximum number of phase bins in the folded data. Must be strictly
        larger than bins_min, approx. 10% larger is a good choice
    wtsp : float
        Multiplicative factor between consecutive boxcar width trials. The
        smallest width is always 1 phase bin, and the sequence of width
        trials is generated with the formula:
        W(n+1) = max( floor(wtsp x W(n)), W(n) + 1 )
        wtsp = 1.5 gives the following sequence of width trials (in number of
        phase bins): 1, 2, 3, 4, 6, 9, 13, 19 ...
    ducy_max : float
        Maximum duty cycle to optimally search. Limits the maximum width of the
        boxcar matched filters applied to any given profile.
        Example: on a 300 phase bin profile, ducy_max = 0.2 means that no
        boxcar filter of width > 60 bins will be applied
    deredden : bool
        Subtract red noise from each segment before searching
    rmed_width : float
        The width of the running median filter to subtract from the input data
        before processing, in seconds
    rmed_minpts : int
        The running median is calculated of a time scrunched version of the
        input data to save time: rmed_minpts is the minimum number of
        scrunched samples that must fit in the running median window
        Lower values make the running median calculation less accurate but
        faster, due to allowing a higher scrunching factor
    already_normalised : bool
        Assume that each segment is already normalised to zero mean and unit
        standard deviation
    backend : str
        Which periodogram kernel implementation to use: "cpp" (default, the
        compiled libcpp extension) or "python" (the pure-numpy reference port in
        periodogram_py). Both produce the same result to float32 precision.

    Returns
    -------
    segments : list of TimeSeries
        The de-reddened and normalised segments that were actually searched, in
        epoch order
    pgram : Periodogram
        The output of the search, which contains among other things a 2D array
        representing S/N as a function of trial period and trial width.
    """
    periodogram_gappy = _resolve_backend(_PERIODOGRAM_GAPPY_BACKENDS, backend)

    # Number of samples missing between each pair of consecutive segments.
    # Accessing this also enforces that every segment carries an 'mjd' epoch.
    gaps = np.asarray(tsgappy.gap_samples, dtype=np.uintp)
    if np.any(gaps < 0):
        raise ValueError(
            "overlapping segments detected (negative inter-series gap); "
            "ffa_search_gappy requires non-overlapping time series"
        )

    ### Prepare each segment: deredden then normalise IN THAT ORDER
    segments = []
    for segment in tsgappy:
        if deredden:
            segment = segment.deredden(rmed_width, minpts=rmed_minpts)
        if not already_normalised:
            segment = segment.normalise()
        segments.append(segment)

    # Pass the segments to the gap-aware C++ kernel without stitching them into
    # a single contiguous array. It FFA-transforms each segment on its own and
    # combines the per-segment transforms across the gaps.
    data_list = [np.ascontiguousarray(seg.data, dtype=np.float32) for seg in segments]

    widths = generate_width_trials(bins_min, ducy_max=ducy_max, wtsp=wtsp)
    periods, foldbins, snrs = periodogram_gappy(
        data_list, gaps, tsgappy.tsamp, widths, period_min, period_max, bins_min, bins_max
    )
    pgram = Periodogram(
        widths, periods, foldbins, snrs, metadata=segments[0].metadata
    )
    return segments, pgram
