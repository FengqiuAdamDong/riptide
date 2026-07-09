import numpy as np

from riptide.libffa import downsample


def downsample_vertical(X, factor):
    m, __ = X.shape

    if not factor > 1:
        raise ValueError("factor must be > 1")
    if not factor < m:
        raise ValueError(
            "factor must be strictly smaller than the number of input lines"
        )

    Y = np.ascontiguousarray(X.T)
    out = np.asarray([downsample(arr, factor) for arr in Y])
    return np.ascontiguousarray(out.T)


def fold(ts, period, bins, subints=None, epoch=None):
    """
    Fold TimeSeries at given period

    Parameters
    ----------
    ts : TimeSeries
        Input time series to fold
    period : float
        Period in seconds
    bins : int
        Number of phase bins
    subints : int or None, optional
        Number of desired sub-integrations. If None, the number of
        sub-integrations will be the number of full periods that fit in
        the data (default: None)
    epoch : float or None, optional
        Reference folding epoch, as an MJD. It can be arbitrarily far in the
        past or future relative to the data: only its value modulo 'period'
        matters. If given, ``ts.metadata`` must contain an 'mjd' key giving
        the epoch of the first sample of 'ts'. At most one period's worth of
        data is discarded from the start of 'ts' so that the fold is made
        phase-coherent with 'epoch', i.e. phase bin 0 of the output aligns
        with phase 0 of a fold referenced to 'epoch'. If None, the fold
        starts at the first sample of 'ts' (default: None)

    Returns
    -------
    folded : ndarray
        The folded data as a numpy array. If subints > 1, it has a shape
        (subints, bins). Otherwise it is a 1D array with 'bins' elements.

    Raises
    ------
    ValueError: if the data cannot be folded with the requested parameters,
    e.g. bin width is shorter than sampling time, or subint length is shorter
    than requested period, or not enough data is left after aligning the
    fold with 'epoch'
    """
    if epoch is not None:
        mjd = ts.metadata.get("mjd")
        if mjd is None:
            raise ValueError(
                "ts.metadata must contain an 'mjd' epoch in order to fold "
                "with a reference 'epoch'"
            )

        # Time elapsed between 'epoch' and the start of the data, in seconds.
        # This can be arbitrarily large in magnitude (or negative): only its
        # value modulo 'period' matters to align the fold's phase 0 with
        # 'epoch'.
        dt = (mjd - epoch) * 86400.0
        skip = (-dt) % period
        period_samples = int(round(period / ts.tsamp))
        # Rounding 'skip' to the nearest sample can push it up to a full
        # period's worth of samples if it lies within half a sample of
        # 'period' (which floating-point cancellation can easily produce for
        # an 'epoch' far away from the data). Wrap it back down: that edge
        # case is equivalent to no skip at all.
        skip_samples = int(round(skip / ts.tsamp)) % period_samples

        if skip_samples >= ts.nsamp:
            raise ValueError(
                "not enough data to align the fold with the requested "
                "'epoch': the data span is shorter than the phase offset "
                "between its start and 'epoch'"
            )

        if skip_samples:
            ts = type(ts)(ts.data[skip_samples:], ts.tsamp, metadata=ts.metadata)

    if period > ts.length:
        raise ValueError("Period exceeds data length")

    tbin = period / bins
    if not tbin > ts.tsamp:
        raise ValueError("Bin width is shorter than sampling time")

    if subints is not None:
        subints = int(subints)
        if not subints >= 1:
            raise ValueError("subints must be >= 1 or None")

        full_periods = ts.length / period
        if subints > full_periods:
            raise ValueError(
                f"subints ({subints}) exceeds the number of signal periods that fit in the data ({full_periods})"
            )

    factor = tbin / ts.tsamp
    tsdown = ts.downsample(factor)
    m = tsdown.nsamp // bins
    nsamp_eff = m * bins

    folded = tsdown.data[:nsamp_eff].reshape(m, bins)
    folded *= (m * factor) ** -0.5

    if subints == 1 or m == 1:
        return folded.sum(axis=0)
    elif subints is None:
        return folded
    elif subints == m:
        return folded
    else:
        # vertical downsampling factor
        vf = m / subints
        return downsample_vertical(folded, vf)
