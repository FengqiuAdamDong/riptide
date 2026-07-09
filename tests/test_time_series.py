import os
import tempfile

import numpy as np
from pytest import raises, warns
from riptide import TimeSeries, Metadata, save_json, load_json


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FLOAT_ATOL = 1.0e-6


# NOTE: a TimeSeries has only two basic attributes: data and tsamp
# *** That's only what we test here ***
# Anything else is handled by the Metadata class
def test_presto():
    def check_data(ts, refdata):
        assert ts.nsamp == 16
        assert ts.tsamp == 64e-6
        assert ts.data.dtype == np.float32
        assert np.allclose(ts.data, refdata)

    # The actual data expected to be in all test .dat files
    refdata = np.arange(16)

    fname = os.path.join(DATA_DIR, "fake_presto_radio.inf")
    ts = TimeSeries.from_presto_inf(fname)
    check_data(ts, refdata)

    fname = os.path.join(DATA_DIR, "fake_presto_radio_breaks.inf")
    ts = TimeSeries.from_presto_inf(fname)
    check_data(ts, refdata)

    # Calling TimeSeries.from_presto_inf() on X-ray and Gamma data should raise a warning
    # about the noise stats being non-Gaussian
    with warns(UserWarning):
        fname = os.path.join(DATA_DIR, "fake_presto_xray.inf")
        ts = TimeSeries.from_presto_inf(fname)
        check_data(ts, refdata)


def test_sigproc():
    refdata = np.arange(16)  # what is supposed to be in the data
    filenames = [
        "fake_sigproc_float32.tim",
        "fake_sigproc_uint8.tim",
        "fake_sigproc_int8.tim",
    ]

    for fname in filenames:
        fname = os.path.join(DATA_DIR, fname)
        ts = TimeSeries.from_sigproc(fname)
        assert ts.nsamp == 16
        assert ts.tsamp == 64e-6
        assert ts.data.dtype == np.float32
        assert np.allclose(ts.data, refdata)

    # Check that trying to read 8-bit SIGPROC data without a 'signed'
    # header key raises an error
    with raises(ValueError):
        fname = os.path.join(DATA_DIR, "fake_sigproc_uint8_nosignedkey.tim")
        ts = TimeSeries.from_sigproc(fname)


def test_numpy_binary():
    refdata = np.arange(16)
    tsamp = 64e-6

    def check_ts(ts):
        assert ts.nsamp == refdata.size
        assert ts.tsamp == tsamp
        assert ts.data.dtype == np.float32
        assert np.allclose(ts.data, refdata)

    ts = TimeSeries.from_numpy_array(refdata, tsamp)
    check_ts(ts)

    with tempfile.NamedTemporaryFile(suffix=".npy") as f:
        # re-creates the file, still gets deleted on exiting context mgr
        np.save(f.name, refdata)
        ts = TimeSeries.from_npy_file(f.name, tsamp)
        check_ts(ts)

    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        # re-creates the file, still gets deleted on exiting context mgr
        refdata.astype(np.float32).tofile(f.name)
        ts = TimeSeries.from_binary(f.name, tsamp)
        check_ts(ts)


def test_generate():
    length = 10.0  # s
    tsamp = 0.01  # s
    period = 1.0  # s
    amplitude = 25.0

    # Generate noiseless data to check its amplitude
    ts = TimeSeries.generate(length, tsamp, period, amplitude=amplitude, stdnoise=0)

    assert ts.length == length
    assert ts.tsamp == tsamp
    assert ts.data.dtype == np.float32
    assert np.allclose(sum(ts.data**2) ** 0.5, amplitude, atol=FLOAT_ATOL)


def test_methods():
    """
    NOTE: This tests that the code runs, but not the output data quality,
    i.e. if dereddening removes low-frequency noise well
    """
    length = 10.0  # s
    tsamp = 1.0e-3  # s
    period = 1.0  # s
    amplitude = 25.0
    stdnoise = 1.0

    tsorig = TimeSeries.generate(
        length, tsamp, period, amplitude=amplitude, stdnoise=stdnoise
    )
    ts = tsorig.copy()

    ### Normalisation inplace / out of place ###
    tscopy = ts.copy()
    ts = ts.normalise()
    tscopy.normalise(inplace=True)

    assert np.allclose(ts.data.mean(), 0.0, atol=FLOAT_ATOL)
    assert np.allclose(ts.data.std(), 1.0, atol=FLOAT_ATOL)
    assert np.allclose(ts.data, tscopy.data, atol=FLOAT_ATOL)

    ### Dereddening inplace / out of place ###
    tscopy = ts.copy()
    ts = ts.deredden(width=0.5, minpts=51)
    tscopy.deredden(width=0.5, minpts=51, inplace=True)
    assert np.allclose(ts.data, tscopy.data, atol=FLOAT_ATOL)

    # De-reddening should turn constant data into zeros
    tsconst = TimeSeries.generate(length, tsamp, period, amplitude=0, stdnoise=0)
    tsconst._data += 42.42
    assert np.allclose(tsconst.deredden(0.5, minpts=51).data, 0.0, atol=FLOAT_ATOL)

    ### Downsampling ##
    dsfactor = 10
    ts = tsorig.downsample(dsfactor)
    tscopy = tsorig.copy()
    tscopy.downsample(dsfactor, inplace=True)

    assert ts.tsamp == tsorig.tsamp * dsfactor
    assert ts.nsamp == tsorig.nsamp // dsfactor
    assert ts.length == tsorig.length

    assert tscopy.tsamp == tsorig.tsamp * dsfactor
    assert tscopy.nsamp == tsorig.nsamp // dsfactor
    assert tscopy.length == tsorig.length

    with raises(ValueError):  # stricly < 1
        ts = tsorig.downsample(0.55)

    with raises(ValueError):  # excessive
        ts = tsorig.downsample(tsorig.nsamp * 10)

    ### Folding ###
    bins = 100

    # Fold with nsubs = None (default to number of periods that fit in data)
    X10 = tsorig.fold(1.0, bins, subints=None)
    assert X10.shape == (10, bins)

    # Fold with nsubs < number of periods
    X2 = tsorig.fold(1.0, bins, subints=2)
    assert X2.shape == (2, bins)

    # Fold with nsubs = number of periods that fit in data
    # This is a special case where internally fold() has to avoid downsampling along the time axis
    m = int(length / period)
    Xm = tsorig.fold(1.0, bins, subints=m)

    # Fold into a single 1D profile array
    prof = tsorig.fold(1.0, bins, subints=1)

    # All methods should return the same folded profile
    assert np.allclose(prof, X2.sum(axis=0), atol=FLOAT_ATOL)
    assert np.allclose(prof, X10.sum(axis=0), atol=FLOAT_ATOL)
    assert np.allclose(prof, Xm.sum(axis=0), atol=FLOAT_ATOL)

    # Too many requested subints
    with raises(ValueError):
        Xerr = tsorig.fold(1.0, bins, subints=1000000)

    # subints can't be < 1
    with raises(ValueError):
        Xerr = tsorig.fold(1.0, bins, subints=0)

    # Too many requested bins
    with raises(ValueError):
        Xerr = tsorig.fold(1.0, 1000000, subints=None)

    # Period too long
    with raises(ValueError):
        Xerr = tsorig.fold(1.0e6, bins, subints=None)

    # Period too short
    with raises(ValueError):
        Xerr = tsorig.fold(1.0e-6, bins, subints=None)


def test_fold_epoch():
    """
    Check that folding two segments of the same underlying periodic signal,
    with a shared reference 'epoch', produces phase-coherent profiles: the
    signal peak lands on the same phase bin in both, regardless of where
    each segment starts. Also check that this holds even when 'epoch' is
    arbitrarily far away (only its value modulo 'period' matters), and that
    omitting 'epoch' breaks this coherence.
    """
    length = 20.0  # s, i.e. 20 signal periods
    tsamp = 1.0e-3  # s
    period = 1.0  # s
    bins = 200
    phi0 = 0.37
    ducy = 0.05
    amplitude = 50.0
    mjd0 = 60000.0

    full = TimeSeries.generate(
        length, tsamp, period, phi0=phi0, ducy=ducy, amplitude=amplitude, stdnoise=0.0
    )

    # Arbitrary cut point, deliberately NOT a multiple of the period, so that
    # the two segments start at different signal phases.
    istart = 13425
    assert istart % int(round(period / tsamp)) != 0

    meta_before = Metadata({**dict(full.metadata), "mjd": mjd0})
    meta_after = Metadata(
        {**dict(full.metadata), "mjd": mjd0 + istart * tsamp / 86400.0}
    )
    before = TimeSeries(full.data[:istart], tsamp, copy=True, metadata=meta_before)
    after = TimeSeries(full.data[istart:], tsamp, copy=True, metadata=meta_after)

    # Folded with a common reference epoch, both segments' peaks should land
    # on (near enough) the same phase bin.
    prof_before = before.fold(period, bins, subints=1, epoch=mjd0)
    prof_after = after.fold(period, bins, subints=1, epoch=mjd0)
    peak_before = int(np.argmax(prof_before))
    peak_after = int(np.argmax(prof_after))
    assert abs(peak_before - peak_after) <= 1

    # Without a shared epoch, the 'after' segment folds starting at its own
    # first sample, which is out of phase with 'before': the peak should
    # land on a noticeably different bin.
    prof_after_noepoch = after.fold(period, bins, subints=1)
    peak_after_noepoch = int(np.argmax(prof_after_noepoch))
    assert abs(peak_before - peak_after_noepoch) > 10

    # The reference epoch can be arbitrarily far in the past or future:
    # only its value modulo 'period' matters.
    epoch_far_past = mjd0 - 1.0e6 * period / 86400.0
    epoch_far_future = mjd0 + 1.0e6 * period / 86400.0
    prof_far_past = before.fold(period, bins, subints=1, epoch=epoch_far_past)
    prof_far_future = before.fold(period, bins, subints=1, epoch=epoch_far_future)
    assert np.allclose(prof_before, prof_far_past, atol=FLOAT_ATOL)
    assert np.allclose(prof_before, prof_far_future, atol=FLOAT_ATOL)

    # 'epoch' requires an 'mjd' entry in the metadata
    no_mjd = TimeSeries(full.data, tsamp)
    with raises(ValueError):
        no_mjd.fold(period, bins, epoch=mjd0)


def test_serialization():
    length = 10.0  # s
    tsamp = 1.0e-3  # s
    period = 1.0  # s
    amplitude = 25.0
    stdnoise = 1.0

    ts = TimeSeries.generate(
        length,
        tsamp,
        period,
        stdnoise=stdnoise,
        amplitude=amplitude,
    )

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        save_json(f.name, ts)
        tscopy = load_json(f.name)

    assert ts.tsamp == tscopy.tsamp
    assert ts.nsamp == tscopy.nsamp
    assert ts.length == tscopy.length
    assert np.allclose(ts.data, tscopy.data, atol=FLOAT_ATOL)
