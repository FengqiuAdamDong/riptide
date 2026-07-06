"""Pure-numpy reference implementation of the riptide C++ periodogram kernels.

This is a line-for-line port of the C++ headers in ``src/riptide/cpp`` used to
*independently* verify the compiled ``libcpp`` extension. Every kernel in the
chain is reimplemented here in numpy, including the FFA transform/merge:

    downsample.hpp   -> downsampled_size / downsampled_variance / downsample
    kernels.hpp      -> fused_rollback_add / circular_prefix_sum / diff_max
    transforms.hpp   -> merge / transform
    snr.hpp          -> snr1 / snr2
    periodogram.hpp  -> ceilshift / periodogram_length / periodogram /
                        periodogram_gappy

The C++ does the FFA transform and S/N arithmetic in 32-bit float, so this port
keeps float32 internally to reproduce the same rounding; trial periods are kept
in float64 like the C++.

It is NOT meant to be fast - it mirrors the C++ control flow (explicit loops)
for clarity and bit-level fidelity, not performance.
"""
import math

import numpy as np
from matplotlib import pyplot as plt

F32 = np.float32


# ---------------------------------------------------------------------------
# downsample.hpp
# ---------------------------------------------------------------------------
def downsampled_size(num_samples, f):
    """floor(num_samples / f) -- number of samples after downsampling by f."""
    return int(math.floor(num_samples / f))


def downsampled_variance(num_samples, f):
    """Variance of white noise after downsampling by a real-valued factor f."""
    k = math.floor(f)
    r = f - k
    x = downsampled_size(num_samples, f) * r
    if x > 1:
        return f - 1.0 / 3.0
    return (k - 1) ** 2 + 2.0 / 3.0 * x ** 2 - x + 1.0


def downsample(data, f):
    """Downsample a 1D float array by a real-valued factor f > 1.

    Faithful port of riptide::downsample, accumulating in float32 exactly as the
    C++ does.
    """
    data = np.asarray(data, dtype=F32)
    N = data.size
    if not (f > 1.0 and f <= N):
        raise ValueError("Downsampling factor must verify: 1 < f <= size")
    n = downsampled_size(N, f)
    out = np.empty(n, dtype=F32)
    for k in range(n):
        start = k * f
        end = start + f
        imin = int(math.floor(start))
        imax = int(min(math.floor(end), N - 1.0))
        wmin = F32((imin + 1) - start)
        wmax = F32(end - imax)
        acc = wmin * data[imin]
        for i in range(imin + 1, imax):
            acc = F32(acc + data[i])
        acc = F32(acc + wmax * data[imax])
        out[k] = acc
    return out


# ---------------------------------------------------------------------------
# kernels.hpp
# ---------------------------------------------------------------------------
def fused_rollback_add(x, y, shift):
    """z = x + roll(y, -shift), in float32. Port of riptide::fused_rollback_add."""
    return (np.asarray(x, dtype=F32) + np.roll(np.asarray(y, dtype=F32), -int(shift))).astype(F32)


def circular_prefix_sum(x, nsum):
    """Circular prefix sum over nsum elements. Port of riptide::circular_prefix_sum.

    Uses a float64 accumulator within the first wrap (as the C++ does) then casts
    the per-wrap total back to float32.
    """
    x = np.asarray(x, dtype=F32)
    size = x.size
    out = np.empty(nsum, dtype=F32)
    jmax = min(size, nsum)
    acc = 0.0  # double accumulator
    for j in range(jmax):
        acc += float(x[j])
        out[j] = F32(acc)
    if nsum <= size:
        return out
    sumx = F32(acc)
    q = nsum // size
    r = nsum % size
    for i in range(1, q):
        out[i * size:(i + 1) * size] = out[0:size] + F32(i) * sumx
    out[q * size:q * size + r] = out[0:r] + F32(q) * sumx
    return out


def diff_max(a, b, size):
    """max_i (a[i] - b[i]) for i in [0, size). Port of riptide::diff_max."""
    return F32(np.max(np.asarray(a, dtype=F32)[:size] - np.asarray(b, dtype=F32)[:size]))


# ---------------------------------------------------------------------------
# transforms.hpp
# ---------------------------------------------------------------------------
def merge(thead, ttail, out_rows):
    """Merge two FFA sub-transforms into 'out_rows' rows. Port of riptide::merge."""
    thead = np.asarray(thead, dtype=F32)
    ttail = np.asarray(ttail, dtype=F32)
    m = out_rows
    p = thead.shape[1]
    out = np.empty((m, p), dtype=F32)
    kh = F32((thead.shape[0] - 1.0) / (m - 1.0)) if m > 1 else F32(0.0)
    kt = F32((ttail.shape[0] - 1.0) / (m - 1.0)) if m > 1 else F32(0.0)
    for s in range(m):
        # NOTE: keep this arithmetic in strict float32 to reproduce the C++
        # rounding of 'kh * s + 0.5f'. numpy promotes float32 * python-int to
        # float64, which would round differently and shift rows by one.
        h = int(kh * F32(s) + F32(0.5))
        t = int(kt * F32(s) + F32(0.5))
        b = s - (h + t)
        out[s] = fused_rollback_add(thead[h], ttail[t], h + b)
    return out

def transform_gappy(block,gap_rows):
    """FFA transform of a (rows, cols) array. Port of riptide::transform.

    Returns a (rows, cols) array, functionally identical to the C++ recursion
    (the C++ ping-pongs temp/out buffers; here we return arrays instead).
    """
    block = np.asarray(block, dtype=F32)
    m, p = block.shape
    if m == len(gap_rows):
        # print(np.sum(block), "rows are all gaps, returning block")
        # print(m)
        return block
    if m == 1:
        return block.copy()
    if m == 2:
        #this is effectively merge for a block of size 2
        out = np.empty((2, p), dtype=F32)
        out[0] = (block[0] + block[1]).astype(F32)
        out[1] = fused_rollback_add(block[0], block[1], 1)
        return out
    h = m >> 1  # head_size() this is a bitwise shift and is effectively a divide by 2

    gap_rows_head = gap_rows[gap_rows < h]
    gap_rows_tail = gap_rows[gap_rows >= h] - h
    thead = transform_gappy(block[:h],gap_rows_head)
    ttail = transform_gappy(block[h:],gap_rows_tail)
    return merge(thead, ttail, m)


def transform(block):
    """FFA transform of a (rows, cols) array. Port of riptide::transform.

    Returns a (rows, cols) array, functionally identical to the C++ recursion
    (the C++ ping-pongs temp/out buffers; here we return arrays instead).
    """
    block = np.asarray(block, dtype=F32)
    m, p = block.shape
    if m == 1:
        return block.copy()
    if m == 2:
        #this is effectively merge for a block of size 2
        out = np.empty((2, p), dtype=F32)
        out[0] = (block[0] + block[1]).astype(F32)
        out[1] = fused_rollback_add(block[0], block[1], 1)
        return out
    h = m >> 1  # head_size() this is a bitwise shift and is effectively a divide by 2
    thead = transform(block[:h])
    ttail = transform(block[h:])
    return merge(thead, ttail, m)


# ---------------------------------------------------------------------------
# snr.hpp
# ---------------------------------------------------------------------------
def snr1(arr, widths, stdnoise):
    """S/N of a single profile for each trial width. Port of riptide::snr1."""
    arr = np.asarray(arr, dtype=F32)
    size = arr.size
    wmax = int(np.max(widths))
    cpfsum = circular_prefix_sum(arr, size + wmax)
    total = cpfsum[size - 1]  # sum of the input profile
    out = np.empty(len(widths), dtype=F32)
    for iw, w in enumerate(widths):
        w = int(w)
        # boxcar of width w with zero mean and unit square sum
        h = F32(math.sqrt((size - w) / float(size * w)))
        b = F32(w / float(size - w) * h)
        dmax = diff_max(cpfsum[w:], cpfsum, size)
        out[iw] = F32(((h + b) * dmax - b * total) / F32(stdnoise))
    return out


def snr2(block, widths, stdnoise):
    """S/N of every row of a 2D block. Port of riptide::snr2."""
    block = np.asarray(block, dtype=F32)
    out = np.empty((block.shape[0], len(widths)), dtype=F32)
    for i in range(block.shape[0]):
        out[i] = snr1(block[i], widths, stdnoise)
    return out


# ---------------------------------------------------------------------------
# periodogram.hpp
# ---------------------------------------------------------------------------
def ceilshift(rows, cols, pmax):
    """First shift whose trial period reaches pmax. Port of riptide::ceilshift."""
    return int(math.ceil(cols * (rows - 1.0) * (1.0 - cols / pmax)))


def _check_arguments(size, tsamp, period_min, period_max, bins_min, bins_max):
    assert tsamp > 0, "tsamp must be > 0"
    assert period_min > 0, "period_min must be > 0"
    assert period_max > period_min, "period_max must be > period_min"
    assert bins_min > 1, "bins_min must be > 1"
    assert bins_max >= bins_min, "bins_max must be >= bins_min"
    assert period_min >= tsamp * bins_min, "Must have: period_min >= tsamp * bins_min"


def _rows_eval(rows, bins, period_ceil):
    """min(rows, ceilshift(...)) with the C++ size_t semantics (a negative
    ceilshift wraps to a huge value, so min() picks 'rows')."""
    cs = ceilshift(rows, bins, period_ceil)
    if cs < 0:
        return rows
    return min(rows, cs)


def periodogram_length(size, tsamp, period_min, period_max, bins_min, bins_max):
    """Total number of trial periods. Port of riptide::periodogram_length."""
    _check_arguments(size, tsamp, period_min, period_max, bins_min, bins_max)
    ds_ini = period_min / (tsamp * bins_min)
    ds_geo = (bins_max + 1.0) / bins_min
    num_downsamplings = int(math.ceil(math.log(period_max / period_min) / math.log(ds_geo)))
    length = 0
    for ids in range(num_downsamplings):
        f = ds_ini * ds_geo ** ids
        tau = f * tsamp
        period_max_samples = period_max / tau
        n = downsampled_size(size, f)
        bstop = min(bins_max, n, int(period_max_samples))
        for bins in range(bins_min, bstop + 1):
            rows = n // bins
            period_ceil = min(period_max_samples, bins + 1.0)
            length += _rows_eval(rows, bins, period_ceil)
    return length


def periodogram(data, tsamp, widths, period_min, period_max, bins_min, bins_max):
    """Compute a periodogram. Port of riptide::periodogram.

    Returns (periods, foldbins, snrs) like libcpp.periodogram:
      periods  : (length,)              float64
      foldbins : (length,)              uint32
      snrs     : (length, num_widths)   float32
    """
    data = np.asarray(data, dtype=F32)
    size = data.size
    widths = np.asarray(widths)
    _check_arguments(size, tsamp, period_min, period_max, bins_min, bins_max)

    ds_ini = period_min / (tsamp * bins_min)
    ds_geo = (bins_max + 1.0) / bins_min
    num_downsamplings = int(math.ceil(math.log(period_max / period_min) / math.log(ds_geo)))

    periods, foldbins, snr_blocks = [], [], []

    for ids in range(num_downsamplings):
        f = ds_ini * ds_geo ** ids
        tau = f * tsamp
        period_max_samples = period_max / tau
        n = downsampled_size(size, f)

        inp = data if f == 1 else downsample(data, f)

        bstop = min(bins_max, n, int(period_max_samples))
        for bins in range(bins_min, bstop + 1):
            rows = n // bins
            stdnoise = math.sqrt(rows * downsampled_variance(size, f))
            period_ceil = min(period_max_samples, bins + 1.0)
            rows_eval = _rows_eval(rows, bins, period_ceil)

            ffa = transform(inp[:rows * bins].reshape(rows, bins))
            snr_blocks.append(snr2(ffa[:rows_eval], widths, stdnoise))

            for s in range(rows_eval):
                periods.append(tau * bins * bins / (bins - s / (rows - 1.0)))
                foldbins.append(bins)

    periods = np.asarray(periods, dtype=np.float64)
    foldbins = np.asarray(foldbins, dtype=np.uint32)
    if snr_blocks:
        snrs = np.vstack(snr_blocks).astype(F32)
    else:
        snrs = np.empty((0, len(widths)), dtype=F32)
    return periods, foldbins, snrs


def periodogram_gappy(data_list, gaps, tsamp, widths, period_min, period_max,
                      bins_min, bins_max):
    """Gap-aware periodogram. Port of riptide::periodogram_gappy.

    data_list : list of 1D float arrays (one per segment) the data passed in here should be normalised
    gaps      : array of num_data-1 ints (missing samples between segments)
    Returns (periods, foldbins, snrs) like libcpp.periodogram_gappy.
    """
    data_list = [np.asarray(d, dtype=F32) for d in data_list]
    num_data = len(data_list)
    assert num_data >= 1, "num_data must be >= 1"
    sizes = [d.size for d in data_list]
    gaps = [int(g) for g in np.asarray(gaps).ravel()]
    size = sum(sizes) + sum(gaps)
    _check_arguments(size, tsamp, period_min, period_max, bins_min, bins_max)
    #make a new data array and pad with the gaps
    data = np.zeros(size, dtype=F32)
    for i in range(num_data):
        if i == 0:
            data[:sizes[i]] = data_list[i]
        else:
            start = sum(sizes[:i]) + sum(gaps[:i])
            data[start:start + sizes[i]] = data_list[i]

    print("Data size:", size)
    ds_ini = period_min / (tsamp * bins_min)
    ds_geo = (bins_max + 1.0) / bins_min
    num_downsamplings = int(math.ceil(math.log(period_max / period_min) / math.log(ds_geo)))

    periods, foldbins, snr_blocks = [], [], []

    for ids in range(num_downsamplings):
        # print(ids)
        f = ds_ini * ds_geo ** ids
        tau = f * tsamp
        period_max_samples = period_max / tau
        n = downsampled_size(size, f)

        # Locate every gap in the DOWNSAMPLED array exactly as the demo does:
        # the full zero-padded series (seg0 | zeros(gap0) | seg1 | ...) is
        # downsampled as a single unit, so a raw sample position P maps to the
        # output index downsampled_size(P, f). Compute the gap edges from the
        # cumulative raw positions rather than summing per-segment downsampled
        # sizes -- floor(a/f) + floor(b/f) != floor((a+b)/f), and only the
        # whole-array mapping is consistent with inp = downsample(data, f).
        gap_start_indexes = []
        gap_end_indexes = []
        for i in range(len(gaps)):
            raw_seg_end = sum(sizes[:i + 1]) + sum(gaps[:i])  # end of segment i
            gap_start_indexes.append(downsampled_size(raw_seg_end, f))
            gap_end_indexes.append(downsampled_size(raw_seg_end + gaps[i], f))


        # Downsample the full zero-padded array as a single unit, exactly as
        # demo_gappy_timeseries.py does for its zero-pad ground truth. The gap
        # edges computed above index into this same downsampled array.
        inp = data if f == 1 else downsample(data, f)
        #plot to verify
        # plt.figure()
        # plt.plot(inp, label='data')
        # for i in range(len(gap_start_indexes)):
        #     plt.axvspan(gap_start_indexes[i], gap_end_indexes[i], color='red', alpha=0.5, label='gap' if i == 0 else "")
        # plt.show()
        bstop = min(bins_max, n, int(period_max_samples))
        for bins in range(bins_min, bstop + 1):
            # print(bins)
            rows = n // bins
            #calculate which rows are in the gap and which are not
            stdnoise = math.sqrt(rows * downsampled_variance(size, f))
            period_ceil = min(period_max_samples, bins + 1.0)
            rows_eval = _rows_eval(rows, bins, period_ceil)

            gap_rows = []
            #add one to the start subtract one from the end to get the rows that are in the gap
            gap_rows_starts = [(g // bins)+1 for g in gap_start_indexes]
            gap_rows_ends = [(g // bins)-1 for g in gap_end_indexes]
            for i in range(len(gap_start_indexes)):
                gap_row_start = gap_start_indexes[i] // bins
                gap_row_end = gap_end_indexes[i] // bins
                gap_rows.extend(range(gap_row_start+1, gap_row_end))

            gap_rows = np.array(gap_rows, dtype=np.int32)
            ffa = transform_gappy(inp[:rows * bins].reshape(rows, bins),gap_rows)
            snr_blocks.append(snr2(ffa[:rows_eval], widths, stdnoise))

            for s in range(rows_eval):
                periods.append(tau * bins * bins / (bins - s / (rows - 1.0)))
                foldbins.append(bins)

    periods = np.asarray(periods, dtype=np.float64)
    foldbins = np.asarray(foldbins, dtype=np.uint32)
    if snr_blocks:
        snrs = np.vstack(snr_blocks).astype(F32)
    else:
        snrs = np.empty((0, len(widths)), dtype=F32)
    return periods, foldbins, snrs

