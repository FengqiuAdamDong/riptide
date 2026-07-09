#ifndef DOWNSAMPLE_HPP
#define DOWNSAMPLE_HPP

#include <cstddef> // size_t
#include <cmath>
#include <cstring>   // memcpy()
#include <stdexcept>
#include <algorithm> // min(), max(), fill()
#include <vector>

namespace riptide {

// Check that 1 < f <= size; throw std::invalid_argument if that is not the case
void check_downsampling_factor(size_t size, double f)
    {
    bool valid = (f > 1.0) & (f <= size);
    if (!valid)
        throw std::invalid_argument("Downsampling factor must verify: 1 < f <= size");
    }

/*
Number of samples in a time series after it has been downsampled by a real-valued factor f
*/
size_t downsampled_size(size_t num_samples, double f)
    {
    return floor(num_samples / f);
    }

/*
Variance of background Gaussian noise after downsampling a time series with 'num_samples' elements by a real-valued factor f.
*/
double downsampled_variance(size_t num_samples, double f)
    {
    const double k = floor(f);
    const double r = f - k;
    const double x = downsampled_size(num_samples, f) * r;
    if (x > 1)
        return f - 1.0 / 3.0;
    else
        return pow(k-1, 2) + 2.0/3.0 * pow(x, 2) - x + 1.0;
    }


/* 
Downsample input array by a real-valued factor. Output must have capacity for floor((size - 1.0) / f) elements
*/
void downsample(const float* __restrict__ in, size_t size, double f, float* __restrict__ out)
    {
    check_downsampling_factor(size, f);

    // N = number of input samples
    // n = number of valid output samples
    const size_t N = size;
    const size_t n = downsampled_size(N, f);

    // k = output sample index
    for (size_t k = 0; k < n; ++k)
        {
        // minimum and maximum x-coordinate (real-valued)
        // of the input data range required to compute output sample k
        const double start = k * f;
        const double end = start + f;

        // minimum (inclusive) and maximum (inclusive)
        // input sample indices that must be read to compute output sample k
        const size_t imin = floor(start);
        // NOTE: floor() is OK in imax calculation, 
        // because the input sample index imax covers the input x-coordinate range [imax, imax+1]
        // NOTE 2: when f divides size, for the last output sample we have end = N,
        // hence the need to enforce imax < N
        const size_t imax = std::min(floor(end), N - 1.0);

        // Weights to be applied to samples imin and imax
        // Other input samples are weighted by 1
        const float wmin = (imin + 1) - start; // ceil(k*f) - k*f
        const float wmax = end - imax;         // (k+1)*f - floor((k+1)*f)

        float acc = wmin * in[imin];
        for (size_t i = imin + 1; i < imax; ++i)
            acc += in[i];
        acc += wmax * in[imax];

        out[k] = acc;
        }
    }


/*
Downsample a gappy time series by a real-valued factor f, touching only the
real samples.

Inputs describe the segments exactly as in periodogram_gappy(): 'data' holds
'num_data' segment pointers, 'sizes' their sample counts, 'gaps' the number of
missing samples between consecutive segments, and 'total_size' the span of the
equivalent zero-padded series (sum of sizes and gaps).

Outputs one downsampled "piece" per segment (ds_segs[i]) plus the global
downsampled index at which it starts (ds_starts[i]). Adding the pieces into a
zero array of length downsampled_size(total_size, f) is bitwise identical to
downsample() applied to the fully zero-padded series
(seg0 | zeros(gap0) | seg1 | ...): each segment is zero-padded out to the
enclosing global window boundaries and downsampled with the global window
phase -- window starts are computed as (k_first + k) * f - a, the identical
double product the whole-array code would form, minus an exactly-representable
integer, and the zero padding adds exact 0.0f into the float accumulator. A
window straddling two segments across a gap smaller than f appears in both
pieces as partial sums; adding overlapping pieces in segment order reproduces
the whole-array accumulation. With f == 1 the pieces are plain copies of the
segments and ds_starts are their raw sample offsets.

Faithful port of periodogram_py.downsample_gappy.
*/
void downsample_gappy(
    const float* const* __restrict__ data,
    const size_t* __restrict__ sizes,
    const size_t* __restrict__ gaps,
    size_t num_data,
    size_t total_size,
    double f,
    std::vector<std::vector<float>>& ds_segs,
    std::vector<size_t>& ds_starts)
    {
    ds_segs.assign(num_data, std::vector<float>());
    ds_starts.assign(num_data, 0);

    if (f == 1)
        {
        size_t S = 0;
        for (size_t i = 0; i < num_data; ++i)
            {
            ds_segs[i].assign(data[i], data[i] + sizes[i]);
            ds_starts[i] = S;
            S += sizes[i] + (i + 1 < num_data ? gaps[i] : 0);
            }
        return;
        }

    const size_t n = downsampled_size(total_size, f);
    if (n == 0) // no complete output window; all pieces stay empty
        return;
    std::vector<float> seg_pad;

    size_t S = 0; // global raw start of the current segment
    for (size_t i = 0; i < num_data; ++i)
        {
        const size_t E = S + sizes[i]; // global raw end (exclusive)

        // window k spans global [k*f, (k+1)*f); those overlapping [S, E):
        const size_t k_first = floor(S / f);
        const size_t k_last = std::min<size_t>(n - 1, size_t(ceil(E / f)) - 1);
        ds_starts[i] = k_first;
        if (k_last < k_first) // segment past the last complete window
            {
            S = E + (i + 1 < num_data ? gaps[i] : 0);
            continue;
            }

        const size_t a = floor(k_first * f); // first raw index any window reads
        const size_t b = std::min(floor((k_last + 1) * f), total_size - 1.0); // last raw index read
        seg_pad.assign(b - a + 1, 0.0f);
        // tail samples past b are read by no window (the whole-array downsample
        // ignores them too); only the last segment can hit this
        const size_t stop = std::min(E, b + 1);
        memcpy(seg_pad.data() + (S - a), data[i], (stop - S) * sizeof(float));

        const size_t nloc = seg_pad.size();
        std::vector<float>& out = ds_segs[i];
        out.resize(k_last - k_first + 1);
        for (size_t k = 0; k < k_last - k_first + 1; ++k)
            {
            // the global window position, rebased by the integer 'a' (exact)
            const double start = (k_first + k) * f - a;
            const double end = start + f;
            const size_t imin = floor(start);
            const size_t imax = std::min(floor(end), nloc - 1.0);
            const float wmin = (imin + 1) - start;
            const float wmax = end - imax;

            float acc = wmin * seg_pad[imin];
            for (size_t j = imin + 1; j < imax; ++j)
                acc += seg_pad[j];
            acc += wmax * seg_pad[imax];
            out[k] = acc;
            }

        S = E + (i + 1 < num_data ? gaps[i] : 0);
        }
    }

} // namespace riptide

#endif // DOWNSAMPLE_HPP
