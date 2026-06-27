#ifndef PERIODOGRAM_HPP
#define PERIODOGRAM_HPP

#include <cstddef> // size_t
#include <cstdint>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

#include "downsample.hpp"
#include "transforms.hpp"
#include "block.hpp"
#include "snr.hpp"


namespace riptide {


void periodogram_check_arg(bool condition, std::string const& errmsg)
    {
    if (!(condition))
        throw std::invalid_argument(errmsg);
    }


void periodogram_check_arguments(
    size_t size, 
    double tsamp,
    double period_min, 
    double period_max, 
    size_t bins_min, 
    size_t bins_max)
    {    
    periodogram_check_arg(tsamp > 0, "tsamp must be > 0");
    periodogram_check_arg(period_min > 0, "period_min must be > 0");
    periodogram_check_arg(period_max > period_min, "period_max must be > period_min");
    periodogram_check_arg(bins_min > 1, "bins_min must be > 1");
    periodogram_check_arg(bins_max >= bins_min, "bins_max must be >= bins_min");
    periodogram_check_arg(period_min >= tsamp * bins_min, "Must have: period_min >= tsamp * bins_min ");
    // NOTE: we don't check period_max, the search will automatically stop when the maximum allowable trial period is reached
    }


/*
Returns the first shift in an FFA transform that corresponds to a trial period
equal to, or greater than pmax. pmax must be expressed in units of the 
sampling interval.

This function is useful to calculate how many rows of an FFA transform should
be evaluated for S/N, since often we wish to only consider the rows that
correspond to period trials smaller than p + 1. The index of the last row
to evaluate is equal to ceilshift - 1, or equivalently, the total number
of rows to evaluate is equal to ceilshift.
*/
size_t ceilshift(size_t rows, size_t cols, double pmax)
    {
    return ceil(cols * (rows - 1.0) * (1.0 - cols / pmax));
    }


/*
Returns the total number of trial periods in a periodogram
*/
size_t periodogram_length(
    size_t size,
    double tsamp,
    double period_min,
    double period_max,
    size_t bins_min,
    size_t bins_max)
    {
    periodogram_check_arguments(size, tsamp, period_min, period_max, bins_min, bins_max);

    // Initial downsampling factor
    // We want: ds_ini * tsamp * bmin = period_min
    double ds_ini = period_min / (tsamp * bins_min);

    // Geometric growth factor for the downsampling factor
    double ds_geo = (bins_max + 1.0) / bins_min;

    // Number of required downsampling cycles
    size_t num_downsamplings = ceil(log(period_max / period_min) / log(ds_geo));
    size_t length = 0; // total number of period trials, to be calculated

    /* Downsampling loop */
    for (size_t ids = 0; ids < num_downsamplings; ++ids)
        {
        const double f = ds_ini * pow(ds_geo, ids); // current downsampling factor
        const double tau = f * tsamp; // current sampling time
        const double period_max_samples = period_max / tau;
        const size_t n = downsampled_size(size, f); // current number of input samples

        // Min and max number of bins with which to FFA transform in order to
        // cover all trial periods between period_min and period_max.
        // NOTE: bstop is INclusive
        // Also, we MUST enforce bstop <= n, to avoid doing an FFA transform with 0 rows
        const size_t bstart = bins_min;
        const size_t bstop = std::min({ bins_max, n, size_t(period_max_samples) });

        /* FFA transform loop */
        for (size_t bins = bstart; bins <= bstop; ++bins)
            {
            const size_t rows = n / bins;
            const double period_ceil = std::min(period_max_samples, bins + 1.0);
            const size_t rows_eval = std::min(rows, ceilshift(rows, bins, period_ceil));
            length += rows_eval;
            }
        }
    return length;
    }


/*
Compute the periodogram of a time series that has been normalised to zero mean and unit variance.
Outputs are: trial periods (num_periods elements), number of phase bins used in the fold (num_periods elements), 
and signal to noise ratio (num_periods * num_widths elements)
*/
void periodogram(
    const float* __restrict__ data,
    size_t size,
    double tsamp,
    const size_t* __restrict__ widths,
    size_t num_widths,
    double period_min,
    double period_max,
    size_t bins_min,
    size_t bins_max,
    double* __restrict__ periods,
    uint32_t* __restrict__ foldbins,
    float* __restrict__ snr)
    {
    periodogram_check_arguments(size, tsamp, period_min, period_max, bins_min, bins_max);

    // Initial downsampling factor
    // We want: ds_ini * tsamp * bmin = period_min
    double ds_ini = period_min / (tsamp * bins_min);

    // Geometric growth factor for the downsampling factor
    double ds_geo = (bins_max + 1.0) / bins_min;

    // Number of required downsampling cycles
    size_t num_downsamplings = ceil(log(period_max / period_min) / log(ds_geo));

    // Allocate buffers
    const size_t bufsize = downsampled_size(size, ds_ini);
    std::unique_ptr<float[]> input_mem(new float[bufsize]);
    std::unique_ptr<float[]> ffabuf_mem(new float[bufsize]);
    std::unique_ptr<float[]> ffaout_mem(new float[bufsize]);
    const float* input = input_mem.get();
    float* ffabuf = ffabuf_mem.get();
    float* ffaout = ffaout_mem.get();

    /* Downsampling loop */
    for (size_t ids = 0; ids < num_downsamplings; ++ids)
        {
        const double f = ds_ini * pow(ds_geo, ids); // current downsampling factor
        const double tau = f * tsamp; // current sampling time
        const double period_max_samples = period_max / tau;
        const size_t n = downsampled_size(size, f); // current number of input samples

        // downsample() requires f > 1, but we still allow searching the data at their
        // original resolution.
        if (f == 1) {
            input = data;
        }            
        else {
            downsample(data, size, f, input_mem.get());
            input = input_mem.get();
        }

        // Min and max number of bins with which to FFA transform in order to
        // cover all trial periods between period_min and period_max.
        // NOTE: bstop is INclusive
        // Also, we MUST enforce bstop <= n, to avoid doing an FFA transform with 0 rows
        const size_t bstart = bins_min;
        const size_t bstop = std::min({ bins_max, n, size_t(period_max_samples) });

        /* FFA transform loop */
        for (size_t bins = bstart; bins <= bstop; ++bins)
            {
            const size_t rows = n / bins;
            const float stdnoise = sqrt(rows * downsampled_variance(size, f));
            const double period_ceil = std::min(period_max_samples, bins + 1.0);
            const size_t rows_eval = std::min(rows, ceilshift(rows, bins, period_ceil));

            transform(input, rows, bins, ffabuf, ffaout);
            
            auto block = ConstBlock(ffaout, rows_eval, bins);
            snr2(block, widths, num_widths, stdnoise, snr);

            for (size_t s = 0; s < rows_eval; ++s)
                {
                periods[s] = tau * bins * bins / (bins - s / (rows - 1.0));
                foldbins[s] = bins;
                }

            snr += rows_eval * num_widths;
            periods += rows_eval;
            foldbins += rows_eval;
            }
        }
    }


/*
Like periodogram(), but for a "gappy" time series made of several separate data
segments that are not contiguous in time.

Instead of a single contiguous array, the caller passes:
  - 'data'  : an array of 'num_data' pointers, one per segment. Each segment must
              be a contiguous float time series, normalised to zero mean and unit
              variance.
  - 'sizes' : an array of 'num_data' sample counts; sizes[i] is the number of
              samples in segment data[i].
  - 'gaps'  : an array of 'num_data - 1' sample counts; gaps[i] is the number of
              MISSING samples (at the original 'tsamp' resolution) between the end
              of segment i and the start of segment i + 1.

This function deliberately STOPS after FFA-transforming each segment, so that the
gap-aware combination of the per-segment transforms (and the subsequent S/N
evaluation) can be filled in by hand. See the clearly marked manual-edit section
inside the FFA transform loop.
*/
void periodogram_gappy(
    const float* const* __restrict__ data,
    const size_t* __restrict__ sizes,
    const size_t* __restrict__ gaps,
    size_t num_data,
    double tsamp,
    const size_t* __restrict__ widths,
    size_t num_widths,
    double period_min,
    double period_max,
    size_t bins_min,
    size_t bins_max,
    double* __restrict__ periods,
    uint32_t* __restrict__ foldbins,
    float* __restrict__ snr)
    {
    periodogram_check_arg(num_data >= 1, "num_data must be >= 1");

    // Total span of the equivalent gappy series in samples: the data samples plus
    // the missing samples that fall in the gaps. This drives the downsampling
    // schedule, playing the same role as 'size' does in periodogram().
    size_t total_size = 0;
    for (size_t i = 0; i < num_data; ++i)
        total_size += sizes[i];
    for (size_t i = 0; i + 1 < num_data; ++i)
        total_size += gaps[i];

    periodogram_check_arguments(total_size, tsamp, period_min, period_max, bins_min, bins_max);

    // Initial downsampling factor
    // We want: ds_ini * tsamp * bmin = period_min
    double ds_ini = period_min / (tsamp * bins_min);

    // Geometric growth factor for the downsampling factor
    double ds_geo = (bins_max + 1.0) / bins_min;

    // Number of required downsampling cycles
    size_t num_downsamplings = ceil(log(period_max / period_min) / log(ds_geo));

    // Per-segment work buffers, each large enough to hold its segment at the
    // finest (initial) downsampling factor, i.e. its largest downsampled size.
    // Allocated once and reused across the downsampling loop. The per-segment
    // FFA outputs (ffaout_mem) persist between iterations so they remain
    // available to the manual-edit section all at once.
    std::vector<std::unique_ptr<float[]>> input_mem(num_data);
    std::vector<std::unique_ptr<float[]>> ffabuf_mem(num_data);
    std::vector<std::unique_ptr<float[]>> ffaout_mem(num_data);
    for (size_t i = 0; i < num_data; ++i)
        {
        const size_t bufsize = downsampled_size(sizes[i], ds_ini);
        input_mem[i].reset(new float[bufsize]);
        ffabuf_mem[i].reset(new float[bufsize]);
        ffaout_mem[i].reset(new float[bufsize]);
        }

    /* Downsampling loop */
    for (size_t ids = 0; ids < num_downsamplings; ++ids)
        {
        const double f = ds_ini * pow(ds_geo, ids); // current downsampling factor
        const double tau = f * tsamp; // current sampling time
        const double period_max_samples = period_max / tau;

        // Downsample every segment to the current resolution, and express each
        // gap in the current (downsampled) number of samples.
        std::vector<const float*> input(num_data);
        std::vector<size_t> n(num_data); // downsampled sample count per segment
        for (size_t i = 0; i < num_data; ++i)
            {
            n[i] = downsampled_size(sizes[i], f);

            // downsample() requires f > 1, but we still allow searching the data
            // at their original resolution.
            if (f == 1)
                {
                input[i] = data[i];
                }
            else
                {
                downsample(data[i], sizes[i], f, input_mem[i].get());
                input[i] = input_mem[i].get();
                }
            }

        std::vector<size_t> gap(num_data > 0 ? num_data - 1 : 0);
        for (size_t i = 0; i + 1 < num_data; ++i)
            gap[i] = (size_t) llround(gaps[i] / f);

        // The shortest segment caps the number of phase bins we can FFA transform
        // with, to avoid a transform with 0 rows on any segment.
        size_t nmin = n[0];
        for (size_t i = 1; i < num_data; ++i)
            nmin = std::min(nmin, n[i]);

        // Min and max number of bins with which to FFA transform in order to
        // cover all trial periods between period_min and period_max.
        // NOTE: bstop is INclusive.
        const size_t bstart = bins_min;
        const size_t bstop = std::min({ bins_max, nmin, size_t(period_max_samples) });

        /* FFA transform loop */
        for (size_t bins = bstart; bins <= bstop; ++bins)
            {
            // FFA transform each segment independently into its own persistent
            // ffaout buffer. After this loop, every segment's transform is
            // available simultaneously, ready to be combined across the gaps.
            std::vector<size_t> rows(num_data);
            for (size_t i = 0; i < num_data; ++i)
                {
                rows[i] = n[i] / bins;
                transform(input[i], rows[i], bins, ffabuf_mem[i].get(), ffaout_mem[i].get());
                }

            // ================== STOP: manual code edits below ==================
            //
            // For the current trial number of phase bins 'bins', you now have,
            // for each segment i in [0, num_data):
            //   - ConstBlock(ffaout_mem[i].get(), rows[i], bins)
            //         the FFA transform of segment i; row s is the folded profile
            //         for shift s (trial period bins*bins / (bins - s/(rows-1))).
            //   - gap[i] (for i < num_data - 1)
            //         the gap before segment i+1, in samples at the current
            //         resolution 'tau'.
            // plus: tau, bins, rows, period_max_samples, period_min/max,
            //       widths/num_widths.
            //
            // TODO (by hand):
            //   1. Combine the per-segment transforms across the gaps into a
            //      single set of folded profiles with consistent phase, taking
            //      gap[i] into account when aligning shifts between segments.
            //   2. Evaluate S/N of the combined profiles with snr2(), using the
            //      appropriate stdnoise.
            //   3. Write the trial periods into periods[], the fold bins into
            //      foldbins[], the S/N into snr[], and advance the periods,
            //      foldbins and snr pointers as periodogram() does.
            //
            // ===================================================================
            }
        }
    }


} // namespace riptide

#endif // PERIODOGRAM_HPP