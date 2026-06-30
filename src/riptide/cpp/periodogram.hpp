#ifndef PERIODOGRAM_HPP
#define PERIODOGRAM_HPP

#include <cstddef> // size_t
#include <cstdint>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstdio>

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
Debug helper: dump a (rows x cols) row-major float buffer to a raw binary file
so it can be loaded and plotted as a 2D intensity map in Python. The shape is
encoded in the filename, e.g. "debug_ffaout_seg0_r123_c240.bin", and the payload
is rows*cols little-endian float32 values, no header. Load with:
    np.fromfile(path, dtype=np.float32).reshape(rows, cols)
*/
inline void dump_block_bin(const char* path, const float* data, size_t rows, size_t cols)
    {
    FILE* fp = fopen(path, "wb");
    if (!fp)
        {
        // fprintf(stderr, "dump_block_bin: could not open %s for writing\n", path);
        return;
        }
    fwrite(data, sizeof(float), rows * cols, fp);
    fclose(fp);
    // fprintf(stderr, "dump_block_bin: wrote %s (%zu x %zu)\n", path, rows, cols);
    // fflush(stderr);
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
    //print every element of the data so that I can see what it looks likes
    // printf("num_data: %zu\n", num_data);
    // for (size_t i = 0; i < num_data; ++i)
    //     {
    //       for (size_t j = 0; j < sizes[i]; ++j)
    //         {
    //           printf("data[%zu][%zu]: %f\n", i, j, data[i][j]);
    //         }
    //     }

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

    // FFA output buffer for the whole gappy series treated as a single,
    // zero-padded-across-the-gaps contiguous series. Sized to the full span
    // (data + gaps) at the finest (initial) downsampling factor, i.e. its
    // largest possible downsampled size.
    const size_t total_bufsize = downsampled_size(total_size, ds_ini);
    std::unique_ptr<float[]> ffaout_total(new float[total_bufsize]);

    /* Downsampling loop */
    for (size_t ids = 0; ids < num_downsamplings; ++ids)
        {
        // fprintf(stderr, "[ds %zu/%zu] starting downsampling loop\n", ids, num_downsamplings);
        const double f = ds_ini * pow(ds_geo, ids); // current downsampling factor
        const double tau = f * tsamp; // current sampling time
        const double period_max_samples = period_max / tau;
        const size_t n_total = downsampled_size(total_size, f);

        // fprintf(stderr,
            // "[ds %zu/%zu] f=%.4f tau=%.6e period_max_samples=%.2f n_total=%zu total_bufsize=%zu\n",
            // ids, num_downsamplings, f, tau, period_max_samples, n_total, total_bufsize);
        // fflush(stderr);

        // Downsample every segment to the current resolution, and express each
        // gap in the current (downsampled) number of samples.
        std::vector<const float*> input(num_data);
        std::vector<size_t> n(num_data); // downsampled sample count per segment
        for (size_t i = 0; i < num_data; ++i)
            {
            n[i] = downsampled_size(sizes[i], f);

            // fprintf(stderr,
                // "  [ds %zu] downsampling segment %zu: sizes[i]=%zu -> n[i]=%zu (f=%.4f)\n",
                // ids, i, sizes[i], n[i], f);
            // fflush(stderr);

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
            {
            gap[i] = (size_t) llround(gaps[i] / f);
            // fprintf(stderr, "  [ds %zu] gap[%zu] = %zu (orig %zu)\n",
                // ids, i, gap[i], gaps[i]);
            // fflush(stderr);
            }

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

        // fprintf(stderr, "  [ds %zu] nmin=%zu bstart=%zu bstop=%zu\n",
            // ids, nmin, bstart, bstop);
        // fflush(stderr);

        /* FFA transform loop */
        for (size_t bins = bstart; bins <= bstop; ++bins)
            {
            // FFA transform each segment independently into its own persistent
            // ffaout buffer. After this loop, every segment's transform is
            // available simultaneously, ready to be combined across the gaps.
            // fprintf(stderr, "  [ds %zu/%zu] FFA transform loop: bins=%zu/%zu\n", ids, num_downsamplings, bins, bstop);
            // fflush(stderr);


            std::vector<size_t> rows(num_data);
            std::vector<float> stdnoise(num_data);
            std::vector<double> period_ceil(num_data);
            std::vector<size_t> rows_eval(num_data);

            for (size_t i = 0; i < num_data; ++i)
                {
                rows[i] = n[i] / bins;
                stdnoise[i] = sqrt(rows[i] * downsampled_variance(sizes[i], f));
                period_ceil[i] = std::min(period_max_samples, bins + 1.0);
                rows_eval[i] = std::min(rows[i], ceilshift(rows[i], bins, period_ceil[i]));

                const size_t seg_bufsize = downsampled_size(sizes[i], ds_ini);
                // fprintf(stderr,
                    // "    [ds %zu bins %zu] transform seg %zu: rows=%zu rows_eval=%zu "
                    // "writes=%zu floats, seg_bufsize=%zu %s\n",
                    // ids, bins, i, rows[i], rows_eval[i], rows[i] * bins, seg_bufsize,
                    // (rows[i] * bins > seg_bufsize) ? "*** OVERFLOW ***" : "");
                // fflush(stderr);

                transform(input[i], rows[i], bins, ffabuf_mem[i].get(), ffaout_mem[i].get());

                // fprintf(stderr, "    [ds %zu bins %zu] transform seg %zu OK\n", ids, bins, i);
                // fflush(stderr);
                }

            // Dump the per-segment FFA transforms for ONE iteration so they can
            // be plotted as 2D intensity maps. Gated to the first downsampling
            // and first bins value to avoid flooding the disk; change the
            // condition to capture a different (ids, bins) slice.
            // const bool dump_this_iter = (ids == 0 && bins == bstart);
            // if (dump_this_iter)
            //     {
            //     for (size_t i = 0; i < num_data; ++i)
            //         {
            //         char path[256];
            //         snprintf(path, sizeof(path),
            //             "debug_ffaout_seg%zu_r%zu_c%zu.bin", i, rows[i], bins);
            //         dump_block_bin(path, ffaout_mem[i].get(), rows[i], bins);
            //         }
            //     }
            //time to merge all the ffaout_mems together

            const size_t total_rows = n_total / bins;
            // IMPORTANT: cap the evaluated rows by min(period_max_samples, bins + 1),
            // exactly as periodogram_length() does. The 'bins + 1' term restricts
            // each transform to trial periods in [bins, bins + 1) so successive bins
            // values don't produce overlapping (and over-counted) trials. Without
            // it, total_rows_eval balloons up to total_rows and we write far more
            // trials than the output arrays were sized for -> buffer overrun.
            const double total_period_ceil = std::min(period_max_samples, bins + 1.0);
            const size_t total_rows_eval = std::min(total_rows, ceilshift(total_rows, bins, total_period_ceil));

            // fprintf(stderr,
            //     "  [ds %zu bins %zu] total_rows=%zu total_rows_eval=%zu "
            //     "merge writes=%zu floats into ffaout_total (bufsize=%zu) %s\n",
            //     ids, bins, total_rows, total_rows_eval, total_rows_eval * bins, total_bufsize,
            //     (total_rows_eval * bins > total_bufsize) ? "*** OVERFLOW ***" : "");
            // fflush(stderr);

            for (size_t i = 0; i < num_data-1; ++i)
                {
                //merge ffaout_mem[i] and ffaout_mem[i+1] into ffaout_total
                //
                // fprintf(stderr,
                    // "    merging seg %zu (rows_eval=%zu) + seg %zu (rows_eval=%zu) "
                    // "-> ffaout_total (out_rows=%zu, bins=%zu, f=%.4f)\n",
                    // i, rows_eval[i], i+1, rows_eval[i+1], total_rows_eval, bins, f);
                // fflush(stderr);
                merge_gappy(ConstBlock(ffaout_mem[i].get(), rows_eval[i], bins),
                      ConstBlock(ffaout_mem[i+1].get(), rows_eval[i+1], bins),
                      Block(ffaout_total.get(), total_rows_eval, bins));
                // fprintf(stderr, "    merged seg %zu and %zu OK\n", i, i+1);
                // fflush(stderr);
                }

            // Dump the merged transform for the same iteration as the segments.
            // if (dump_this_iter)
            //     {
            //     char path[256];
            //     snprintf(path, sizeof(path),
            //         "debug_ffaout_total_r%zu_c%zu.bin", total_rows_eval, bins);
            //     dump_block_bin(path, ffaout_total.get(), total_rows_eval, bins);
            //     }

            // Noise level of the merged profile. Each phase bin of the merged
            // transform is the sum of the real (non-gap) samples that fall in
            // it; the zero-padded gaps contribute no noise. So the number of
            // samples summed per bin is the sum of the per-segment row counts,
            // NOT total_rows (which would wrongly count the gaps).
            size_t merged_rows = 0;
            for (size_t i = 0; i < num_data; ++i)
                merged_rows += rows[i];
            const float stdnoise_total =
                sqrt(merged_rows * downsampled_variance(total_size, f));

            // Evaluate S/N of the merged profiles for every trial width.
            auto block = ConstBlock(ffaout_total.get(), total_rows_eval, bins);
            // fprintf(stderr, "  [ds %zu bins %zu] snr2: rows=%zu bins=%zu num_widths=%zu stdnoise_total=%f -> writes %zu floats\n",
                // ids, bins, total_rows_eval, bins, num_widths, stdnoise_total, total_rows_eval * num_widths);
            // fflush(stderr);
            snr2(block, widths, num_widths, stdnoise_total, snr);
            // fprintf(stderr, "  [ds %zu bins %zu] snr2 OK\n", ids, bins);
            // fflush(stderr);

            // Record the trial period and number of phase bins for each of the
            // evaluated rows. The period spacing is set by total_rows, the full
            // height of the merged transform (matching periodogram()).
            for (size_t s = 0; s < total_rows_eval; ++s)
                {
                periods[s] = tau * bins * bins / (bins - s / (total_rows - 1.0));
                foldbins[s] = bins;
                }
            // fprintf(stderr, "  [ds %zu bins %zu] wrote %zu periods/foldbins\n",
                // ids, bins, total_rows_eval);
            // fflush(stderr);

            // Advance the output pointers past the trials just written.
            snr += total_rows_eval * num_widths;
            periods += total_rows_eval;
            foldbins += total_rows_eval;
            // fprintf(stderr, "  [ds %zu bins %zu] advanced output pointers (snr+=%zu, periods/foldbins+=%zu)\n",
                // ids, bins, total_rows_eval * num_widths, total_rows_eval);
            // fflush(stderr);
            }
        }
    }


} // namespace riptide

#endif // PERIODOGRAM_HPP
