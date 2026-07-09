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
OLD implementation of the gappy periodogram, kept for comparison against the
faster segment-wise periodogram_gappy() below.

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

The gappy series is treated as ONE zero-padded contiguous series
(seg0 | zeros(gap0) | seg1 | zeros(gap1) | ...), downsampled and FFA-transformed
as a single unit. This is exactly the zero-pad ground truth of the gappy
periodogram: the zeros hold each segment's samples at their correct phase while
contributing nothing to the folded sums. The only optimisation over a plain
periodogram() on the padded series is that transform_gappy() skips FFA sub-blocks
that fall entirely inside a gap (they are all-zero, so their transform is
themselves). The result is therefore bit-for-bit identical to periodogram() run
on the zero-padded series.
*/
void periodogram_gappy_old(
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

    // Build the zero-padded contiguous series: each segment is copied to its
    // offset, the inter-segment gaps are left as zeros.
    std::unique_ptr<float[]> data_padded(new float[total_size]);
    std::fill(data_padded.get(), data_padded.get() + total_size, 0.0f);
    {
    size_t offset = 0;
    for (size_t i = 0; i < num_data; ++i)
        {
        memcpy(data_padded.get() + offset, data[i], sizes[i] * sizeof(float));
        offset += sizes[i];
        if (i + 1 < num_data)
            offset += gaps[i];
        }
    }
    const float* full = data_padded.get();

    // Initial downsampling factor
    // We want: ds_ini * tsamp * bmin = period_min
    double ds_ini = period_min / (tsamp * bins_min);

    // Geometric growth factor for the downsampling factor
    double ds_geo = (bins_max + 1.0) / bins_min;

    // Number of required downsampling cycles
    size_t num_downsamplings = ceil(log(period_max / period_min) / log(ds_geo));

    // Work buffers, each large enough to hold the whole padded series at the
    // finest (initial) downsampling factor, i.e. its largest downsampled size.
    const size_t bufsize = downsampled_size(total_size, ds_ini);
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
        const size_t n = downsampled_size(total_size, f); // current number of input samples

        // downsample() requires f > 1, but we still allow searching the data at
        // their original resolution.
        if (f == 1)
            input = full;
        else
            {
            downsample(full, total_size, f, input_mem.get());
            input = input_mem.get();
            }

        // Locate every gap in the DOWNSAMPLED array. The full zero-padded series
        // is downsampled as a single unit, so a raw sample position P maps to the
        // output index downsampled_size(P, f). Compute the gap edges from the
        // cumulative raw positions (floor(a/f) + floor(b/f) != floor((a+b)/f), so
        // only the whole-array mapping is consistent with input = downsample()).
        const size_t num_gaps = (num_data > 0) ? num_data - 1 : 0;
        std::vector<size_t> gap_start_ds(num_gaps);
        std::vector<size_t> gap_end_ds(num_gaps);
        {
        size_t raw = 0; // cumulative raw sample position
        for (size_t i = 0; i < num_gaps; ++i)
            {
            raw += sizes[i]; // end of segment i == start of gap i
            gap_start_ds[i] = downsampled_size(raw, f);
            gap_end_ds[i]   = downsampled_size(raw + gaps[i], f);
            raw += gaps[i];  // advance to start of segment i + 1
            }
        }

        // Min and max number of bins with which to FFA transform in order to
        // cover all trial periods between period_min and period_max.
        // NOTE: bstop is INclusive.
        // Also, we MUST enforce bstop <= n, to avoid doing an FFA transform with 0 rows
        const size_t bstart = bins_min;
        const size_t bstop = std::min({ bins_max, n, size_t(period_max_samples) });

        /* FFA transform loop */
        for (size_t bins = bstart; bins <= bstop; ++bins)
            {
            const size_t rows = n / bins;
            const float stdnoise = sqrt(rows * downsampled_variance(total_size, f));
            const double period_ceil = std::min(period_max_samples, bins + 1.0);
            const size_t rows_eval = std::min(rows, ceilshift(rows, bins, period_ceil));

            // Rows of the (rows x bins) reshaped block that fall ENTIRELY inside a
            // gap: for gap i, rows strictly between gap_start_ds[i]/bins and
            // gap_end_ds[i]/bins. These sub-blocks are all zeros and are skipped
            // by transform_gappy. Built in ascending order (segments are ordered).
            std::vector<size_t> gap_rows;
            for (size_t i = 0; i < num_gaps; ++i)
                {
                const size_t gr_start = gap_start_ds[i] / bins;
                const size_t gr_end = gap_end_ds[i] / bins;
                for (size_t r = gr_start + 1; r < gr_end && r < rows; ++r)
                    gap_rows.push_back(r);
                }

            transform_gappy(input, rows, bins, ffabuf, ffaout,
                            gap_rows.data(), gap_rows.size());

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
Gappy periodogram computed segment-wise, without ever materialising the
zero-padded series. Takes the same arguments and produces the same output
layout as periodogram_gappy_old().

Pipeline per downsampling factor f:
  1. downsample_gappy(): each segment is downsampled on the global window grid
     into its own small "piece" (see downsample.hpp); the huge runs of zeros in
     the gaps are never touched.
  2. For every 'bins' trial the virtual (rows x bins) downsampled block is cut
     at the gaps: segment i owns rows [r0, r1) with r0 = 0 or the last row of
     gap i-1, and r1 = the first row of gap i (inclusive of the partial
     boundary rows) or 'rows'. Each block is assembled additively from the
     pieces (extract_rows) and FFA-transformed on its own.
  3. The per-segment transforms are chain-merged across the gaps: merging into
     out_rows = gr_end[i] + tail_rows spans the pure-gap rows virtually, so
     the interior gap rows are never transformed or even allocated.

The result is NOT bitwise identical to periodogram_gappy_old(): the FFA shift
of each input row is quantised through the merge tree, and the segment-wise
tree differs from the full binary tree over the padded block. Both schedules
stay within +/- 1 phase bin of the ideal shift round(s * r / (rows - 1)), so
the two outputs agree at the level of the FFA's inherent quantisation error
(peaks coincide; off-peak S/N differs by a few tenths at most).

When a gap spans fewer than ~2 rows, the last gap row is clamped up to the
first (the segments are treated as contiguous) so no rows are double-counted.

Faithful port of periodogram_py.periodogram_gappy.
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
    // the missing samples that fall in the gaps.
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

    // Work buffers. A segment block can reach rows + 1 rows when a gap starts
    // inside the truncated tail of the (rows x bins) reshape (then
    // gr_start == rows + 1), hence the + bins_max margin over the largest
    // downsampled size.
    const size_t bufsize = downsampled_size(total_size, ds_ini) + bins_max;
    std::unique_ptr<float[]> block_mem(new float[bufsize]);   // assembled input block
    std::unique_ptr<float[]> ffatemp_mem(new float[bufsize]); // transform scratch
    std::unique_ptr<float[]> segffa_mem(new float[bufsize]);  // current segment transform
    std::unique_ptr<float[]> mergea_mem(new float[bufsize]);  // merge accumulator (ping)
    std::unique_ptr<float[]> mergeb_mem(new float[bufsize]);  // merge accumulator (pong)

    // Per-segment downsampled pieces, recomputed once per downsampling factor
    std::vector<std::vector<float>> ds_segs;
    std::vector<size_t> ds_starts;

    const size_t num_gaps = num_data - 1;
    std::vector<size_t> gap_start_ds(num_gaps);
    std::vector<size_t> gap_end_ds(num_gaps);
    std::vector<size_t> gr_start(num_gaps); // first row of each gap (exclusive of boundary row)
    std::vector<size_t> gr_end(num_gaps);   // last row of each gap

    /* Downsampling loop */
    for (size_t ids = 0; ids < num_downsamplings; ++ids)
        {
        const double f = ds_ini * pow(ds_geo, ids); // current downsampling factor
        const double tau = f * tsamp; // current sampling time
        const double period_max_samples = period_max / tau;
        const size_t n = downsampled_size(total_size, f); // current number of input samples

        downsample_gappy(data, sizes, gaps, num_data, total_size, f, ds_segs, ds_starts);

        // Locate every gap in the (virtual) DOWNSAMPLED array: a raw sample
        // position P maps to the output index downsampled_size(P, f). Compute
        // the gap edges from the cumulative raw positions
        // (floor(a/f) + floor(b/f) != floor((a+b)/f), so only the whole-array
        // mapping is consistent with the downsampled pieces).
        {
        size_t raw = 0; // cumulative raw sample position
        for (size_t i = 0; i < num_gaps; ++i)
            {
            raw += sizes[i]; // end of segment i == start of gap i
            gap_start_ds[i] = downsampled_size(raw, f);
            gap_end_ds[i]   = downsampled_size(raw + gaps[i], f);
            raw += gaps[i];  // advance to start of segment i + 1
            }
        }

        // Min and max number of bins with which to FFA transform in order to
        // cover all trial periods between period_min and period_max.
        // NOTE: bstop is INclusive.
        // Also, we MUST enforce bstop <= n, to avoid doing an FFA transform with 0 rows
        const size_t bstart = bins_min;
        const size_t bstop = std::min({ bins_max, n, size_t(period_max_samples) });

        /* FFA transform loop */
        for (size_t bins = bstart; bins <= bstop; ++bins)
            {
            const size_t rows = n / bins;
            const float stdnoise = sqrt(rows * downsampled_variance(total_size, f));
            const double period_ceil = std::min(period_max_samples, bins + 1.0);
            const size_t rows_eval = std::min(rows, ceilshift(rows, bins, period_ceil));

            // First row fully inside each gap (gr_start) and last row fully
            // inside it (gr_end); the partial boundary rows belong to the
            // segments. If a gap spans fewer than ~2 rows gr_end < gr_start
            // and the segment blocks would overlap, double-counting the
            // boundary rows in the merge: clamp gr_end up to gr_start so the
            // two segments are treated as contiguous.
            for (size_t i = 0; i < num_gaps; ++i)
                {
                gr_start[i] = gap_start_ds[i] / bins + 1;
                const ptrdiff_t e = ptrdiff_t(gap_end_ds[i] / bins) - 1;
                gr_end[i] = std::max(e, ptrdiff_t(gr_start[i]));
                }

            // Assemble rows [r0, r1) of the virtual (rows x bins) downsampled
            // block, additively from the per-segment pieces -- bitwise
            // identical to slicing a fully materialised downsampled array.
            // Pieces only overlap when a gap is smaller than one window;
            // adding them in segment order reproduces the whole-array
            // accumulation of the straddling window.
            auto extract_rows = [&](size_t r0, size_t r1, float* out)
                {
                const size_t lo = r0 * bins;
                const size_t hi = r1 * bins;
                std::fill(out, out + (hi - lo), 0.0f);
                for (size_t j = 0; j < num_data; ++j)
                    {
                    const size_t s = std::max(lo, ds_starts[j]);
                    const size_t e = std::min(hi, ds_starts[j] + ds_segs[j].size());
                    for (size_t idx = s; idx < e; ++idx)
                        out[idx - lo] += ds_segs[j][idx - ds_starts[j]];
                    }
                };

            // FFA transform segment 0 into the merge accumulator
            float* acc = mergea_mem.get();
            float* other = mergeb_mem.get();
            size_t r1 = (num_data > 1) ? gr_start[0] : rows;
            extract_rows(0, r1, block_mem.get());
            transform(block_mem.get(), r1, bins, ffatemp_mem.get(), acc);
            size_t acc_rows = r1;

            // Chain-merge the remaining segments across the gaps. Merging into
            // out_rows = gr_end[i-1] + tail_rows makes the shift budget span
            // the (never materialised) pure-gap rows.
            for (size_t i = 1; i < num_data; ++i)
                {
                const size_t r0 = gr_end[i - 1];
                r1 = (i + 1 < num_data) ? gr_start[i] : rows;
                periodogram_check_arg(r0 < r1,
                    "degenerate gap geometry: a segment folds to zero rows");
                extract_rows(r0, r1, block_mem.get());
                transform(block_mem.get(), r1 - r0, bins, ffatemp_mem.get(), segffa_mem.get());

                const size_t m = gr_end[i - 1] + (r1 - r0);
                if (m == 1)
                    // merge() divides by m - 1; the python port defines the
                    // shifts as 0 in this case: out row 0 = head 0 + tail 0
                    add(acc, segffa_mem.get(), bins, other);
                else
                    merge(ConstBlock(acc, acc_rows, bins),
                          ConstBlock(segffa_mem.get(), r1 - r0, bins),
                          Block(other, m, bins));
                std::swap(acc, other);
                acc_rows = m;
                }

            auto block = ConstBlock(acc, rows_eval, bins);
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


} // namespace riptide

#endif // PERIODOGRAM_HPP
