#ifndef TRANSFORMS_HPP
#define TRANSFORMS_HPP

#include <cstddef> // size_t
#include <cstring> // memcpy()
#include <cstdio>  // fprintf()
#include <vector>

#include "kernels.hpp"
#include "block.hpp"


namespace riptide {

void merge(ConstBlock thead, ConstBlock ttail, Block out)
    {
    const size_t m = out.rows;
    const size_t p = out.cols;
    const float kh = (thead.rows - 1.0f) / (m - 1.0f);
    const float kt = (ttail.rows - 1.0f) / (m - 1.0f);

    for (size_t s = 0; s < m; ++s)
        {
        const size_t h = kh * s + 0.5f;
        const size_t t = kt * s + 0.5f;
        const size_t b = s - (h + t);
        //print h, t, b
        fused_rollback_add(thead.rowptr(h), ttail.rowptr(t), p, h+b, out.rowptr(s));
        }
    }


void transform(ConstBlock input, Block temp, Block out)
    {
    const size_t m = input.rows;
    const size_t p = input.cols;

    if (m == 2)
        {
        add(input.rowptr(0), input.rowptr(1), p, out.rowptr(0));
        fused_rollback_add(input.rowptr(0), input.rowptr(1), p, 1, out.rowptr(1));
        return;
        }
    else if (m == 1)
        {
        memcpy(out.data, input.data, p * sizeof(float));
        return;
        }

    transform(input.head(), out.head(), temp.head());
    transform(input.tail(), out.tail(), temp.tail());
    merge(temp.head().as_const(), temp.tail().as_const(), out);
    }


Block transform(const float* input, size_t rows, size_t cols, float* temp, float* out)
    {
    transform(
        ConstBlock(input, rows, cols),
        Block(temp, rows, cols),
        Block(out, rows, cols)
        );
    return Block(out, rows, cols);
    }


/*
Gap-aware FFA transform. Identical to transform(), except every sub-block that
falls ENTIRELY inside a gap is copied through untouched instead of being
transformed and merged. Such a sub-block is all zeros (the gaps are zero-padded),
so its FFA transform is itself; skipping the recursion/merge there is a pure
optimisation that leaves the result bit-for-bit identical to transforming the
zero-padded block with transform().

'gap_rows' lists the row indices (in [0, rows), sorted ascending) of the input
block that lie completely inside a gap. It is consumed/rebased in place across
the recursion, so callers should pass a private, mutable copy.

Faithful port of periodogram_py.transform_gappy.
*/
void transform_gappy(ConstBlock input, Block temp, Block out,
                     size_t* gap_rows, size_t num_gap_rows)
    {
    const size_t m = input.rows;
    const size_t p = input.cols;

    // Whole sub-block is inside a gap => all zeros => transform is itself.
    if (num_gap_rows == m)
        {
        memcpy(out.data, input.data, m * p * sizeof(float));
        return;
        }

    if (m == 1)
        {
        memcpy(out.data, input.data, p * sizeof(float));
        return;
        }
    if (m == 2)
        {
        add(input.rowptr(0), input.rowptr(1), p, out.rowptr(0));
        fused_rollback_add(input.rowptr(0), input.rowptr(1), p, 1, out.rowptr(1));
        return;
        }

    const size_t h = input.head_size();

    // gap_rows is sorted ascending: rows < h go to the head, rows >= h go to the
    // tail (rebased by -h). Rebase the tail portion in place before recursing.
    size_t split = 0;
    while (split < num_gap_rows && gap_rows[split] < h)
        ++split;
    for (size_t i = split; i < num_gap_rows; ++i)
        gap_rows[i] -= h;

    transform_gappy(input.head(), out.head(), temp.head(), gap_rows, split);
    transform_gappy(input.tail(), out.tail(), temp.tail(), gap_rows + split, num_gap_rows - split);
    merge(temp.head().as_const(), temp.tail().as_const(), out);
    }


Block transform_gappy(const float* input, size_t rows, size_t cols, float* temp, float* out,
                      const size_t* gap_rows, size_t num_gap_rows)
    {
    // Private, mutable copy: the recursion rebases gap row indices in place.
    std::vector<size_t> gr(gap_rows, gap_rows + num_gap_rows);
    transform_gappy(
        ConstBlock(input, rows, cols),
        Block(temp, rows, cols),
        Block(out, rows, cols),
        gr.data(), gr.size()
        );
    return Block(out, rows, cols);
    }


} // namespace riptide

#endif // TRANSFORMS_HPP
