#!/usr/bin/env python3
"""Plot the FFA transform buffers dumped by periodogram_gappy().

The C++ debug code writes raw float32 buffers named like
    debug_ffaout_seg0_r123_c240.bin     (per-segment transforms)
    debug_ffaout_total_r456_c240.bin    (merged transform)
where r<rows> and c<cols> encode the 2D shape. This script loads each one and
shows it as a 2D intensity map (rows = trial-period shift, cols = phase bin).

Usage:
    python plot_ffa_dumps.py [directory]   # default: current directory
"""
import re
import sys
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

SHAPE_RE = re.compile(r"_r(\d+)_c(\d+)\.bin$")


def load(path):
    m = SHAPE_RE.search(path)
    if not m:
        raise ValueError(f"cannot parse shape from filename: {path}")
    rows, cols = int(m.group(1)), int(m.group(2))
    data = np.fromfile(path, dtype=np.float32)
    if data.size != rows * cols:
        raise ValueError(
            f"{path}: file has {data.size} floats, expected {rows*cols} "
            f"({rows} x {cols})"
        )
    return data.reshape(rows, cols)


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    paths = sorted(glob.glob(os.path.join(directory, "debug_ffaout_*.bin")))
    if not paths:
        print(f"no debug_ffaout_*.bin files found in {directory!r}")
        return

    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, path in zip(axes[0], paths):
        block = load(path)
        im = ax.imshow(
            block, aspect="auto", origin="lower", interpolation="nearest",
            cmap="viridis",
        )
        ax.set_title(f"{os.path.basename(path)}\n{block.shape[0]} x {block.shape[1]}")
        ax.set_xlabel("phase bin")
        ax.set_ylabel("trial-period shift (row)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="intensity")

    fig.tight_layout()
    out = os.path.join(directory, "ffa_dumps.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
