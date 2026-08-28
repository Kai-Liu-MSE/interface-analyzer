"""Command-line entry point for compact interface extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from .orientation import parse_miller
from .trajectory import extract_trajectory, select_cfg_files, write_interface_pickle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract compact interfaces from selectable atom-wise descriptors.")
    parser.add_argument("cfg_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("full_y", "y_window", "2d"), default="2d")
    parser.add_argument("--xz-grid", type=float, default=2.5)
    parser.add_argument("--y-grid", type=float, default=2.5)
    parser.add_argument("--y-width", type=float)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--coarse-graining", choices=("kernel", "bin"), default="kernel")
    parser.add_argument("--window-n", type=int, default=5)
    parser.add_argument("--lattice-constant", type=float, default=4.05)
    parser.add_argument("--descriptor", choices=("orientation_phi", "ptm_rmsd", "q4", "q6", "qbar4", "qbar6"), default="orientation_phi")
    parser.add_argument("--bond-order-cutoff", type=float, default=3.82)
    parser.add_argument("--ptm-rmsd-cutoff", type=float, default=0.15)
    parser.add_argument("--miller-x", type=parse_miller, default=(0, 0, 1))
    parser.add_argument("--miller-y", type=parse_miller, default=(1, -1, 0))
    parser.add_argument("--miller-z", type=parse_miller, default=(1, 1, 0))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--save-field", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = select_cfg_files(args.cfg_dir, start_step=args.start_step, end_step=args.end_step, stride=args.stride)
    results = extract_trajectory(paths, workers=args.workers, mode=args.mode, xz_grid=args.xz_grid, y_grid=args.y_grid, y_width=args.y_width, radius=args.radius, n=args.window_n, coarse_graining=args.coarse_graining, lattice_constant=args.lattice_constant, descriptor=args.descriptor, bond_order_cutoff=args.bond_order_cutoff, ptm_rmsd_cutoff=args.ptm_rmsd_cutoff, miller_x=args.miller_x, miller_y=args.miller_y, miller_z=args.miller_z, save_field=args.save_field)
    output = write_interface_pickle(results, args.output)
    print(f"Wrote {len(results)} frames to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
