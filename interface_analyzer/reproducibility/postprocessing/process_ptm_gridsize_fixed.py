import glob
import os
import time
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import warnings
import argparse
from ovito.io import import_file

import ovito._extensions.pyscript
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

from interface_analyzer import PTMModifier, analyze_by_custom_modifier


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "dataset"
DEFAULT_OUTPUT = Path("./cfg_post_ptm_gridsize_1_5.pkl")

# --- GRID/PTM PARAMETERS ---
GRID_SIZE_X = 1.5
GRID_SIZE_Z = GRID_SIZE_X
RMSD_MAX = 0.15
N_WINDOW = None  # None -> use PTMModifier default


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process CFG files with PTM and a fixed grid size.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pattern", default="cfg.*")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--first_frame", type=int, default=None)
    parser.add_argument("--last_frame", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N matching files.")
    return parser.parse_args()


def frame_id_from_path(path):
    return int(Path(path).name.split(".")[-1])


def collect_files(input_dir, pattern, first_frame=None, last_frame=None, limit=None):
    files = sorted(glob.glob(str(input_dir / pattern)), key=frame_id_from_path)
    if first_frame is not None:
        files = [f for f in files if frame_id_from_path(f) >= first_frame]
    if last_frame is not None:
        files = [f for f in files if frame_id_from_path(f) <= last_frame]
    if limit is not None:
        files = files[:limit]
    return files


def _bins_from_cell(cfg_path):
    node = import_file(str(cfg_path))
    data = node.compute()
    cell = data.cell[:]

    lx = float(cell[0, 0])
    lz = float(cell[2, 2])

    binsx = max(1, int(round(lx / GRID_SIZE_X)))
    binsz = max(1, int(round(lz / GRID_SIZE_Z)))
    return binsx, binsz, lx, lz


def worker(cfg_path):
    binsx, binsz, lx, lz = _bins_from_cell(cfg_path)

    modifier_kwargs = dict(binsx=binsx, binsz=binsz, rmsd_max=RMSD_MAX)
    if N_WINDOW is not None:
        modifier_kwargs["n"] = N_WINDOW

    ptm_modifier = PTMModifier(**modifier_kwargs)
    result_dict = analyze_by_custom_modifier(str(cfg_path), ptm_modifier)

    # Save auto-grid metadata for traceability
    result_dict["grid_size_x"] = GRID_SIZE_X
    result_dict["grid_size_z"] = GRID_SIZE_Z
    result_dict["binsx_auto"] = binsx
    result_dict["binsz_auto"] = binsz
    result_dict["lx"] = lx
    result_dict["lz"] = lz

    fname = str(cfg_path)
    try:
        frame_id = frame_id_from_path(fname)
    except ValueError:
        frame_id = fname
    return frame_id, result_dict


def main():
    args = parse_args()

    print(f"Starting parallel analysis with {args.max_workers} workers.")
    print(f"Reading configuration files from: {args.input_dir.resolve()}")
    print(f"Saving results to: {args.output.resolve()}")
    print(f"Target grid size: dx={GRID_SIZE_X}, dz={GRID_SIZE_Z}")

    files = collect_files(
        args.input_dir,
        args.pattern,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        limit=args.limit,
    )

    if not files:
        raise FileNotFoundError(f"No configuration files found in the directory: {args.input_dir.resolve()}")

    print(f"Found {len(files)} files for processing.")

    results_all = {}
    failed = {}
    start = time.time()

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker, f): f for f in files}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing CFG files"):
            src = futures[fut]
            try:
                frame_id, res = fut.result()
                results_all[frame_id] = res
            except Exception as e:
                failed[src] = str(e)

    end = time.time()
    print(f"\nAnalysis complete. Total processing time: {end - start:.2f} seconds")
    print(f"Successful frames: {len(results_all)}")
    print(f"Failed frames: {len(failed)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results_all, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Successfully saved aggregated results to: {args.output.resolve()}")

    if failed:
        failed_log = args.output.with_suffix(".failed.txt")
        with open(failed_log, "w", encoding="utf-8") as f:
            for k, v in failed.items():
                f.write(f"{k}\t{v}\n")
        print(f"Failure details written to: {failed_log.resolve()}")


if __name__ == "__main__":
    main()
