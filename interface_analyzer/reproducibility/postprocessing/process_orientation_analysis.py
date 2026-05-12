import glob
import os
import time
import pickle
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

from interface_analyzer import Orientation_analysis

DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "dataset"
DEFAULT_OUTPUT = Path("./100_010_cfg_post_orientation_grid_2_5_d_6.pkl")

MAX_WORKERS = 8

# --- ANALYSIS PARAMETERS ---
# Orientation_analysis uses a_grid to define resolution instead of binsx/binsz
ANALYSIS_PARAMS = {
    "lattice_constant": 4.134,
    "miller_x":[1, 0, 0],
    "miller_y":[0, 1, 0],
    "miller_z":[0, 0, 1],
    "a_grid": 2.5,        # Grid spacing in Angstroms (determines resolution)
    "d": 6.0,             # Smoothing radius
    "n": 5,               # Window size for Brown interface method
    "solid_value": 1,
    "liquid_value": 2
}

def parse_args():
    parser = argparse.ArgumentParser(description="Post-process CFG files with orientation analysis.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pattern", default="cfg.Al_100_010.*")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
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

def worker(cfg_path):
    """
    Calls Orientation_analysis directly for each file.
    """
    # Call the standalone function with the parameters defined above
    result_dict = Orientation_analysis(
        str(cfg_path),
        **ANALYSIS_PARAMS
    )

    # Extract frame ID from the filename (e.g., cfg.100 -> 100)
    try:
        frame_id = frame_id_from_path(cfg_path)
    except ValueError:
        frame_id = str(cfg_path)

    return frame_id, result_dict

def main():
    args = parse_args()
    files = collect_files(
        args.input_dir,
        args.pattern,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        limit=args.limit,
    )

    print(f"Starting parallel analysis with {args.max_workers} workers.")
    print(f"Reading CFG files from: {args.input_dir.resolve()}")
    print(f"Parameters: a_grid={ANALYSIS_PARAMS['a_grid']} A, d={ANALYSIS_PARAMS['d']} A")
    print(f"Matched files: {len(files)}")

    if not files:
        raise FileNotFoundError(f"No configuration files found in {args.input_dir.resolve()}")

    results_all = {}
    start = time.time()

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker, f): f for f in files}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing CFG files"):
            try:
                frame_id, res = fut.result()
                results_all[frame_id] = res
            except Exception as e:
                print(f"Error processing file {futures[fut]}: {e}")

    end = time.time()
    print(f"\nAnalysis complete. Total time: {end - start:.2f} seconds")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results_all, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
