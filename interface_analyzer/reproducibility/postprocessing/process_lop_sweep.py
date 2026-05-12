import glob
import os
import time
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

from interface_analyzer import Orientation_analysis

DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "dataset"


def format_float_for_filename(x):
    """
    Convert float to a filename-safe string:
    2.0 -> 2_0
    2.5 -> 2_5
    10.0 -> 10_0
    """
    return f"{x:.1f}".replace(".", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process orientation analysis with given a_grid and d.")
    parser.add_argument("--a_grid", type=float, required=True, help="Grid spacing in Angstroms")
    parser.add_argument("--d", type=float, required=True, help="Smoothing radius in Angstroms")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pattern", default="cfg.Al_100_010.*")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel workers")
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


def worker(task):
    """
    Top-level worker function for multiprocessing.
    Must NOT be defined inside main().
    """
    cfg_path, analysis_params = task

    result_dict = Orientation_analysis(
        str(cfg_path),
        **analysis_params
    )

    try:
        frame_id = frame_id_from_path(cfg_path)
    except ValueError:
        frame_id = str(cfg_path)

    return frame_id, result_dict


def main():
    args = parse_args()

    a_grid = args.a_grid
    d = args.d
    max_workers = args.max_workers

    a_grid_str = format_float_for_filename(a_grid)
    d_str = format_float_for_filename(d)

    save_path_pkl = args.output or Path(f"./100_010_cfg_post_orientation_grid_{a_grid_str}_ang_part_d_{d_str}.pkl")

    analysis_params = {
        "lattice_constant": 4.134,
        "miller_x": [1, 0, 0],
        "miller_y": [0, 1, 0],
        "miller_z": [0, 0, 1],
        "a_grid": a_grid,
        "d": d,
        "n": 5,
        "solid_value": 1,
        "liquid_value": 2
    }

    print(f"Starting parallel analysis with {max_workers} workers.")
    print(f"Reading CFG files from: {args.input_dir.resolve()}")
    print(f"Parameters: a_grid={a_grid:.1f} A, d={d:.1f} A")
    print(f"Output file: {save_path_pkl}")

    files = collect_files(
        args.input_dir,
        args.pattern,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        limit=args.limit,
    )
    print(f"Matched files: {len(files)}")

    if not files:
        raise FileNotFoundError(f"No configuration files found in {args.input_dir.resolve()}")

    # Build task list
    tasks = [(f, analysis_params) for f in files]

    results_all = {}
    start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, task): task[0] for task in tasks}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing CFG files"):
            try:
                frame_id, res = fut.result()
                results_all[frame_id] = res
            except Exception as e:
                print(f"Error processing file {futures[fut]}: {e}")

    end = time.time()
    print(f"\nAnalysis complete. Total time: {end - start:.2f} seconds")

    try:
        save_path_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path_pkl, "wb") as f:
            pickle.dump(results_all, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved results to: {save_path_pkl.resolve()}")
    except Exception as e:
        print(f"Error saving pickle file: {e}")


if __name__ == "__main__":
    main()
