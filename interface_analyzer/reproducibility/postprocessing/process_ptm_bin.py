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

# =========================
# ARGUMENT PARSING
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=float, required=True)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pattern", default="cfg.*")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--first_frame", type=int, default=None)
    parser.add_argument("--last_frame", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N matching files.")
    return parser.parse_args()


# =========================
# FORMAT FLOAT FOR FILENAME
# =========================
def fmt(x):
    return f"{x:.1f}".replace(".", "_")


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


# =========================
# MAIN
# =========================
def worker(task):
    cfg_path, grid_size = task

    GRID_SIZE_X = grid_size
    GRID_SIZE_Z = grid_size

    RMSD_MAX = 0.15
    N_WINDOW = None

    node = import_file(str(cfg_path))
    data = node.compute()
    cell = data.cell[:]

    lx = float(cell[0, 0])
    lz = float(cell[2, 2])

    binsx = max(1, int(round(lx / GRID_SIZE_X)))
    binsz = max(1, int(round(lz / GRID_SIZE_Z)))

    modifier_kwargs = dict(binsx=binsx, binsz=binsz, rmsd_max=RMSD_MAX)
    if N_WINDOW is not None:
        modifier_kwargs["n"] = N_WINDOW

    ptm_modifier = PTMModifier(**modifier_kwargs)
    result_dict = analyze_by_custom_modifier(str(cfg_path), ptm_modifier)

    # metadata
    result_dict["grid_size_x"] = GRID_SIZE_X
    result_dict["grid_size_z"] = GRID_SIZE_Z
    result_dict["binsx_auto"] = binsx
    result_dict["binsz_auto"] = binsz
    result_dict["lx"] = lx
    result_dict["lz"] = lz

    try:
        frame_id = frame_id_from_path(cfg_path)
    except ValueError:
        frame_id = str(cfg_path)

    return frame_id, result_dict


def main():
    args = parse_args()

    GRID_SIZE = args.grid_size
    MAX_WORKERS = args.max_workers

    SAVE_PATH_PKL = args.output or Path(f"./cfg_post_ptm_gridsize_{fmt(GRID_SIZE)}.pkl")

    print(f"Grid size = {GRID_SIZE} A")
    print(f"Reading configuration files from: {args.input_dir.resolve()}")
    print(f"Saving to: {SAVE_PATH_PKL}")

    files = collect_files(
        args.input_dir,
        args.pattern,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        limit=args.limit,
    )
    print(f"Matched files: {len(files)}")

    if not files:
        raise FileNotFoundError(f"No CFG files found in {args.input_dir.resolve()}.")

    tasks = [(f, GRID_SIZE) for f in files]

    results_all = {}
    failed = {}

    start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, t): t[0] for t in tasks}

        for fut in tqdm(as_completed(futures), total=len(futures)):
            src = futures[fut]
            try:
                frame_id, res = fut.result()
                results_all[frame_id] = res
            except Exception as e:
                failed[src] = str(e)

    end = time.time()

    print(f"Done in {end-start:.2f} s")

    SAVE_PATH_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH_PKL, "wb") as f:
        pickle.dump(results_all, f)

    if failed:
        with open(SAVE_PATH_PKL.with_suffix(".failed.txt"), "w") as f:
            for k, v in failed.items():
                f.write(f"{k}\t{v}\n")


if __name__ == "__main__":
    main()
