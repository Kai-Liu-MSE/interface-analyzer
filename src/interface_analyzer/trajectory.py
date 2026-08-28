"""Batch trajectory extraction and portable compressed-pickle output."""

from __future__ import annotations

import gzip
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .orientation import analyze_orientation_interface


def frame_step(path: str | Path) -> int | str:
    """Return the trailing numeric LAMMPS step when present."""
    name = Path(path).name
    match = re.search(r"\.(\d+)(?:\.gz)?$", name)
    return int(match.group(1)) if match else name


def select_cfg_files(
    cfg_dir: str | Path, *, start_step: int | None = None, end_step: int | None = None, stride: int = 1
) -> list[Path]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    paths = sorted(Path(cfg_dir).glob("cfg.*"), key=frame_step)
    selected: list[Path] = []
    for path in paths:
        step = frame_step(path)
        if not isinstance(step, int):
            continue
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue
        if step % stride:
            continue
        selected.append(path)
    return selected


def _worker(payload: tuple[str, dict[str, Any]]) -> tuple[int | str, dict[str, object]]:
    path, options = payload
    return frame_step(path), analyze_orientation_interface(path, **options)


def extract_trajectory(
    cfg_paths: Iterable[str | Path], *, workers: int = 1, **options: Any
) -> dict[int | str, dict[str, object]]:
    """Extract interfaces independently for all configurations.

    Atom-wise LOP is intentionally calculated only once inside each frame task,
    then shared by all requested coarse-graining work for that frame.
    """
    paths = [Path(path) for path in cfg_paths]
    if not paths:
        raise ValueError("No CFG files selected")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    payload = [(str(path), options) for path in paths]
    results: dict[int | str, dict[str, object]] = {}
    if workers == 1:
        for item in tqdm(payload, desc="Extracting interfaces"):
            step, result = _worker(item)
            results[step] = result
        return results
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, item): item[0] for item in payload}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting interfaces"):
            step, result = future.result()
            results[step] = result
    return results


def write_interface_pickle(results: dict[int | str, dict[str, object]], output: str | Path, *, compression_level: int = 6) -> Path:
    """Write the compact interface schema, using gzip when the suffix is ``.gz``."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=compression_level) as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with path.open("wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path
