#!/usr/bin/env python3
"""Turn ADNI T1 DICOM archives into compact volumes a dataloader can stream.

Reads the `.dcm` members **straight out of the zip**. The ADNI T1 archive
expands to ~133 GB across 800k files; the volumes it yields at 128^3 float16
are ~4 MB each. Extracting first would cost 133 GB of disk and a long tar of
small-file I/O to produce data that is then thrown away.

Output is a directory of `<image_id>.npy` plus `manifest.csv` carrying the
label and the cohort's own train/val/test split, which is what
`train_mri_3d.py --data-root` consumes.

    uv run --with pandas --with pydicom --with numpy --with scipy python \
        python/tools/preprocess_adni.py \
            --zip  /mnt/esl-E/ADNI/imaging/mri_t1/FedUQ_T1_MRI.zip \
            --cohort /mnt/esl-E/ADNI/derived/cohort_scans.csv \
            --out  /mnt/esl-E/ADNI/preprocessed/t1_128 \
            --shape 128 128 128
"""

from __future__ import annotations

import argparse
import io
import os
import re
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

IMAGE_ID_RE = re.compile(r"/I(\d+)/")


def log(msg: str) -> None:
    print(msg, flush=True)


def group_by_scan(zf: zipfile.ZipFile) -> dict[int, list[str]]:
    """Map image_id -> its DICOM members. One scan is one `I<id>` directory."""
    scans: dict[int, list[str]] = defaultdict(list)
    for name in zf.namelist():
        if not name.lower().endswith(".dcm"):
            continue
        m = IMAGE_ID_RE.search(name)
        if m:
            scans[int(m.group(1))].append(name)
    return scans


def read_volume(zf: zipfile.ZipFile, members: list[str]):
    """Assemble one 3D volume from a DICOM series.

    Slices are ordered by their position along the slice normal rather than by
    filename or InstanceNumber: ADNI mixes conventions across sites and eras,
    and a wrongly ordered stack looks plausible while being anatomically
    scrambled. Returns (volume, spacing_zyx) or None if the series is unusable.
    """
    import pydicom

    slices = []
    for name in members:
        try:
            with zf.open(name) as fh:
                ds = pydicom.dcmread(io.BytesIO(fh.read()), force=True)
            if not hasattr(ds, "pixel_array"):
                continue
            slices.append(ds)
        except Exception:
            continue

    if len(slices) < 16:
        return None

    ref = slices[0]
    try:
        orient = np.asarray(ref.ImageOrientationPatient, dtype=float)
        normal = np.cross(orient[:3], orient[3:])
        slices.sort(key=lambda s: float(np.dot(np.asarray(s.ImagePositionPatient, float), normal)))
    except Exception:
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))

    try:
        vol = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    except Exception:
        return None

    # Rescale to stored units where the header says to.
    slope = float(getattr(ref, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ref, "RescaleIntercept", 0) or 0)
    vol = vol * slope + intercept

    py, px = (float(v) for v in getattr(ref, "PixelSpacing", (1.0, 1.0)))
    pz = float(getattr(ref, "SliceThickness", 1.0) or 1.0)
    # Prefer the real gap between the first two slices; SliceThickness ignores
    # any inter-slice gap and would distort the aspect ratio.
    try:
        p0 = np.asarray(slices[0].ImagePositionPatient, float)
        p1 = np.asarray(slices[1].ImagePositionPatient, float)
        gap = float(np.linalg.norm(p1 - p0))
        if gap > 0:
            pz = gap
    except Exception:
        pass

    return vol, (pz, py, px)


def resize_to(vol: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    from scipy.ndimage import zoom

    factors = [t / s for t, s in zip(shape, vol.shape)]
    out = zoom(vol, factors, order=1)
    # zoom can be off by a voxel; pad or crop to land exactly on `shape`.
    fixed = np.zeros(shape, dtype=np.float32)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(shape, out.shape))
    fixed[sl] = out[sl]
    return fixed


def normalise(vol: np.ndarray) -> np.ndarray:
    """Clip outliers, then z-score over foreground voxels only.

    Background is most of an MRI volume; including it in the statistics drags
    the mean down and makes scans from different scanners less comparable,
    which is exactly the variation ADNI is full of.
    """
    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol = np.clip(vol, lo, hi)
    fg = vol > np.percentile(vol, 20)
    mu = vol[fg].mean() if fg.any() else vol.mean()
    sd = vol[fg].std() if fg.any() else vol.std()
    return (vol - mu) / (sd + 1e-6)


def _worker(job):
    """Convert one scan. Runs in its own process with its own zip handle.

    A ZipFile cannot be shared across processes, and this workload is a mix of
    I/O (reading members) and CPU (resampling), so processes beat threads.
    """
    zip_path, members, image_id, out_path, shape = job
    try:
        with zipfile.ZipFile(zip_path) as zf:
            result = read_volume(zf, members)
        if result is None:
            return image_id, False, "unreadable series"
        vol, spacing = result
        vol = normalise(resize_to(vol, shape))
        np.save(out_path, vol.astype(np.float16))
        return image_id, True, spacing
    except Exception as e:  # noqa: BLE001 - one bad scan must not stop the run
        return image_id, False, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path, help="ADNI T1 DICOM zip")
    ap.add_argument("--cohort", required=True, type=Path, help="cohort_scans.csv")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shape", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--limit", type=int, default=0, help="stop after N scans (for a trial run)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)),
                    help="parallel conversion processes")
    args = ap.parse_args()

    import pandas as pd

    shape = tuple(args.shape)
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort, low_memory=False)
    labels = cohort.set_index("image_id")[["label", "split_random", "PTID"]].to_dict("index")
    log(f"cohort: {len(cohort)} scans")

    with zipfile.ZipFile(args.zip) as zf:
        scans = group_by_scan(zf)
        usable = {i: m for i, m in scans.items() if i in labels}
        log(f"zip: {len(scans)} scans, {len(usable)} of them in the cohort")

        rows, jobs = [], []
        for image_id, members in sorted(usable.items()):
            dest = args.out / f"{image_id}.npy"
            meta = labels[image_id]
            row = (image_id, dest.name, meta["label"], meta["split_random"], meta["PTID"])
            if dest.exists() and not args.overwrite:
                rows.append(row)
                continue
            jobs.append(((str(args.zip), members, image_id, str(dest), shape), row))
            if args.limit and len(jobs) >= args.limit:
                break

    log(f"converting {len(jobs)} scans with {args.workers} workers "
        f"({len(rows)} already present)")

    done = failed = 0
    t0 = time.perf_counter()
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for (image_id, ok, info), (_, row) in zip(
                pool.map(_worker, [j for j, _ in jobs], chunksize=1), jobs
            ):
                if ok:
                    rows.append(row)
                    done += 1
                else:
                    failed += 1
                    log(f"  skip {image_id}: {info}")
                if (done + failed) % 10 == 0:
                    rate = (done + failed) / (time.perf_counter() - t0)
                    eta = (len(jobs) - done - failed) / max(rate, 1e-6)
                    log(f"  {done + failed}/{len(jobs)}  {rate:.2f} scans/s  eta {eta/60:.1f} min")

    man = pd.DataFrame(rows, columns=["image_id", "file", "label", "split", "ptid"])
    man.to_csv(args.out / "manifest.csv", index=False)

    log("=" * 60)
    log(f"wrote {len(man)} volumes to {args.out}  ({failed} unreadable)")
    log(f"shape {shape} float16 -> {np.prod(shape) * 2 / 1e6:.1f} MB each")
    if len(man):
        log("labels: " + man.label.value_counts().to_dict().__str__())
        log("splits: " + man.split.value_counts().to_dict().__str__())
    log("=" * 60)
    return 0 if len(man) else 1


if __name__ == "__main__":
    raise SystemExit(main())
