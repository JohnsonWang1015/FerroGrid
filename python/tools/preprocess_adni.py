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
import tempfile
import zipfile
import collections
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

IMAGE_ID_RE = re.compile(r"/I(\d+)/")
#: ADNI/<PTID>/<series>/<YYYY-MM-DD_HH_MM_SS.0>/I<id>/...
PTID_DATE_RE = re.compile(r"^ADNI/([^/]+)/[^/]+/(\d{4}-\d{2}-\d{2})_[^/]*/I(\d+)/")


def log(msg: str) -> None:
    print(msg, flush=True)


#: ADNI ships two archive flavours and both appear in the same collection.
DICOM_SUFFIXES = (".dcm",)
NIFTI_SUFFIXES = (".nii", ".nii.gz")


def group_by_scan(zf: zipfile.ZipFile) -> tuple[dict[int, list[str]], str]:
    """Map image_id -> its members, and report which flavour the archive is.

    A scan is one `I<id>` directory either way. Raw archives hold a DICOM
    series per scan; the "Complete" collections hold a single preprocessed
    NIfTI, already gradwarp/B1/N3-corrected, which is both better input and
    two orders of magnitude fewer files to read.
    """
    scans: dict[int, list[str]] = defaultdict(list)
    kinds: set[str] = set()
    for name in zf.namelist():
        lower = name.lower()
        if lower.endswith(NIFTI_SUFFIXES):
            kind = "nifti"
        elif lower.endswith(DICOM_SUFFIXES):
            kind = "dicom"
        else:
            continue
        m = IMAGE_ID_RE.search(name)
        if m:
            scans[int(m.group(1))].append(name)
            kinds.add(kind)

    if len(kinds) > 1:
        raise SystemExit(f"archive mixes {sorted(kinds)}; split it and run once per flavour")
    return scans, (kinds.pop() if kinds else "empty")


def read_nifti(zf: zipfile.ZipFile, member: str):
    """Read one NIfTI volume out of the archive.

    Reorients to canonical (RAS) first: ADNI scans arrive in assorted
    orientations, and stacking them without reorienting trains the model on
    whichever way each scanner happened to store its axes.
    """
    import nibabel as nib

    data = zf.read(member)
    suffix = ".nii.gz" if member.lower().endswith(".nii.gz") else ".nii"
    # nibabel needs a real path for gzip members; a temp file is simpler and
    # cheaper than reimplementing its file-map handling.
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        img = nib.as_closest_canonical(nib.load(tmp.name))
        vol = np.asarray(img.dataobj, dtype=np.float32)
        zooms = img.header.get_zooms()[:3]

    if vol.ndim == 4:  # occasional singleton time axis
        vol = vol[..., 0]
    if vol.ndim != 3:
        return None
    return vol, tuple(float(z) for z in zooms)


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


# One open ZipFile per worker process, not per scan. Opening an archive reads
# its central directory, which for the 46.8 GB ADNI set means parsing entries
# for ~800k files -- doing that once per scan would cost far more than the
# conversion itself.
_ZF = None


def _init_worker(zip_path: str) -> None:
    global _ZF
    _ZF = zipfile.ZipFile(zip_path)


def _worker(job):
    """Convert one scan, reusing this process's open archive.

    A ZipFile cannot be shared across processes, and the work is a mix of I/O
    (reading members) and CPU (resampling), so processes beat threads.
    """
    zip_path, members, image_id, out_path, shape, kind = job
    try:
        global _ZF
        if _ZF is None:  # standalone call, e.g. from a test
            _ZF = zipfile.ZipFile(zip_path)
        result = read_nifti(_ZF, members[0]) if kind == "nifti" else read_volume(_ZF, members)
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
    cols = ["label", "split_random", "PTID"]
    by_image_id = cohort.set_index("image_id")[cols].to_dict("index")
    # Preprocessed ADNI collections carry their own IDA image IDs, distinct
    # from the raw series they derive from, so image_id alone will not join a
    # "Complete" archive to the cohort. Subject plus scan date does.
    by_ptid_date = {
        (r.PTID, str(r.scan_date)[:10]): {
            "label": r.label, "split_random": r.split_random, "PTID": r.PTID
        }
        for r in cohort.itertuples()
    }
    log(f"cohort: {len(cohort)} scans")

    with zipfile.ZipFile(args.zip) as zf:
        scans, kind = group_by_scan(zf)
        if kind == "empty":
            raise SystemExit("no .dcm or .nii members found under an I<id> directory")

        # Match on image_id where it works, else on subject + scan date.
        meta_for: dict[int, dict] = {}
        matched_by = collections.Counter()
        for image_id, members in scans.items():
            if image_id in by_image_id:
                meta_for[image_id] = by_image_id[image_id]
                matched_by["image_id"] += 1
                continue
            m = PTID_DATE_RE.match(members[0])
            if m and (key := (m.group(1), m.group(2))) in by_ptid_date:
                meta_for[image_id] = by_ptid_date[key]
                matched_by["ptid+date"] += 1

        usable = {i: m for i, m in scans.items() if i in meta_for}
        log(f"zip: {len(scans)} {kind} scans, {len(usable)} matched to the cohort "
            f"({dict(matched_by)})")
        if not usable:
            raise SystemExit(
                "nothing matched. Check that --cohort covers the subjects in this archive."
            )

        rows, jobs = [], []
        for image_id, members in sorted(usable.items()):
            dest = args.out / f"{image_id}.npy"
            meta = meta_for[image_id]
            row = (image_id, dest.name, meta["label"], meta["split_random"], meta["PTID"])
            if dest.exists() and not args.overwrite:
                rows.append(row)
                continue
            jobs.append(((str(args.zip), members, image_id, str(dest), shape, kind), row))
            if args.limit and len(jobs) >= args.limit:
                break

    log(f"converting {len(jobs)} scans with {args.workers} workers "
        f"({len(rows)} already present)")

    done = failed = 0
    t0 = time.perf_counter()
    if jobs:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(args.zip),),
        ) as pool:
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
