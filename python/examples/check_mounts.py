#!/usr/bin/env python3
"""Check that a job's data mounts are visible and writable inside the container.

Run this before a long training job -- it costs seconds and catches the two
mistakes that otherwise surface an hour in: a path that was never mounted, and
an output directory the job's uid cannot write to.

    ferro train --nodes 1 --gpus-per-node 1 -f \
        --mount /mnt/adni_data:/mnt/adni_data:ro --mount /mnt/adni_work \
        python/examples/check_mounts.py /mnt/adni_data /mnt/adni_work
"""

import os
import sys


def fstype(path: str) -> str:
    """Filesystem backing `path`, from the container's own mount table.

    Worth showing: "cifs" or "nfs4" here explains ownership surprises that
    look inexplicable otherwise, because both fix uid/gid at mount time on the
    host rather than honouring the on-disk owner.
    """
    best, kind = "", "unknown"
    try:
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt, typ = parts[1], parts[2]
                if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                    best, kind = mnt, typ
    except OSError:
        pass
    return kind


def check(path: str) -> bool:
    if not os.path.isdir(path):
        print(f"MISSING    {path}  -- not mounted into the container")
        return False

    readable = os.access(path, os.R_OK)
    writable = os.access(path, os.W_OK)
    try:
        entries = len(os.listdir(path))
    except OSError as e:
        entries = f"unreadable ({e.strerror})"

    print(f"OK         {path}  fstype={fstype(path)} readable={readable} "
          f"writable={writable} entries={entries}")

    if not writable:
        st = os.stat(path)
        print(f"           owned by uid={st.st_uid} gid={st.st_gid} mode={oct(st.st_mode)[-3:]}; "
              f"job runs as uid={os.getuid()} gid={os.getgid()}")
        print("           fine for a read-only dataset; fix it before using this "
              "as --out-dir")
    return readable


def main() -> int:
    paths = sys.argv[1:] or ["/mnt/adni_data", "/mnt/adni_work"]
    print(f"uid={os.getuid()} gid={os.getgid()} cwd={os.getcwd()}")
    ok = [check(p) for p in paths]
    # Only a missing/unreadable mount is a hard failure; a read-only dataset
    # mount is the normal case.
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
