"""Download trained cloud artifacts from the remote GPU host.

Uses a remote tarball + single-file SFTP transfer because this host's SFTP
implementation can stat directories but fails to list them reliably.

Usage:
  uv run python scripts/download_cloud.py
  uv run python scripts/download_cloud.py --remote /data/aigc/models --local models
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.upload_cloud import get_client


def _normalize_remote_arg(remote: str) -> str:
    """Undo Git Bash/MSYS path mangling for remote POSIX paths.

    Examples:
    - /data/aigc/models -> /data/aigc/models
    - C:/Program Files/Git/data/aigc/models -> /data/aigc/models
    """
    remote = remote.replace("\\", "/")
    m = re.match(r"^[A-Za-z]:/Program Files/Git/(.*)$", remote)
    if m:
        return "/" + m.group(1).lstrip("/")
    return remote


def main() -> int:
    parser = argparse.ArgumentParser(description="Download cloud artifacts via remote tarball")
    parser.add_argument("--remote", default="/data/aigc/models", help="Remote directory to archive and download")
    parser.add_argument("--local", default="models", help="Local destination path")
    args = parser.parse_args()

    remote_dir = _normalize_remote_arg(args.remote).rstrip("/")
    remote_path = PurePosixPath(remote_dir)
    remote_parent = str(remote_path.parent)
    remote_name = remote_path.name
    remote_archive = f"/data/{remote_name}.tar.gz"
    local_root = Path(args.local)

    client = get_client()
    sftp = client.open_sftp()
    tmp_dir = Path(tempfile.mkdtemp(prefix="aigc_cloud_"))
    local_archive = tmp_dir / f"{remote_name}.tar.gz"

    try:
        stdin, stdout, stderr = client.exec_command(
            f'tar -czf "{remote_archive}" -C "{remote_parent}" "{remote_name}"',
            timeout=300,
        )
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            err = stderr.read().decode("utf-8", errors="replace")
            print(err, file=sys.stderr)
            return exit_code

        sftp.get(remote_archive, str(local_archive))

        local_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(local_archive, "r:gz") as tar:
            tar.extractall(path=local_root.parent)

        print(f"Downloaded {remote_dir} -> {local_root}")
        return 0
    finally:
        try:
            client.exec_command(f'rm -f "{remote_archive}"', timeout=60)
        except Exception:
            pass
        sftp.close()
        client.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
