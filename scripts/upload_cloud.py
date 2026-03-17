"""Upload training files to cloud GPU and set up environment."""

import argparse
import os
import re
import sys
from pathlib import Path

import paramiko

HOST = "egtmw5gvs4tqw9dlsnow.deepln.com"
PORT = 49570
USER = "root"
PASS = "wH7I6ttJSEMRctlVTIASt13HqxGSe4VU"
REMOTE_BASE = "/data/aigc"


def get_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return client


def run_cmd(client, cmd, timeout=300):
    print(f"  > {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if out.strip():
        print(out.rstrip())
    if err.strip():
        print(err.rstrip(), file=sys.stderr)
    return exit_code, out, err


def upload_file(sftp, local_path, remote_path):
    remote_dir = os.path.dirname(remote_path)
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        # Create dirs recursively
        parts = remote_dir.split("/")
        for i in range(2, len(parts) + 1):
            d = "/".join(parts[:i])
            try:
                sftp.stat(d)
            except FileNotFoundError:
                sftp.mkdir(d)

    local_size = os.path.getsize(local_path)
    print(f"  Uploading {local_path} -> {remote_path} ({local_size / 1024 / 1024:.1f} MB)")
    sftp.put(str(local_path), remote_path)


def normalize_remote_path(path: str) -> str:
    """Normalize remote POSIX paths when invoked from Git Bash on Windows.

    Git Bash may rewrite `/data/...` into `C:/Program Files/Git/data/...` before
    Python receives argv. Convert that form back to the intended POSIX path.
    """
    normalized = path.replace("\\", "/")
    normalized = re.sub(r"^[A-Za-z]:/Program Files/Git", "", normalized)
    if not normalized.startswith("/"):
        normalized = f"/{normalized.lstrip('/')}"
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Upload training script and dataset files to the cloud GPU host")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Local dataset directory containing train.jsonl/val.jsonl/test.jsonl",
    )
    parser.add_argument(
        "--remote-dataset-dir",
        default=f"{REMOTE_BASE}/dataset",
        help="Remote directory where dataset split files should be uploaded",
    )
    parser.add_argument(
        "--remote-script-path",
        default=f"{REMOTE_BASE}/train_cloud.py",
        help="Remote path for the training entrypoint script",
    )
    args = parser.parse_args()

    # Avoid Git Bash / MSYS converting remote POSIX paths like /data/... into
    # local Windows-looking paths such as C:/Program Files/Git/data/...
    os.environ.setdefault("MSYS_NO_PATHCONV", "1")
    os.environ.setdefault("MSYS2_ARG_CONV_EXCL", "*")

    local_base = Path(__file__).parent.parent
    dataset_dir = args.dataset_dir or (local_base / "dataset" / "processed")

    files = {
        str(local_base / "scripts" / "train_cloud.py"): args.remote_script_path,
        str(dataset_dir / "train.jsonl"): f"{args.remote_dataset_dir}/train.jsonl",
        str(dataset_dir / "val.jsonl"): f"{args.remote_dataset_dir}/val.jsonl",
        str(dataset_dir / "test.jsonl"): f"{args.remote_dataset_dir}/test.jsonl",
    }

    # Check local files exist
    for local_path in files:
        if not os.path.exists(local_path):
            print(f"ERROR: {local_path} not found!")
            sys.exit(1)

    print("=== Connecting to cloud GPU ===")
    client = get_client()
    sftp = client.open_sftp()

    remote_dataset_dir = normalize_remote_path(args.remote_dataset_dir)
    remote_script_path = normalize_remote_path(args.remote_script_path)
    remote_script_dir = os.path.dirname(remote_script_path)

    print("\n=== Creating directories ===")
    run_cmd(client, f"mkdir -p {remote_dataset_dir} {remote_script_dir} {REMOTE_BASE}/models")

    files = {
        str(local_base / "scripts" / "train_cloud.py"): remote_script_path,
        str(dataset_dir / "train.jsonl"): f"{remote_dataset_dir}/train.jsonl",
        str(dataset_dir / "val.jsonl"): f"{remote_dataset_dir}/val.jsonl",
        str(dataset_dir / "test.jsonl"): f"{remote_dataset_dir}/test.jsonl",
    }

    print("\n=== Uploading files ===")
    for local_path, remote_path in files.items():
        upload_file(sftp, local_path, remote_path)

    sftp.close()

    print("\n=== Verifying uploads ===")
    run_cmd(client, f'ls -lh "{remote_script_dir}"')
    run_cmd(client, f'ls -lh "{remote_dataset_dir}"')
    run_cmd(client, f'sh -lc "wc -l {remote_dataset_dir}/*.jsonl"')

    client.close()
    print("\n=== Upload complete ===")


if __name__ == "__main__":
    main()
