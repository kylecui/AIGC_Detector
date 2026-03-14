"""Upload training files to cloud GPU and set up environment."""

import os
import sys
from pathlib import Path

import paramiko

HOST = "qmvgypgt2ehhxauesnow.deepln.com"
PORT = 47939
USER = "root"
PASS = "l1sTMYxt1pKeBtMuCjX4dReBMVW7TpVq"
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


def main():
    local_base = Path(__file__).parent.parent
    dataset_dir = local_base / "dataset" / "processed"

    # Files to upload
    files = {
        # Training script
        str(local_base / "scripts" / "train_cloud.py"): f"{REMOTE_BASE}/train_cloud.py",
        # Dataset files
        str(dataset_dir / "train.jsonl"): f"{REMOTE_BASE}/dataset/train.jsonl",
        str(dataset_dir / "val.jsonl"): f"{REMOTE_BASE}/dataset/val.jsonl",
        str(dataset_dir / "test.jsonl"): f"{REMOTE_BASE}/dataset/test.jsonl",
    }

    # Check local files exist
    for local_path in files:
        if not os.path.exists(local_path):
            print(f"ERROR: {local_path} not found!")
            sys.exit(1)

    print("=== Connecting to cloud GPU ===")
    client = get_client()
    sftp = client.open_sftp()

    print("\n=== Creating directories ===")
    run_cmd(client, f"mkdir -p {REMOTE_BASE}/dataset {REMOTE_BASE}/models")

    print("\n=== Uploading files ===")
    for local_path, remote_path in files.items():
        upload_file(sftp, local_path, remote_path)

    sftp.close()

    print("\n=== Verifying uploads ===")
    run_cmd(client, f"ls -lh {REMOTE_BASE}/")
    run_cmd(client, f"ls -lh {REMOTE_BASE}/dataset/")
    run_cmd(client, f"wc -l {REMOTE_BASE}/dataset/*.jsonl")

    client.close()
    print("\n=== Upload complete ===")


if __name__ == "__main__":
    main()
