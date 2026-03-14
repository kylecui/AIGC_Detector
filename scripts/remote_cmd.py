"""Execute commands on remote cloud GPU via SSH."""

import sys

import paramiko

HOST = "qmvgypgt2ehhxauesnow.deepln.com"
PORT = 47939
USER = "root"
PASS = "l1sTMYxt1pKeBtMuCjX4dReBMVW7TpVq"


def run(cmd: str, timeout: int = 300) -> tuple[int, str, str]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    return exit_code, out, err


if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "echo hello"
    code, out, err = run(cmd)
    if out:
        print(out, end="")
    if err:
        print(err, end="", file=sys.stderr)
    sys.exit(code)
