"""
Access controls for evaluation runs.

Two controls keep the resolving change out of the agent's reach:

* ``isolated_checkout`` replaces the agent's working copy with the tree of one
  commit in a fresh single-commit git repository, so later history (including
  the fix) cannot be reached with git from inside the checkout. The full clone
  used to export the tree lives outside the agent's working directory.
* ``install_guard`` adds a Python audit hook. While agent-written code runs
  (inside ``guarded()``), it refuses network connections to hosts other than
  the model API and loopback, refuses to start network or git programs, and
  refuses to open or list files under the user's home directory or the shared
  temporary directories outside the checkout, the Python installation, and a
  private temporary directory of the run; programs started by the agent are
  checked the same way, including their working directory. Harness code
  outside ``guarded()`` is not affected.

Refusals raise PermissionError inside the agent's code and are counted, so a
report can state how many times an agent tried to leave the checkout.

``AGENT_GUARD_MODE=shared-tmp`` selects the guard used for most issue runs of the IEEE Access
resubmission (2026-11-18): the shared temporary
directory is allowed, only absolute and ``..`` path arguments of programs are checked, and the
working directory of programs and ``os.chdir`` are not checked. The default (``strict``) is the
guard of the configurations that consult domain agents and of the issue runs that had read the shared temporary
directory and were repeated.
"""

from __future__ import annotations

import contextvars
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

_ACTIVE = contextvars.ContextVar("agent_code_guard", default=False)
_STATE: dict = {"installed": False, "codebase": None, "allowed": [], "hosts": set(), "blocked": []}

ALLOWED_HOSTS = {"api.anthropic.com", "localhost", "127.0.0.1", "::1"}
BLOCKED_PROGRAMS = {
    "git", "gh", "curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "netcat",
    "telnet", "ftp", "http", "https", "pip", "pip3", "uv", "python", "python3", "open",
}
HOME = Path.home().resolve()
GUARD_MODE = os.environ.get("AGENT_GUARD_MODE", "strict")
if GUARD_MODE not in ("strict", "shared-tmp"):
    raise ValueError(f"AGENT_GUARD_MODE must be strict or shared-tmp, not {GUARD_MODE!r}")
SHARED_TMP = GUARD_MODE == "shared-tmp"
# Shared temporary directories: other processes (and earlier sessions) leave files there.
TEMP_ROOTS = {Path(p).resolve() for p in ("/tmp", "/private/tmp", "/var/tmp", "/var/folders", "/private/var/folders")
              if Path(p).exists()} | {Path(tempfile.gettempdir()).resolve()}
DENIED_ROOTS = {HOME} if SHARED_TMP else {HOME} | TEMP_ROOTS


def _resolve(path, base=None) -> Path | None:
    if isinstance(path, int):
        return None
    try:
        p = Path(os.fsdecode(path) if SHARED_TMP else os.path.expanduser(os.fsdecode(path)))
    except TypeError:
        return None
    if not p.is_absolute():
        p = Path(base or os.getcwd()) / p
    if SHARED_TMP:
        return Path(os.path.normpath(p))
    try:
        return p.resolve()
    except OSError:
        return Path(os.path.normpath(p))


def _path_denied(path, base=None) -> bool:
    p = _resolve(path, base)
    if p is None:
        return False
    for root in _STATE["allowed"]:
        if p == root or root in p.parents:
            return False
    return any(p == root or root in p.parents for root in DENIED_ROOTS)


def _note(kind: str, detail: str) -> None:
    _STATE["blocked"].append((kind, detail[:300]))


def _program_denied(argv, cwd=None) -> str | None:
    if SHARED_TMP:
        cwd = None
    if cwd is not None and _path_denied(cwd):
        return f"cwd={os.fsdecode(cwd)}"
    base = _resolve(cwd) if cwd is not None else None
    if isinstance(argv, (str, bytes)):
        try:
            tokens = shlex.split(os.fsdecode(argv))
        except ValueError:
            tokens = os.fsdecode(argv).split()
    else:
        tokens = []
        for a in argv or []:
            if not isinstance(a, (str, bytes, os.PathLike)):
                continue
            a = os.fsdecode(a)
            # A shell script passed as one argument (sh -c "cd x && git log"): check its words too.
            try:
                tokens.extend(shlex.split(a) if " " in a else [a])
            except ValueError:
                tokens.extend(a.split())
    for i, tok in enumerate(tokens):
        name = os.path.basename(tok)
        # The program itself, or a program started through a shell.
        if (i == 0 or tokens[i - 1] in {"|", "&&", ";", "||", "-c", "exec", "env"}) and (
            name in BLOCKED_PROGRAMS or name.startswith("python")
        ):
            return name
        if "://" in tok or "github.com" in tok:
            return tok
        if SHARED_TMP:
            if (tok.startswith("/") or tok.startswith("..")) and _path_denied(tok):
                return tok
            continue
        if tok.startswith(("/", "..", "~", "$HOME", "${HOME}")) or (i > 0 and tokens[i - 1] in {"cd", "pushd"}):
            if _path_denied(tok.replace("${HOME}", str(HOME)).replace("$HOME", str(HOME)), base):
                return tok
    return None


def _hook(event: str, args) -> None:
    if not _ACTIVE.get():
        return
    if event == "socket.getaddrinfo":
        host = args[0]
        host = host.decode() if isinstance(host, bytes) else host
        if host and host not in _STATE["hosts"]:
            _note("network", str(host))
            raise PermissionError(f"Network access to {host} is not available in this evaluation")
    elif event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "os.spawn"):
        argv = args[0] if event == "os.system" else args[1]
        cwd = args[2] if event == "subprocess.Popen" and len(args) > 2 else None
        bad = _program_denied(argv, cwd)
        if bad:
            _note("process", f"{bad}: {argv}")
            raise PermissionError(f"Running {bad} is not available in this evaluation")
    elif event == "os.chdir" and not SHARED_TMP:
        if args and _path_denied(args[0]):
            _note("file", f"chdir {args[0]}")
            raise PermissionError(f"{args[0]} is outside the repository checkout")
    elif event in ("open", "os.listdir", "os.scandir"):
        path = args[0] if args else None
        if path is not None and _path_denied(path):
            _note("file", str(path))
            raise PermissionError(f"{path} is outside the repository checkout")


def install_guard(codebase: Path | str, extra_allowed: list[Path | str] = ()) -> None:
    """Install the audit hook once per process and point it at ``codebase``."""
    set_guard_codebase(codebase, extra_allowed)
    if _STATE["installed"]:
        return
    # Every connection by name or by address goes through getaddrinfo first.
    _STATE["hosts"] = set(ALLOWED_HOSTS)
    sys.addaudithook(_hook)
    _STATE["installed"] = True


def set_guard_codebase(codebase: Path | str, extra_allowed: list[Path | str] = ()) -> None:
    roots = {Path(codebase).resolve(), Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
             Path(sys.exec_prefix).resolve(),
             Path(tempfile.gettempdir()).resolve() if SHARED_TMP else _private_tmp()}
    roots |= {Path(p).resolve() for p in sys.path if p and Path(p).is_absolute() and "site-packages" in p}
    roots |= {Path(p).resolve() for p in extra_allowed}
    _STATE["codebase"] = Path(codebase).resolve()
    _STATE["allowed"] = sorted(roots)


def _private_tmp() -> Path:
    """A fresh temporary directory for this run; agent code that asks for a temp dir gets it."""
    if not _STATE.get("tmp"):
        d = Path(tempfile.mkdtemp(prefix="agent_guard_")).resolve()
        _STATE["tmp"] = d
        os.environ["TMPDIR"] = str(d)
        tempfile.tempdir = str(d)
    return _STATE["tmp"]


@contextmanager
def guarded():
    """Run agent-written code under the guard (no effect before install_guard)."""
    token = _ACTIVE.set(_STATE["installed"])
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def pop_blocked() -> list[tuple[str, str]]:
    """Return and clear the refusals recorded since the last call."""
    out, _STATE["blocked"] = _STATE["blocked"], []
    return out


def isolated_checkout(work_dir: Path, history_repo: Path, commit: str) -> None:
    """Make ``work_dir`` a single-commit git repository holding the tree of ``commit``.

    ``history_repo`` is a full clone (bare or not) outside the agent's reach.
    """
    work_dir = Path(work_dir)
    if "repos_isolated" not in work_dir.resolve().parts:
        raise ValueError(f"refusing to rebuild {work_dir}: isolated checkouts must live under a repos_isolated/ folder")
    work_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            for child in work_dir.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            break
        except OSError:
            # A git process from the previous snapshot may still be writing into .git.
            time.sleep(2)
    else:
        raise RuntimeError(f"could not clear {work_dir}")
    archive = subprocess.Popen(["git", "--git-dir", str(_git_dir(history_repo)), "archive", commit],
                               stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(work_dir)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError(f"git archive {commit} failed in {history_repo}")
    env = {**os.environ, "GIT_AUTHOR_NAME": "snapshot", "GIT_AUTHOR_EMAIL": "snapshot@localhost",
           "GIT_COMMITTER_NAME": "snapshot", "GIT_COMMITTER_EMAIL": "snapshot@localhost",
           "GIT_AUTHOR_DATE": "2000-01-01T00:00:00", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00"}
    quiet = ["-c", "gc.auto=0", "-c", "maintenance.auto=false", "-c", "core.fsmonitor=false"]
    for cmd in (["git", *quiet, "init", "-q"], ["git", *quiet, "add", "-A", "-f"],
                ["git", *quiet, "commit", "-q", "--no-verify", "-m", "snapshot"]):
        subprocess.run(cmd, cwd=work_dir, env=env, check=True, stdout=subprocess.DEVNULL)


def _git_dir(repo: Path) -> Path:
    repo = Path(repo)
    return repo / ".git" if (repo / ".git").exists() else repo


def commit_exists(history_repo: Path, commit: str) -> bool:
    return subprocess.run(["git", "--git-dir", str(_git_dir(history_repo)), "cat-file", "-e", f"{commit}^{{commit}}"],
                          capture_output=True).returncode == 0
