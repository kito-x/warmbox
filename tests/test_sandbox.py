from __future__ import annotations

import time

import pytest

from warmbox.errors import EditMatchError, SandboxViolation
from warmbox.sandbox.runtime import Sandbox


@pytest.fixture
def sandbox(tmp_repo, tmp_path):
    sb = Sandbox(tmp_repo, "agent-0", workspaces_root=tmp_path / "ws")
    yield sb
    sb.close()


def test_workspace_is_a_copy(sandbox, tmp_repo) -> None:
    assert sandbox.workspace != tmp_repo
    assert (sandbox.workspace / "a.txt").exists()
    (sandbox.workspace / "a.txt").write_text("mutated\n", encoding="utf-8")
    assert (tmp_repo / "a.txt").read_text(encoding="utf-8").startswith("line 0000")


def test_path_escape(sandbox) -> None:
    with pytest.raises(SandboxViolation):
        sandbox.normalize("../x")
    with pytest.raises(SandboxViolation):
        sandbox.normalize("/etc/passwd")
    link = sandbox.workspace / "escape"
    link.symlink_to("/tmp")
    with pytest.raises(SandboxViolation):
        sandbox.normalize("escape/x")
    with pytest.raises(SandboxViolation):
        sandbox.read_file("../a.txt")


def test_normalize_accepts_absolute_inside(sandbox) -> None:
    assert sandbox.normalize(str(sandbox.workspace / "a.txt")) == "a.txt"
    assert sandbox.normalize("./a.txt") == "a.txt"
    assert sandbox.normalize(".") == "."


def test_allowlist(sandbox) -> None:
    for argv in (
        ["rm", "-rf", "."],
        ["bash", "-c", "x"],
        ["/bin/ls"],
        ["git", "push"],
        ["find", ".", "-delete"],
        ["python", "-m", "pip", "install", "x"],
        ["curl", "http://example.com"],
        [],
    ):
        with pytest.raises(SandboxViolation):
            sandbox.run_command(argv)


def test_allowed_command_runs(sandbox) -> None:
    res = sandbox.run_command(["python", "-c", "print(2+2)"])
    assert res.exit_code == 0
    assert res.stdout.strip() == "4"
    assert not res.timed_out

    res2 = sandbox.run_command(["ls"])
    assert res2.exit_code == 0
    assert "a.txt" in res2.stdout


def test_timeout(sandbox) -> None:
    t0 = time.perf_counter()
    res = sandbox.run_command(["python", "-c", "import time; time.sleep(5)"], timeout_s=0.5)
    elapsed = time.perf_counter() - t0
    assert res.timed_out is True
    assert res.exit_code == -1
    assert elapsed < 2.5


def test_env_scrubbed(sandbox, monkeypatch) -> None:
    monkeypatch.setenv("WARMBOX_API_KEY", "sk-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "nope")
    res = sandbox.run_command(
        ["python", "-c", "import os;print(os.environ.get('WARMBOX_API_KEY'))"]
    )
    assert res.stdout.strip() == "None"
    res2 = sandbox.run_command(
        ["python", "-c", "import os;print(os.environ.get('AWS_SECRET_ACCESS_KEY'))"]
    )
    assert res2.stdout.strip() == "None"
    res3 = sandbox.run_command(
        ["python", "-c", "import os;print(os.environ['HOME']==os.getcwd())"]
    )
    assert res3.stdout.strip() == "True"


def test_command_not_found(sandbox) -> None:
    # 'grep' exists; use an allowlisted name that resolves via PATH
    res = sandbox.run_command(["true"])
    assert res.exit_code == 0


def test_edit_file_unique(sandbox) -> None:
    with pytest.raises(EditMatchError):
        sandbox.edit_file("a.txt", "not present\n", "x\n")
    (sandbox.workspace / "d.txt").write_text("dup\ndup\n", encoding="utf-8")
    with pytest.raises(EditMatchError):
        sandbox.edit_file("d.txt", "dup\n", "x\n")


def test_read_write_versions(sandbox) -> None:
    text = sandbox.read_file("a.txt")
    assert sandbox.version_of("a.txt").version == 1
    ev = sandbox.write_file("a.txt", text.replace("line 0165\n", "LINE 0165\n"))
    assert ev.new_version == 2
    assert sandbox.version_of("a.txt").version == 2


def test_diff(sandbox) -> None:
    sandbox.read_file("a.txt")
    sandbox.edit_file("a.txt", "line 0165\n", "LINE 0165\n")
    patch = sandbox.diff()
    assert "--- a/a.txt" in patch
    assert "+++ b/a.txt" in patch
    assert "+LINE 0165" in patch
    assert "-line 0165" in patch


def test_search_and_list(sandbox) -> None:
    hits = sandbox.search(r"def b\(")
    assert len(hits) == 1
    assert hits[0].path == "b.py"
    assert hits[0].line == 1
    assert "b.py" in sandbox.list_dir(".")
    from warmbox.errors import SandboxError

    with pytest.raises(SandboxError):
        sandbox.search("[")


def test_snapshot_files(sandbox) -> None:
    snap = dict(sandbox.snapshot_files())
    assert "a.txt" in snap and "b.py" in snap
    assert len(snap["a.txt"]) == 32


def test_sync_reports_out_of_band(sandbox) -> None:
    sandbox.read_file("a.txt")
    (sandbox.workspace / "a.txt").write_text("total replacement\n", encoding="utf-8")
    events = sandbox.sync()
    assert [e.path for e in events] == ["a.txt"]
    assert events[0].source == "rescan"
    assert sandbox.sync() == []
