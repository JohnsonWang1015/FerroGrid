"""PATH handling in ferro-setup: detection must not warn about a working
PATH, and --add-to-path must be safe to re-run."""

import os
from pathlib import Path

import pytest

from ferro_setup import add_to_path, on_path, shell_rc


def test_on_path_accepts_equivalent_spellings(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for spelling in (str(bin_dir), f"{bin_dir}/", f"{bin_dir}/../bin"):
        assert on_path(bin_dir, os.pathsep.join(["/usr/bin", spelling]))


def test_on_path_follows_symlinked_home(tmp_path):
    real = tmp_path / "real" / "bin"
    real.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "real")
    assert on_path(real, str(link / "bin"))


def test_on_path_false_when_absent(tmp_path):
    assert not on_path(tmp_path / "bin", "/usr/bin:/bin")
    assert not on_path(tmp_path / "bin", "")


@pytest.mark.parametrize(
    "shell,expected",
    [("bash", ".bashrc"), ("/usr/bin/zsh", ".zshrc"), ("fish", "config.fish")],
)
def test_shell_rc_known_shells(shell, expected):
    target = shell_rc(shell)
    assert target is not None
    assert target[0].name == expected


def test_shell_rc_unknown_shell_gives_up():
    assert shell_rc("/bin/tcsh") is None


def test_add_to_path_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.delenv("ZDOTDIR", raising=False)
    bin_dir = tmp_path / ".local" / "bin"

    rc = add_to_path(bin_dir)
    assert rc == tmp_path / ".bashrc"
    first = rc.read_text()
    assert str(bin_dir) in first

    assert add_to_path(bin_dir) is None
    assert rc.read_text() == first


def test_add_to_path_declines_unknown_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/tcsh")
    assert add_to_path(tmp_path / "bin") is None
    assert not list(tmp_path.iterdir())


def test_add_to_path_creates_fish_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    bin_dir = tmp_path / ".local" / "bin"

    rc = add_to_path(bin_dir)
    assert rc == tmp_path / ".config" / "fish" / "config.fish"
    assert f'fish_add_path "{bin_dir}"' in rc.read_text()
