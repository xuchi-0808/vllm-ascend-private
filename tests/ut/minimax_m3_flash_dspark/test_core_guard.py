# SPDX-License-Identifier: Apache-2.0
"""The opt-in core compatibility patch must be reviewable and fail closed."""

import subprocess
import sys

import pytest

GUARD_LINE = '                    and "K3DSparkModel" not in self.draft_model_config.architectures\n'
ADDITION = '                    and "MiniMaxM31FlashA3DSparkModel" not in self.draft_model_config.architectures\n'


def _run_patch(example_dir, source, *, apply=False):
    command = [sys.executable, str(example_dir / "patch_core_guard.py"), str(source)]
    if apply:
        command.append("--apply")
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_dry_run_then_apply_and_idempotent_reapply(example_dir, tmp_path):
    source = tmp_path / "speculative.py"
    before = "# untouched prefix\n" + GUARD_LINE + "# untouched suffix\n"
    source.write_text(before, encoding="utf-8")
    backup = source.with_suffix(".py.before-minimax-dspark-flash")
    dry = _run_patch(example_dir, source)
    assert dry.returncode == 0, dry.stderr
    assert "+" + ADDITION.rstrip() in dry.stdout
    assert source.read_text(encoding="utf-8") == before
    assert not backup.exists()

    applied = _run_patch(example_dir, source, apply=True)
    assert applied.returncode == 0, applied.stderr
    expected = before.replace(GUARD_LINE, GUARD_LINE + ADDITION)
    assert source.read_text(encoding="utf-8") == expected
    assert backup.read_text(encoding="utf-8") == before
    reapplied = _run_patch(example_dir, source, apply=True)
    assert reapplied.returncode == 0, reapplied.stderr
    assert source.read_text(encoding="utf-8") == expected
    assert backup.read_text(encoding="utf-8") == before
    assert expected.count(ADDITION) == 1


@pytest.mark.parametrize("guard_count", [0, 2])
def test_changed_or_ambiguous_core_source_is_never_modified(example_dir, tmp_path, guard_count):
    source = tmp_path / "speculative.py"
    before = "# unsupported core revision\n" + GUARD_LINE * guard_count
    source.write_text(before, encoding="utf-8")
    result = _run_patch(example_dir, source, apply=True)
    assert result.returncode != 0
    assert "Core version changed" in result.stderr
    assert source.read_text(encoding="utf-8") == before
    assert not source.with_suffix(".py.before-minimax-dspark-flash").exists()


@pytest.mark.parametrize(
    "before",
    [
        ADDITION,
        GUARD_LINE * 2 + ADDITION,
        GUARD_LINE + "# unrelated guard or source block\n" + ADDITION,
        GUARD_LINE + ADDITION * 2,
    ],
    ids=["missing-anchor", "duplicate-anchor", "nonadjacent-addition", "duplicate-addition"],
)
def test_already_patched_requires_one_adjacent_guard_pair(example_dir, tmp_path, before):
    source = tmp_path / "speculative.py"
    source.write_text(before, encoding="utf-8")
    result = _run_patch(example_dir, source, apply=True)
    assert result.returncode != 0
    assert "Core version changed" in result.stderr
    assert source.read_text(encoding="utf-8") == before
    assert not source.with_suffix(".py.before-minimax-dspark-flash").exists()


def test_existing_backup_is_never_overwritten(example_dir, tmp_path):
    source = tmp_path / "speculative.py"
    source.write_text(GUARD_LINE, encoding="utf-8")
    backup = source.with_suffix(".py.before-minimax-dspark-flash")
    backup.write_text("previous original", encoding="utf-8")
    result = _run_patch(example_dir, source, apply=True)
    assert result.returncode != 0
    assert source.read_text(encoding="utf-8") == GUARD_LINE
    assert backup.read_text(encoding="utf-8") == "previous original"
