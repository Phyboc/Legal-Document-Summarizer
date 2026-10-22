"""Regression tests for the builder's per-document resume journal.

A full IN-ABS build costs hours of Cross-Encoder alignment, so an interrupted run
must be resumable without re-scoring documents that already finished. These tests
drive ``scripts/build_dataset.py`` with a fake encoder, a fake tokenizer and a
fake source dataset, so they run offline and in well under a second.

The behaviour under test:

* every completed document is journaled to
  ``<output_dir>/.<split>.checkpoint.jsonl`` *as it finishes*;
* rerunning with ``--resume`` re-scores only the documents that were not
  journaled, and still writes complete split files;
* the journals are removed once a build completes, so they cannot be mistaken
  for a partial build.

Run:  python -m pytest tests/test_build_resume.py -q
"""
import json
import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _PKG_ROOT / "scripts"
for _path in (str(_PKG_ROOT), str(_SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_dataset as bd  # noqa: E402


class FakeDataset(list):
    """Minimal stand-in for a Hugging Face ``Dataset``."""

    column_names = ["text", "summary", "file"]


def _rows(count: int, split: str = "train") -> FakeDataset:
    return FakeDataset(
        {
            "text": f"judgment body {split} {i}",
            "summary": f"The appeal is dismissed {split} {i}.",
            "file": f"f{i}",
        }
        for i in range(count)
    )


def _fake_build_document(doc_id, split, text, summary, **kwargs):
    build = bd.DocumentBuild(doc_id=doc_id, split=split)
    build.records = [{
        "doc_id": doc_id,
        "chunk_id": 0,
        "section": "reasoning",
        "chunk_text": text,
        "aligned_target": summary,
        "split": split,
        "para_range": [0, 0],
        "chunk_index": 0,
        "structure_fallback": False,
        "matched_sentences": [summary],
        "matched_labels": ["reasoning"],
        "alignment_scores": [0.9],
        "n_chunk_tokens_est": 10,
        "n_target_tokens_est": 5,
        "n_overflow_sentences": 0,
    }]
    build.reference = {
        "doc_id": doc_id, "split": split, "original_reference_summary": summary,
    }
    build.text_sha1 = f"sha-{doc_id}"
    return build


@pytest.fixture()
def patched(monkeypatch):
    """Neutralise the encoder, the tokenizer and the Hub download."""
    calls: list[str] = []

    def fake_require_encoder(allow_stub):
        return "CrossEncoder"

    def fake_tokenizer(text):
        return {"input_ids": text.split()}

    monkeypatch.setattr(bd, "require_real_encoder", fake_require_encoder)
    monkeypatch.setattr(bd, "require_bart_tokenizer", lambda: fake_tokenizer)
    monkeypatch.setattr(bd, "load_split", lambda split, limit=None: _rows(limit or 3, split))

    def build_document(doc_id, split, text, summary, **kwargs):
        calls.append(doc_id)
        if doc_id.endswith("_00001_simulate_crash"):
            raise RuntimeError("simulated crash")
        return _fake_build_document(doc_id, split, text, summary, **kwargs)

    monkeypatch.setattr(bd, "build_document", build_document)
    return calls


def _args(output_dir: Path, extra: list[str] | None = None) -> "bd.argparse.Namespace":
    argv = ["--sample", "3", "--output_dir", str(output_dir)] + (extra or [])
    return bd.parse_args(argv)


def test_second_builder_on_the_same_directory_is_refused(tmp_path, patched):
    # Two builders sharing one --output_dir interleave their append-only journals.
    # This is asserted rather than described: the first lock is held while the
    # second is attempted.
    with bd.BuildLock(tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            bd.run_build(_args(tmp_path))
        assert "Another build already holds" in str(excinfo.value)

    # Released: the same directory is usable again, and the lock file left behind
    # by the earlier holder does not block a later run.
    assert list(tmp_path.glob(".build.lock"))
    bd.run_build(_args(tmp_path))
    records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["doc_id"] for r in records] == ["train_00000", "train_00001", "train_00002"]


def test_duplicate_journal_entries_are_collapsed(tmp_path, patched):
    # Reproduces the observed corruption: 15 journal lines for a 10-document split.
    journal = tmp_path / ".train.checkpoint.jsonl"
    build = _fake_build_document("train_00000", "train", "body", "summary")
    entry = json.dumps(bd.asdict(build), ensure_ascii=False)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text("\n".join([entry, entry, entry]) + "\n", encoding="utf-8")

    entries = bd.read_checkpoint_journal(journal)
    assert len(entries) == 1
    assert entries[0]["doc_id"] == "train_00000"

    # And a resumed build must not double the records of a duplicated document.
    bd.run_build(_args(tmp_path, ["--resume"]))
    records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reappeared = [r for r in records if r["doc_id"] == "train_00000"]
    assert len(reappeared) == 1, reappeared
    assert sorted(r["doc_id"] for r in records) == [
        "train_00000", "train_00001", "train_00002"
    ]


def test_completed_documents_are_journaled_and_journals_are_cleaned(
    tmp_path, patched
):
    bd.run_build(_args(tmp_path))

    # Caught up cleanly: no leftover checkpoint to be mistaken for a partial run.
    assert list(tmp_path.glob(".*.checkpoint.jsonl")) == []
    assert len(patched) == 9  # 3 documents x 3 splits
    for split in ("train", "validation", "test"):
        records = [
            json.loads(line)
            for line in (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 3, split


def test_interrupted_build_resumes_without_rescoring_finished_documents(
    tmp_path, monkeypatch, patched
):
    # Interrupt the build *outside* build_document's own try/except, the way a
    # killed process or a dropped Colab session would.
    original = bd.build_document
    crashed = {"done": False}

    def crash_once(doc_id, split, text, summary, **kwargs):
        if split == "train" and doc_id.endswith("00001") and not crashed["done"]:
            patched.append(doc_id)
            crashed["done"] = True
            raise RuntimeError("simulated crash")
        return original(doc_id, split, text, summary, **kwargs)

    monkeypatch.setattr(bd, "build_document", crash_once)
    with pytest.raises(RuntimeError):
        bd.run_build(_args(tmp_path))

    journal = tmp_path / ".train.checkpoint.jsonl"
    assert journal.exists(), "the finished document must be on disk before the crash"
    journaled = [
        json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["doc_id"] for entry in journaled] == ["train_00000"]

    # Resume: train_00000 must NOT be scored again, and the output must still be
    # a complete build.
    patched.clear()
    bd.run_build(_args(tmp_path, ["--resume"]))

    assert "train_00000" not in patched, "journaled documents must not be re-scored"
    assert set(patched) == {
        "train_00001", "train_00002",
        "validation_00000", "validation_00001", "validation_00002",
        "test_00000", "test_00001", "test_00002",
    }

    records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["doc_id"] for r in records] == ["train_00000", "train_00001", "train_00002"]
    assert list(tmp_path.glob(".*.checkpoint.jsonl")) == []
