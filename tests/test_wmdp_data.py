from __future__ import annotations

import json

from reglu.data.common import normalize_jsonl_records, parse_subset_indices_file
from reglu.data.wmdp import detect_wmdp_corpora


def test_detect_wmdp_corpora(tmp_path):
    (tmp_path / "bio-forget-corpus.jsonl").write_text('{"text":"forget"}\n', encoding="utf-8")
    (tmp_path / "bio-retain-corpus.jsonl").write_text('{"text":"retain"}\n', encoding="utf-8")
    found = detect_wmdp_corpora(str(tmp_path), "bio")
    assert found is not None
    forget_file, retain_file = found
    assert forget_file.name == "bio-forget-corpus.jsonl"
    assert retain_file.name == "bio-retain-corpus.jsonl"


def test_normalize_jsonl_records(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"question": "Q", "answer": "A"}),
                json.dumps({"text": "Plain text"}),
            ]
        ),
        encoding="utf-8",
    )
    records = list(normalize_jsonl_records(path))
    assert records[0] == {"question": "Q", "answer": "A"}
    assert records[1] == {"text": "Plain text"}


def test_parse_subset_indices_file_text(tmp_path):
    path = tmp_path / "indices.txt"
    path.write_text("0, 2\n4\n", encoding="utf-8")
    assert parse_subset_indices_file(str(path)) == [0, 2, 4]
