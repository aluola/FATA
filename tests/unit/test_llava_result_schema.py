import csv

import pytest

from fata.runtimes.llava.result_schema import (
    RESULT_SCHEMA_NAME,
    answer_score_cells,
    canonical_score_text,
    raw_answer_column,
    read_completed_result_ids,
    result_header,
    result_schema,
)


PREFIX = ["Image_ID", "Question"]
SCORES = ["Clean_K576", "FATA_K64"]


def _truth():
    return [
        {
            "image_filename": "open.jpg",
            "question": "How many?",
            "answers": ["two", "2", "three"],
            "type": "open",
        },
        {
            "image_filename": "mc.jpg",
            "question": "Choose.",
            "answers": ["B"],
            "type": "multiple_choice",
        },
    ]


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(result_header(PREFIX, SCORES))
        writer.writerows(rows)


def test_schema_pairs_every_score_with_raw_answer_and_serializes_canonically(tmp_path):
    schema = result_schema(SCORES)
    assert schema["name"] == RESULT_SCHEMA_NAME
    assert schema["version"] == 2
    assert schema["score_format"] == "canonical_fixed_2dp"
    assert schema["score_answer_pairs"] == [
        {
            "score_column": score,
            "raw_answer_column": raw_answer_column(score),
        }
        for score in SCORES
    ]

    truth = _truth()
    open_answers = {"Clean_K576": "two", "FATA_K64": "not present"}
    mc_answers = {"Clean_K576": "Option B.", "FATA_K64": "A"}
    path = tmp_path / "result.csv"
    _write_rows(
        path,
        [
            ["open.jpg", '"How many?"', *answer_score_cells(SCORES, open_answers, truth[0])],
            ["mc.jpg", '"Choose."', *answer_score_cells(SCORES, mc_answers, truth[1])],
        ],
    )
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    assert rows[1][2:4] == ["0.67", "0.00"]
    assert rows[2][2:4] == ["1.00", "0.00"]
    assert read_completed_result_ids(
        path,
        prefix_columns=PREFIX,
        score_columns=SCORES,
        ground_truth_rows=truth,
        expected_values={
            "open.jpg": {"Question": '"How many?"'},
            "mc.jpg": {"Question": '"Choose."'},
        },
    ) == {"open.jpg", "mc.jpg"}


def test_resume_rejects_canonical_range_valid_all_zero_tamper(tmp_path):
    truth = _truth()[:1]
    path = tmp_path / "tampered.csv"
    # Both scalar values look legal, but the first raw answer canonically scores
    # 0.67 and therefore must not be accepted as a completed row.
    _write_rows(
        path,
        [["open.jpg", '"How many?"', "0.00", "0.00", "two", "wrong"]],
    )
    with pytest.raises(RuntimeError, match="score/raw-answer mismatch"):
        read_completed_result_ids(
            path,
            prefix_columns=PREFIX,
            score_columns=SCORES,
            ground_truth_rows=truth,
        )


@pytest.mark.parametrize("stored", ["0.0", "00.00", "+0.00", "nan"])
def test_resume_requires_exact_fixed_two_decimal_score_text(tmp_path, stored):
    truth = _truth()[:1]
    path = tmp_path / "noncanonical.csv"
    _write_rows(
        path,
        [["open.jpg", "ignored", stored, "0.00", "wrong", "wrong"]],
    )
    with pytest.raises(RuntimeError, match="score/raw-answer mismatch"):
        read_completed_result_ids(
            path,
            prefix_columns=PREFIX,
            score_columns=SCORES,
            ground_truth_rows=truth,
        )


def test_resume_rejects_legacy_score_only_header(tmp_path):
    path = tmp_path / "legacy.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*PREFIX, *SCORES])
        writer.writerow(["open.jpg", "ignored", "0.00", "0.00"])
    with pytest.raises(RuntimeError, match=r"answer\+score v2 CSV header"):
        read_completed_result_ids(
            path,
            prefix_columns=PREFIX,
            score_columns=SCORES,
            ground_truth_rows=_truth()[:1],
        )


def test_empty_decoded_answer_is_scored_but_mapping_errors_fail_closed(tmp_path):
    assert canonical_score_text("", _truth()[0]) == "0.00"
    cells = answer_score_cells(
        SCORES, {"Clean_K576": "", "FATA_K64": ""}, _truth()[0]
    )
    assert cells == ["0.00", "0.00", "", ""]

    path = tmp_path / "bad-ground-truth.csv"
    _write_rows(path, [["open.jpg", "ignored", *cells]])
    with pytest.raises(RuntimeError, match="cannot canonically rescore"):
        read_completed_result_ids(
            path,
            prefix_columns=PREFIX,
            score_columns=SCORES,
            ground_truth_rows=[
                {
                    "image_filename": "open.jpg",
                    "question": "How many?",
                    "answers": [],
                    "type": "open",
                }
            ],
        )


def test_answer_score_cells_requires_exact_score_key_set():
    with pytest.raises(ValueError, match="exactly match"):
        answer_score_cells(
            SCORES,
            {"Clean_K576": "two", "unexpected": "two"},
            _truth()[0],
        )
