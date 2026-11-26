import pytest
import pandas as pd

from src.ir_system.weighting.document_statistics import (
    count_documents,
    calculate_df,
    calculate_idf,
    generate_df_idf_table,
    save_df_idf_table,
)


def test_calculate_df():
    positional_index = {
        "care": {"d1": [2], "d5": [2, 8]},
        "patient": {"d1": [1], "d2": [1]},
        "medical": {"d5": [5]}
    }

    result = calculate_df(positional_index)

    assert result == {
        "care": 2,
        "patient": 2,
        "medical": 1
    }


def test_calculate_idf():
    df = {
        "care": 4,
        "patient": 10,
        "medical": 1
    }

    total_documents = 10

    result = calculate_idf(df, total_documents)

    assert result["care"] == pytest.approx(0.397940)
    assert result["patient"] == pytest.approx(0.0)
    assert result["medical"] == pytest.approx(1.0)


def test_generate_df_idf_table():
    df = {
        "care": 4,
        "patient": 9
    }

    idf = {
        "care": 0.397940,
        "patient": 0.045757
    }

    result = generate_df_idf_table(df, idf)

    expected = pd.DataFrame({
        "term": ["care", "patient"],
        "df": [4, 9],
        "idf": [0.397940, 0.045757]
    })

    pd.testing.assert_frame_equal(result, expected)


def test_save_df_idf_table(tmp_path):
    df_idf_table = pd.DataFrame({
        "term": ["care", "patient"],
        "df": [4, 9],
        "idf": [0.397940, 0.045757]
    })

    save_df_idf_table(df_idf_table, tmp_path)

    output_file = tmp_path / "df_idf_table.csv"

    assert output_file.exists()

    saved_table = pd.read_csv(output_file)

    pd.testing.assert_frame_equal(saved_table, df_idf_table)


def test_calculate_df_empty():
    positional_index = {}

    result = calculate_df(positional_index)

    assert result == {}


def test_calculate_idf_empty():
    df = {}
    total_documents = 0

    result = calculate_idf(df, total_documents)

    assert result == {}


def test_generate_df_idf_table_empty():
    result = generate_df_idf_table({}, {})

    assert result.empty
    assert list(result.columns) == ["term", "df", "idf"]


def test_count_documents(tmp_path):
    (tmp_path / "d1.txt").write_text("patient care")
    (tmp_path / "d2.txt").write_text("medical treatment")
    (tmp_path / "d3.txt").write_text("")
    (tmp_path / "notes.csv").write_text("test")

    result = count_documents(tmp_path)

    assert result == 3


def test_count_documents_empty(tmp_path):
    result = count_documents(tmp_path)

    assert result == 0