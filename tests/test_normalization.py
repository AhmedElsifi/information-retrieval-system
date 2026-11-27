import numpy as np
import pandas as pd
import pytest

from src.ir_system.weighting.tfidf import calculate_vector_lengths
from src.ir_system.weighting.normalization import (
    load_tfidf_matrix,
    load_vector_lengths,
    normalize_matrix,
    save_normalized_lengths,
    save_normalized_matrix,
    verify_normalized_lengths,
)


@pytest.fixture
def tfidf_matrix():
    return pd.DataFrame({
        "d1": [3.0, 4.0, 0.0],
        "d2": [0.0, 0.0, 2.0],
        "d3": [0.0, 0.0, 0.0],
    }, index=["care", "patient", "medical"])


def test_calculate_vector_lengths(tfidf_matrix):
    result = calculate_vector_lengths(tfidf_matrix)

    assert result["d1"] == pytest.approx(5.0)
    assert result["d2"] == pytest.approx(2.0)
    assert result["d3"] == pytest.approx(0.0)


def test_normalize_matrix_values(tfidf_matrix):
    result = normalize_matrix(tfidf_matrix)

    assert result.loc["care", "d1"] == pytest.approx(0.6)
    assert result.loc["patient", "d1"] == pytest.approx(0.8)
    assert result.loc["medical", "d2"] == pytest.approx(1.0)


def test_normalize_matrix_unit_length(tfidf_matrix):
    result = normalize_matrix(tfidf_matrix)
    lengths = calculate_vector_lengths(result)

    assert lengths["d1"] == pytest.approx(1.0)
    assert lengths["d2"] == pytest.approx(1.0)


def test_normalize_matrix_zero_vector(tfidf_matrix):
    result = normalize_matrix(tfidf_matrix)

    assert (result["d3"] == 0.0).all()
    assert not result.isna().any().any()


def test_normalize_matrix_uses_given_lengths(tfidf_matrix):
    lengths = pd.Series({"d1": 5.0, "d2": 2.0, "d3": 0.0})

    result = normalize_matrix(tfidf_matrix, lengths)

    pd.testing.assert_frame_equal(result, normalize_matrix(tfidf_matrix))


def test_normalize_matrix_missing_length(tfidf_matrix):
    lengths = pd.Series({"d1": 5.0, "d2": 2.0})

    with pytest.raises(KeyError):
        normalize_matrix(tfidf_matrix, lengths)


def test_normalize_matrix_preserves_shape(tfidf_matrix):
    result = normalize_matrix(tfidf_matrix)

    assert list(result.index) == list(tfidf_matrix.index)
    assert list(result.columns) == list(tfidf_matrix.columns)


def test_verify_normalized_lengths(tfidf_matrix):
    result = verify_normalized_lengths(normalize_matrix(tfidf_matrix))

    assert result["d1"] == pytest.approx(1.0)
    assert result["d2"] == pytest.approx(1.0)
    assert result["d3"] == pytest.approx(0.0)


def test_verify_normalized_lengths_invalid(tfidf_matrix):
    with pytest.raises(ValueError):
        verify_normalized_lengths(tfidf_matrix)


def test_normalize_matrix_real_weights():
    idf_care = np.log10(10 / 3)
    idf_patient = np.log10(10 / 2)
    tfidf_matrix = pd.DataFrame({
        "d1": [(1 + np.log10(2)) * idf_care, 1.0 * idf_patient],
    }, index=["care", "patient"])

    result = normalize_matrix(tfidf_matrix)
    length = np.sqrt(tfidf_matrix["d1"].pow(2).sum())

    assert result.loc["care", "d1"] == pytest.approx(tfidf_matrix.loc["care", "d1"] / length)
    assert calculate_vector_lengths(result)["d1"] == pytest.approx(1.0)


def test_normalize_matrix_empty():
    result = normalize_matrix(pd.DataFrame())

    assert result.empty


def test_save_and_load_roundtrip(tmp_path, tfidf_matrix):
    tfidf_matrix.to_csv(tmp_path / "tfidf_matrix.csv")
    calculate_vector_lengths(tfidf_matrix).to_csv(
        tmp_path / "vector_lengths.csv", header=["vector_length"]
    )

    loaded_matrix = load_tfidf_matrix(tmp_path / "tfidf_matrix.csv")
    loaded_lengths = load_vector_lengths(tmp_path / "vector_lengths.csv")
    normalized = normalize_matrix(loaded_matrix, loaded_lengths)
    normalized_lengths = verify_normalized_lengths(normalized)

    save_normalized_matrix(normalized, tmp_path / "out")
    save_normalized_lengths(normalized_lengths, tmp_path / "out")

    saved = pd.read_csv(tmp_path / "out" / "normalized_tfidf_matrix.csv", index_col=0)
    pd.testing.assert_frame_equal(saved, normalized)
    assert (tmp_path / "out" / "normalized_vector_lengths.csv").exists()


def test_load_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tfidf_matrix(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        load_vector_lengths(tmp_path / "missing.csv")
