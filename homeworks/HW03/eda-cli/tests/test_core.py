from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_quality_flags_new_heuristics():
    """Тест для проверки новых эвристик качества данных."""

    # Создаем DataFrame с константной колонкой и категориальной колонкой с высокой кардинальностью
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],  # обычная колонка
        "constant_col": [1, 1, 1, 1, 1],  # константная колонка (все значения одинаковые)
        "high_card_col": [f"value_{i}" for i in range(5)],  # высокая кардинальность (5 уникальных из 5)
        "numeric_col": [10.5, 20.3, 15.2, 18.7, 12.9],  # обычная числовая колонка
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Проверяем флаг has_constant_columns (должен быть True из-за constant_col)
    assert flags["has_constant_columns"] == True, \
        f"Ожидалось has_constant_columns=True для константной колонки, но получено {flags['has_constant_columns']}"

    # Проверяем флаг has_high_cardinality_categoricals (должен быть True из-за high_card_col)
    # В high_card_col 5 уникальных значений из 5 строк = 100% уникальности > 50% порога
    assert flags["has_high_cardinality_categoricals"] == True, \
        f"Ожидалось has_high_cardinality_categoricals=True для колонки с высокой кардинальностью, " \
        f"но получено {flags['has_high_cardinality_categoricals']}"

    # Проверяем корректность качества score
    assert 0.0 <= flags["quality_score"] <= 1.0
    # Ожидаем снижение score из-за константной колонки и высокой кардинальности
    # Начальный score 1.0 - 0.1 (has_constant_columns) - 0.15 (has_high_cardinality_categoricals) = 0.75
    expected_min_score = 0.7  # С учетом возможного округления
    assert flags["quality_score"] <= expected_min_score, \
        f"Ожидался score ≤ {expected_min_score} из-за проблемных колонок, но получен {flags['quality_score']}"

    # Создаем DataFrame БЕЗ проблемных колонок для проверки False случаев
    df_good = pd.DataFrame({
        "id": [1, 2, 3],
        "category": ["A", "B", "A"],  # 2 уникальных из 3 (66.6% > 50%)
        "value": [10, 20, 30],
    })

    summary_good = summarize_dataset(df_good)
    missing_df_good = missing_table(df_good)
    flags_good = compute_quality_flags(summary_good, missing_df_good)

    # В этом DF нет константных колонок
    assert flags_good["has_constant_columns"] == False, \
        f"Ожидалось has_constant_columns=False для DF без константных колонок"

    # Проверяем конкретные значения в summary для отладки
    high_card_col_summary = next(c for c in summary.columns if c.name == "high_card_col")
    assert high_card_col_summary.unique == 5, f"Ожидалось 5 уникальных значений, но {high_card_col_summary.unique}"
    assert high_card_col_summary.non_null == 5, f"Ожидалось 5 ненулевых значений, но {high_card_col_summary.non_null}"

    constant_col_summary = next(c for c in summary.columns if c.name == "constant_col")
    assert constant_col_summary.unique == 1, f"Ожидалось 1 уникальное значение, но {constant_col_summary.unique}"
    assert constant_col_summary.non_null == 5, f"Ожидалось 5 ненулевых значений, но {constant_col_summary.non_null}"