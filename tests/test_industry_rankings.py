import pandas as pd
import pytest

from src.analysis.industry_rankings import get_industry_mapping


def test_get_industry_mapping_supports_parquet(tmp_path):
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "gsector": [45, 45, 40],
            "sector": ["Information Technology", "Information Technology", "Financials"],
            "other": [1, 2, 3],
        }
    )
    path = tmp_path / "final_dataset.parquet"
    df.to_parquet(path, index=False)

    out = get_industry_mapping(str(path))
    assert list(out.columns) == ["ticker", "gsector", "sector"]
    assert len(out) == 2
    assert set(out["ticker"]) == {"AAA", "BBB"}


def test_get_industry_mapping_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "final_dataset.txt"
    path.write_text("ticker,gsector,sector\nAAA,45,Information Technology\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        get_industry_mapping(str(path))
