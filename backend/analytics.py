"""
Analyze uploaded CSV/Excel files: summary stats, missing values,
correlations, and a text summary that can also be embedded for RAG chat.
"""
import io
import pandas as pd
import numpy as np


def load_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "csv":
        return pd.read_csv(io.BytesIO(content))
    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(content))
    raise ValueError(f"Unsupported data file type: {ext}")


def analyze_dataframe(df: pd.DataFrame) -> dict:
    """Compute a full analytics summary for a dataframe."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Basic shape + column info
    column_info = [
        {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            "unique": int(df[col].nunique()),
        }
        for col in df.columns
    ]

    # Summary stats for numeric columns
    numeric_summary = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        numeric_summary = desc.to_dict()

    # Top categories for categorical columns (first 5 columns, top 5 values each)
    categorical_summary = {}
    for col in categorical_cols[:5]:
        value_counts = df[col].value_counts().head(5)
        categorical_summary[col] = value_counts.to_dict()

    # Correlation matrix (numeric columns only)
    correlation = {}
    top_correlations = []
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(2)
        correlation = corr_matrix.to_dict()

        # Pull out the strongest pairs (excluding self-correlation) for the summary board
        pairs = []
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1:]:
                r = corr_matrix.loc[col_a, col_b]
                if pd.notna(r):
                    pairs.append({"col_a": col_a, "col_b": col_b, "r": float(r)})
        pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
        top_correlations = pairs[:3]

    # Outlier detection via IQR method (standard, robust to skew)
    outlier_info = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        if count > 0:
            outlier_info.append({"column": col, "outlier_count": count})
    outlier_info.sort(key=lambda o: o["outlier_count"], reverse=True)

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Overall data quality score (100 = perfect). Penalize missing data and duplicates.
    total_cells = len(df) * len(df.columns) if len(df.columns) else 1
    total_missing = int(df.isna().sum().sum())
    missing_pct_overall = round((total_missing / total_cells) * 100, 2) if total_cells else 0
    duplicate_pct = round((duplicate_rows / len(df)) * 100, 2) if len(df) else 0
    quality_score = max(0, round(100 - (missing_pct_overall * 0.6) - (duplicate_pct * 0.4)))

    # Auto-generated insights (plain-language, for the summary board)
    insights = []
    if missing_pct_overall > 0:
        worst_col = max(column_info, key=lambda c: c["missing_pct"]) if column_info else None
        insights.append(f"{missing_pct_overall}% of all cells are missing overall.")
        if worst_col and worst_col["missing_pct"] > 0:
            insights.append(f"'{worst_col['name']}' has the most missing data ({worst_col['missing_pct']}%).")
    else:
        insights.append("No missing values found — the dataset is complete.")

    if duplicate_rows > 0:
        insights.append(f"{duplicate_rows} duplicate row(s) found ({duplicate_pct}% of rows).")

    if top_correlations:
        top = top_correlations[0]
        strength = "strong" if abs(top["r"]) >= 0.7 else "moderate" if abs(top["r"]) >= 0.4 else "weak"
        insights.append(f"Strongest relationship: '{top['col_a']}' and '{top['col_b']}' ({strength}, r={top['r']}).")

    if outlier_info:
        top_outlier = outlier_info[0]
        insights.append(f"'{top_outlier['column']}' has {top_outlier['outlier_count']} potential outlier value(s).")

    summary_board = {
        "quality_score": quality_score,
        "duplicate_rows": duplicate_rows,
        "missing_pct_overall": missing_pct_overall,
        "top_correlations": top_correlations,
        "outliers": outlier_info[:5],
        "insights": insights,
    }

    # Simple chart data: first numeric column's distribution by bins,
    # or first categorical column's value counts — whichever exists
    chart_data = None
    if numeric_cols:
        col = numeric_cols[0]
        counts, bin_edges = np.histogram(df[col].dropna(), bins=8)
        chart_data = {
            "type": "histogram",
            "column": col,
            "labels": [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)],
            "values": counts.tolist(),
        }
    elif categorical_cols:
        col = categorical_cols[0]
        vc = df[col].value_counts().head(8)
        chart_data = {"type": "bar", "column": col, "labels": vc.index.tolist(), "values": vc.values.tolist()}

    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": column_info,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlation": correlation,
        "chart_data": chart_data,
        "summary_board": summary_board,
    }


def analytics_to_text_summary(filename: str, analytics: dict) -> str:
    """
    Turn the analytics dict into a plain-text summary so it can be chunked
    and embedded for RAG — lets users ask questions about the data in chat.
    """
    lines = [f"Dataset: {filename}"]
    lines.append(f"Rows: {analytics['shape']['rows']}, Columns: {analytics['shape']['columns']}")

    lines.append("\nColumns:")
    for col in analytics["columns"]:
        lines.append(
            f"- {col['name']} ({col['dtype']}): {col['unique']} unique values, "
            f"{col['missing']} missing ({col['missing_pct']}%)"
        )

    if analytics["numeric_summary"]:
        lines.append("\nNumeric column statistics:")
        for col, stats in analytics["numeric_summary"].items():
            lines.append(
                f"- {col}: mean={stats.get('mean')}, min={stats.get('min')}, "
                f"max={stats.get('max')}, std={stats.get('std')}"
            )

    if analytics["categorical_summary"]:
        lines.append("\nTop categories:")
        for col, values in analytics["categorical_summary"].items():
            top_str = ", ".join(f"{k}: {v}" for k, v in values.items())
            lines.append(f"- {col}: {top_str}")

    if analytics["correlation"]:
        lines.append("\nNotable correlations exist between numeric columns — see correlation matrix in the dashboard.")

    if analytics.get("summary_board", {}).get("insights"):
        lines.append("\nKey insights:")
        for insight in analytics["summary_board"]["insights"]:
            lines.append(f"- {insight}")

    return "\n".join(lines)
