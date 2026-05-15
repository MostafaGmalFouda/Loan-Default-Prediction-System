import io, base64, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType


def _fig_to_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close("all")
    return img


class VisualizationSpark:

    # Categorical Charts
    @staticmethod
    def pie_chart(df: DataFrame, col: str) -> str:
        vals = df.groupBy(col).count().orderBy(F.desc("count")).limit(10).toPandas()
        if vals.empty:
            raise ValueError(f"No data for column '{col}'")
        plt.figure(figsize=(8, 6))
        plt.pie(vals["count"], labels=vals[col].astype(str),
                autopct="%1.1f%%", startangle=140)
        plt.title(f"Pie Chart — {col}")
        return _fig_to_b64()

    @staticmethod
    def bar_chart(df: DataFrame, col: str) -> str:
        vals = df.groupBy(col).count().orderBy(F.desc("count")).limit(15).toPandas()
        plt.figure(figsize=(10, 5))
        sns.barplot(x=vals[col].astype(str), y=vals["count"])
        plt.title(f"Bar Chart — {col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return _fig_to_b64()

    @staticmethod
    def hbar_chart(df: DataFrame, col: str) -> str:
        vals = df.groupBy(col).count().orderBy(F.desc("count")).limit(15).toPandas()
        plt.figure(figsize=(10, 5))
        sns.barplot(y=vals[col].astype(str), x=vals["count"], orient="h")
        plt.title(f"Horizontal Bar — {col}")
        plt.tight_layout()
        return _fig_to_b64()

    @staticmethod
    def count_plot(df: DataFrame, col: str) -> str:
        vals = df.groupBy(col).count().orderBy(F.desc("count")).limit(15).toPandas()
        plt.figure(figsize=(10, 5))
        sns.barplot(x=vals[col].astype(str), y=vals["count"])
        plt.title(f"Count Plot — {col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return _fig_to_b64()

    # Numerical Charts
    @staticmethod
    def histogram(df: DataFrame, col: str) -> str:
        pdf = df.select(col).dropna().limit(5000).toPandas()
        plt.figure(figsize=(10, 5))
        plt.hist(pdf[col].astype(float), bins=30, edgecolor="black", color="steelblue")
        plt.title(f"Histogram — {col}")
        plt.xlabel(col); plt.ylabel("Frequency")
        plt.tight_layout()
        return _fig_to_b64()

    @staticmethod
    def box_plot(df: DataFrame, col: str) -> str:
        pdf = df.select(col).dropna().limit(5000).toPandas()
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=pdf, y=col)
        plt.title(f"Box Plot — {col}")
        plt.tight_layout()
        return _fig_to_b64()

    @staticmethod
    def kde_plot(df: DataFrame, col: str) -> str:
        pdf = df.select(col).dropna().limit(5000).toPandas()
        plt.figure(figsize=(10, 5))
        pdf[col].astype(float).plot(kind="kde")
        plt.title(f"KDE Plot — {col}")
        plt.xlabel(col)
        plt.tight_layout()
        return _fig_to_b64()

    @staticmethod
    def scatter_plot(df: DataFrame, col: str, col2: str) -> str:
        pdf = df.select(col, col2).dropna().limit(2000).toPandas()
        plt.figure(figsize=(10, 5))
        plt.scatter(pdf[col].astype(float), pdf[col2].astype(float),
                    alpha=0.5, color="steelblue", s=15)
        plt.xlabel(col); plt.ylabel(col2)
        plt.title(f"Scatter — {col} vs {col2}")
        plt.tight_layout()
        return _fig_to_b64()

    # Heatmap (all numeric columns)
    @staticmethod
    def heatmap(df: DataFrame) -> str:
        num_cols = [f.name for f in df.schema.fields
                    if isinstance(f.dataType, NumericType)]
        if len(num_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for heatmap")
        n = len(num_cols)
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                v = df.stat.corr(num_cols[i], num_cols[j])
                mat[i][j] = mat[j][i] = v if v is not None else 0.0
        import pandas as pd
        corr_df = pd.DataFrame(mat, index=num_cols, columns=num_cols)
        sz = max(8, n)
        plt.figure(figsize=(sz, sz - 1))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, linewidths=.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        return _fig_to_b64()

    # Main Router
    @classmethod
    def generate(cls, df: DataFrame, chart_type: str,
                 col: str = None, col2: str = None) -> str:

        if chart_type == "heatmap":
            return cls.heatmap(df)

        if not col:
            raise ValueError("Select a column")
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {df.columns[:10]}")

        schema_map = {f.name: f.dataType for f in df.schema.fields}
        dtype      = schema_map.get(col)
        is_cat     = isinstance(dtype, StringType)
        is_num     = isinstance(dtype, NumericType)

        cat_charts = {"pie", "bar", "hbar", "count"}
        num_charts = {"hist", "box", "kde", "scatter"}

        if chart_type in cat_charts and not is_cat:
            raise ValueError(f"'{col}' is numeric — Categorical charts need string columns.")
        if chart_type in num_charts and not is_num:
            raise ValueError(f"'{col}' is categorical — Numerical charts need numeric columns.")

        dispatch = {
            "pie":   cls.pie_chart,
            "bar":   cls.bar_chart,
            "hbar":  cls.hbar_chart,
            "count": cls.count_plot,
            "hist":  cls.histogram,
            "box":   cls.box_plot,
            "kde":   cls.kde_plot,
        }
        if chart_type in dispatch:
            return dispatch[chart_type](df, col)

        if chart_type == "scatter":
            if not col2 or col2 not in df.columns:
                raise ValueError("Select a second column for scatter plot")
            return cls.scatter_plot(df, col, col2)

        raise ValueError(f"Unknown chart type: {chart_type}")