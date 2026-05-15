from pyspark.sql import DataFrame
from pyspark.sql import functions as F

class DataInfoSpark:

    # Get first n rows
    @staticmethod
    def get_head(df: DataFrame, n: int = 10) -> dict:
        pdf = df.limit(n).toPandas().fillna("NaN")
        return {
            "columns":    list(pdf.columns),
            "rows":       [list(map(str, r)) for r in pdf.values.tolist()],
            "total_rows": df.count(),
            "shown_rows": min(n, df.count()),
        }

    # Get columns info: dtype, non-null, null, unique
    @staticmethod
    def get_columns_info(df: DataFrame) -> dict:
        total = df.count()
        rows  = []
        for field in df.schema.fields:
            c       = field.name
            dtype   = str(field.dataType)
            null_c  = df.filter(F.col(c).isNull()).count()
            uniq_c  = df.select(c).distinct().count()
            rows.append([c, dtype, total - null_c, null_c, uniq_c])
        return {"columns": ["Column", "Dtype", "Non-Null", "Null", "Unique"], "rows": rows}

    # Statistical describe
    @staticmethod
    def get_describe(df: DataFrame) -> dict:
        pdf = df.describe().toPandas().fillna("")
        return {
            "columns": list(pdf.columns),
            "rows":    [list(map(str, r)) for r in pdf.values.tolist()],
        }

    # Get columns with missing values
    @staticmethod
    def get_missing(df: DataFrame) -> dict:
        total     = df.count()
        null_expr = [F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in df.columns]
        null_row  = df.select(null_expr).collect()[0].asDict()
        rows = [
            [c, int(null_row[c]), round(null_row[c] / total * 100, 2)]
            for c in df.columns if null_row[c] > 0
        ]
        if not rows:
            rows = [["✅ No missing values", "-", "-"]]
        return {"columns": ["Column", "Missing Count", "Missing %"], "rows": rows}

    # Check duplicate rows
    @staticmethod
    def get_duplicates(df: DataFrame) -> dict:
        total    = df.count()
        distinct = df.distinct().count()
        dup_count = total - distinct
        return {
            "count":   dup_count,
            "message": f"Duplicated rows: {dup_count}",
            "has_duplicates": dup_count > 0,
        }

    # Get top N value counts for a column
    @staticmethod
    def get_value_counts(df: DataFrame, col: str, top_n: int = 10) -> dict:
        if col not in df.columns:
            return None
        vc = (df.groupBy(col)
              .count()
              .orderBy(F.desc("count"))
              .limit(top_n)
              .toPandas())
        rows        = [[str(r[col]), int(r["count"])] for _, r in vc.iterrows()]
        total_uniq  = df.select(col).distinct().count()
        return {
            "table":        {"columns": [col, "Count"], "rows": rows},
            "column":       col,
            "total_unique": total_uniq,
            "top_n":        top_n,
        }