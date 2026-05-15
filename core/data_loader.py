import os
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from core.spark_manager import SparkManager
from config import UPLOAD_FOLDER, SAMPLE_ROWS


class DataLoader:

    def __init__(self):
        self.spark = SparkManager.get_session()

    # Load file (CSV or Excel) to Spark DataFrame
    def load_file(self, file_storage, filename: str) -> tuple[DataFrame, str]:
        ext  = os.path.splitext(filename)[1].lower()
        path = os.path.join(UPLOAD_FOLDER, filename)

        file_storage.save(path)

        if ext == ".csv":
            df = self._load_csv(path)
        elif ext in (".xls", ".xlsx"):
            import pandas as pd
            csv_path = path.replace(ext, ".csv")
            pd.read_excel(path).to_csv(csv_path, index=False)
            df = self._load_csv(csv_path)
            path = csv_path
        else:
            raise ValueError("Only CSV or Excel files are supported")

        return df, path

    # Load CSV with inferSchema and repartition
    def _load_csv(self, path: str) -> DataFrame:
        df = (self.spark.read
              .option("header",      "true")
              .option("inferSchema", "true")
              .option("mode",        "DROPMALFORMED")
              .csv(path))
        n_parts = max(2, df.rdd.getNumPartitions())
        return df.repartition(n_parts)

    # Convert Spark DF to dict for display (sample only)
    @staticmethod
    def df_to_sample(df: DataFrame, max_rows: int = SAMPLE_ROWS) -> dict:
        total = df.count()
        shown = min(max_rows, total)
        pdf   = df.limit(shown).toPandas().fillna("NaN")
        return {
            "columns":    list(pdf.columns),
            "rows":       [list(map(str, r)) for r in pdf.values.tolist()],
            "total_rows": total,
            "shown_rows": shown,
        }

    # Extract column metadata from Spark DataFrame
    @staticmethod
    def get_col_meta(df: DataFrame) -> dict:
        from pyspark.sql.types import NumericType, StringType

        num_cols = [f.name for f in df.schema.fields
                    if isinstance(f.dataType, NumericType)]
        cat_cols = [f.name for f in df.schema.fields
                    if isinstance(f.dataType, StringType)]

        null_expr = [F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in df.columns]
        null_row  = df.select(null_expr).collect()[0].asDict()

        num_nan_cols = [c for c in num_cols if null_row.get(c, 0) > 0]
        cat_nan_cols = [c for c in cat_cols if null_row.get(c, 0) > 0]

        return {
            "columns":      list(df.columns),
            "num_cols":     num_cols,
            "cat_cols":     cat_cols,
            "num_nan_cols": num_nan_cols,
            "cat_nan_cols": cat_nan_cols,
            "null_counts":  null_row,
        }