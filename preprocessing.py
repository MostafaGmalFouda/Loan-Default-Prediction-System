from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, DoubleType, LongType, IntegerType
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, StandardScaler,
    MinMaxScaler, VectorAssembler
)
from pyspark.ml import Pipeline
from typing import Optional
import pandas as pd
 
 
# ─────────────────────────────────────────────────────────────────
# 1. MissingValueHandler
# ─────────────────────────────────────────────────────────────────
class MissingValueHandler:
    """
    Handles detection and imputation of missing values in a PySpark DataFrame.
 
    Methods:
        check_missing(df)           → summary of nulls per column
        fill_mean(df, cols)         → fill nulls with column mean
        fill_median(df, cols)       → fill nulls with column median (approx)
        fill_mode(df, cols)         → fill nulls with column mode
        drop_rows(df, cols)         → drop rows that have nulls in given cols
    """
 
    def check_missing(self, df: DataFrame) -> dict:
        total = df.count()
        null_counts = df.select(
            [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
        ).collect()[0].asDict()
 
        details = []
        for col_name, null_count in null_counts.items():
            details.append({
                "column":      col_name,
                "null_count":  null_count,
                "null_pct":    round((null_count / total) * 100, 2) if total > 0 else 0.0,
                "has_missing": null_count > 0
            })
 
        has_missing = any(d["has_missing"] for d in details)
 
        return {
            "step":        "Missing Value Check",
            "has_missing": has_missing,
            "total_rows":  total,
            "details":     details,
            "message":     (
                f"⚠️ Found missing values in {sum(d['has_missing'] for d in details)} column(s)."
                if has_missing else
                "✅ No missing values found."
            )
        }
 
    # ── helpers ─────────────────────────────────────────────────
 
    def _numeric_cols(self, df: DataFrame, cols: list) -> list:
        numeric = [f.name for f in df.schema.fields
                   if isinstance(f.dataType, NumericType)]
        return [c for c in cols if c in numeric]
 
    def _string_cols(self, df: DataFrame, cols: list) -> list:
        string = [f.name for f in df.schema.fields
                  if isinstance(f.dataType, StringType)]
        return [c for c in cols if c in string]
 
    # ── imputation methods ───────────────────────────────────────
 
    def fill_mean(self, df: DataFrame, cols: list) -> tuple[DataFrame, dict]:
        valid = self._numeric_cols(df, cols)
        if not valid:
            return df, {"step": "Fill Mean", "status": "⚠️ No numeric columns selected.", "filled": []}
 
        means = df.select([F.mean(F.col(c)).alias(c) for c in valid]).collect()[0].asDict()
        df_out = df.fillna(means)
 
        return df_out, {
            "step":    "Fill Mean",
            "status":  "✅ Filled nulls with mean.",
            "filled":  [{"column": c, "value": round(v, 4)} for c, v in means.items()]
        }
 
    def fill_median(self, df: DataFrame, cols: list) -> tuple[DataFrame, dict]:
        valid = self._numeric_cols(df, cols)
        if not valid:
            return df, {"step": "Fill Median", "status": "⚠️ No numeric columns selected.", "filled": []}
 
        medians = {}
        for c in valid:
            median_val = df.approxQuantile(c, [0.5], 0.01)[0]
            medians[c] = median_val
 
        df_out = df.fillna(medians)
 
        return df_out, {
            "step":    "Fill Median",
            "status":  "✅ Filled nulls with approximate median.",
            "filled":  [{"column": c, "value": round(v, 4)} for c, v in medians.items()]
        }
 
    def fill_mode(self, df: DataFrame, cols: list) -> tuple[DataFrame, dict]:
       
        filled_info = []
        df_out = df
 
        for c in cols:
            if c not in df.columns:
                continue
 
            mode_row = (
                df.groupBy(c)
                  .count()
                  .orderBy(F.desc("count"))
                  .filter(F.col(c).isNotNull())
                  .first()
            )
            if not mode_row:
                continue
 
            mode_val = mode_row[c]
 
            # cast the mode value to match the column type to avoid silent failures
            col_type = df.schema[c].dataType
            if isinstance(col_type, (DoubleType, LongType, IntegerType, NumericType)):
                try:
                    mode_val = float(mode_val)
                except (TypeError, ValueError):
                    continue  # skip if cast fails
            else:
                mode_val = str(mode_val)
 
            df_out = df_out.fillna({c: mode_val})
            filled_info.append({"column": c, "value": mode_val})
 
        return df_out, {
            "step":    "Fill Mode",
            "status":  "✅ Filled nulls with mode (most frequent value).",
            "filled":  filled_info
        }
 
    def drop_rows(self, df: DataFrame, cols: Optional[list] = None) -> tuple[DataFrame, dict]:
        before = df.count()
        df_out = df.dropna(subset=cols) if cols else df.dropna()
        after  = df_out.count()
 
        return df_out, {
            "step":         "Drop Rows with Nulls",
            "status":       "✅ Rows dropped successfully.",
            "rows_before":  before,
            "rows_after":   after,
            "rows_dropped": before - after
        }
 
 
# ─────────────────────────────────────────────────────────────────
# 2. DuplicateHandler
# ─────────────────────────────────────────────────────────────────
class DuplicateHandler:
 
    def check_duplicates(self, df: DataFrame, cols: Optional[list] = None) -> dict:
        total  = df.count()
        subset = cols if cols else df.columns
        unique = df.dropDuplicates(subset).count()
        dup_count = total - unique
 
        return {
            "step":             "Duplicate Check",
            "total_rows":       total,
            "unique_rows":      unique,
            "duplicate_count":  dup_count,
            "duplicate_pct":    round((dup_count / total) * 100, 2) if total > 0 else 0.0,
            "checked_cols":     subset,
            "has_duplicates":   dup_count > 0,
            "message": (
                f"⚠️ Found {dup_count} duplicate row(s) ({round((dup_count/total)*100,2)}%)."
                if dup_count > 0 else
                "✅ No duplicate rows found."
            )
        }
 
    def drop_duplicates(self, df: DataFrame, cols: Optional[list] = None) -> tuple[DataFrame, dict]:
        before = df.count()
        df_out = df.dropDuplicates(cols) if cols else df.dropDuplicates()
        after  = df_out.count()
 
        return df_out, {
            "step":          "Drop Duplicates",
            "status":        "✅ Duplicates removed successfully.",
            "rows_before":   before,
            "rows_after":    after,
            "rows_removed":  before - after
        }
 
 
# ─────────────────────────────────────────────────────────────────
# 3. EncoderHandler
# ─────────────────────────────────────────────────────────────────
class EncoderHandler:
 
    def get_categorical_cols(self, df: DataFrame) -> dict:
        cat_cols = [f.name for f in df.schema.fields
                    if isinstance(f.dataType, StringType)]
        return {
            "step":     "Categorical Column Detection",
            "cat_cols": cat_cols,
            "count":    len(cat_cols),
            "message":  (
                f"✅ Found {len(cat_cols)} categorical column(s): {', '.join(cat_cols)}"
                if cat_cols else "ℹ️ No categorical columns found."
            )
        }
 
    def label_encode(self, df: DataFrame, cols: list) -> tuple[DataFrame, dict]:
        valid = [c for c in cols if c in df.columns]
        if not valid:
            return df, {"step": "Label Encoding", "status": "⚠️ No valid columns.", "encoded": []}
 
        stages = [
            StringIndexer(
                inputCol=c,
                outputCol=f"__tmp_{c}",
                handleInvalid="keep",
                stringOrderType="alphabetAsc"
            )
            for c in valid
        ]
        pipeline = Pipeline(stages=stages)
        df_out = pipeline.fit(df).transform(df)
 
        for c in valid:
            df_out = df_out.drop(c).withColumnRenamed(f"__tmp_{c}", c)
 
        return df_out, {
            "step":    "Label Encoding (StringIndexer)",
            "status":  "✅ Label encoding applied (columns replaced in-place).",
            "encoded": [{"column": c, "note": "replaced with numeric index"} for c in valid]
        }
 
    def onehot_encode(self, df: DataFrame, cols: list) -> tuple[DataFrame, dict]:
        
        valid = [c for c in cols if c in df.columns]
        if not valid:
            return df, {"step": "One-Hot Encoding", "status": "⚠️ No valid columns.", "encoded": []}
 
        df_out = df
        encoded_cols = []
 
        string_cols = [c for c in valid
                       if isinstance(df.schema[c].dataType, StringType)]
        if string_cols:
            df_out, _ = self.label_encode(df_out, string_cols)
            indexed_cols = string_cols
        else:
            indexed_cols = valid
 
        for ic in indexed_cols:
            df_out = df_out.withColumn(ic, F.col(ic).cast(DoubleType()))
 
        ohe_stages = [
            OneHotEncoder(inputCol=ic, outputCol=f"{ic}_ohe")
            for ic in indexed_cols
        ]
        pipeline = Pipeline(stages=ohe_stages)
        df_out = pipeline.fit(df_out).transform(df_out)
 
        for ic in indexed_cols:
            encoded_cols.append({"input": ic, "new_col": f"{ic}_ohe"})
 
        return df_out, {
            "step":    "One-Hot Encoding",
            "status":  "✅ One-Hot Encoding applied.",
            "encoded": encoded_cols,
            "note":    "OHE output columns are sparse vectors (MLlib format)."
        }
 
 
# ─────────────────────────────────────────────────────────────────
# 4. ScalerHandler
# ─────────────────────────────────────────────────────────────────
class ScalerHandler:
 
    def _assemble_and_scale(
        self,
        df: DataFrame,
        cols: list,
        scaler,
        output_col: str,
        step_name: str
    ) -> tuple[DataFrame, dict]:
       
        numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
        valid = [c for c in cols if c in df.columns and c in numeric_cols]
 
        if not valid:
            return df, {
                "step":   step_name,
                "status": "⚠️ No valid numeric columns found. Make sure columns are numeric before scaling.",
                "scaled": []
            }
 
        assembler    = VectorAssembler(inputCols=valid, outputCol="_features_vec")
        scaler_model = scaler.setInputCol("_features_vec").setOutputCol(output_col)
 
        pipeline = Pipeline(stages=[assembler, scaler_model])
        df_out   = pipeline.fit(df).transform(df).drop("_features_vec")
 
        return df_out, {
            "step":       step_name,
            "status":     f"✅ Scaling applied. Output column: '{output_col}'",
            "scaled":     valid,
            "output_col": output_col,
            "note":       f"Scaled features are stored as a vector in '{output_col}'."
        }
 
    def standard_scale(self, df: DataFrame, cols: list) -> tuple[DataFrame, dict]:
        scaler = StandardScaler(withMean=True, withStd=True)
        return self._assemble_and_scale(
            df, cols, scaler,
            output_col="scaled_features_standard",
            step_name="Standard Scaling (Z-score)"
        )
 
    def minmax_scale(self, df: DataFrame, cols: list) -> tuple[DataFrame, dict]:
        scaler = MinMaxScaler(min=0.0, max=1.0)
        return self._assemble_and_scale(
            df, cols, scaler,
            output_col="scaled_features_minmax",
            step_name="Min-Max Scaling (0–1)"
        )