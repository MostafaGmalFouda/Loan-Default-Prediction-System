from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, DoubleType, LongType, IntegerType

from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, StandardScaler,
    MinMaxScaler, VectorAssembler
)
from pyspark.ml import Pipeline
from typing import Optional
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col


# Missing Value Handler
class MissingValueHandler:

    def check_missing(self, df: DataFrame) -> dict:
        total = df.count()
        null_counts = df.select(
            [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
        ).collect()[0].asDict()
        details = []
        for col_name, null_count in null_counts.items():
            details.append({
                "column": col_name,
                "null_count": null_count,
                "null_pct": round((null_count / total) * 100, 2) if total > 0 else 0.0,
                "has_missing": null_count > 0
            })
        has_missing = any(d["has_missing"] for d in details)
        return {"has_missing": has_missing, "total_rows": total, "details": details}

    def _numeric_cols(self, df: DataFrame, cols: list) -> list:
        numeric = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
        return [c for c in cols if c in numeric]

    def _string_cols(self, df: DataFrame, cols: list) -> list:
        string = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
        return [c for c in cols if c in string]

    def fill_mean(self, df: DataFrame, cols: list) -> tuple:
        valid = self._numeric_cols(df, cols)
        if not valid:
            return df, "⚠️ No numeric columns"
        means = df.select([F.mean(F.col(c)).alias(c) for c in valid]).collect()[0].asDict()
        return df.fillna(means), f"✅ Filled {len(valid)} columns with mean"

    def fill_median(self, df: DataFrame, cols: list) -> tuple:
        valid = self._numeric_cols(df, cols)
        if not valid:
            return df, "⚠️ No numeric columns"
        medians = {}
        for c in valid:
            median_val = df.approxQuantile(c, [0.5], 0.01)[0]
            medians[c] = median_val
        return df.fillna(medians), f"✅ Filled {len(valid)} columns with median"

    def fill_mode(self, df: DataFrame, cols: list) -> tuple:
        df_out = df
        for c in cols:
            if c not in df.columns:
                continue
            mode_row = (df.groupBy(c).count().orderBy(F.desc("count"))
                       .filter(F.col(c).isNotNull()).first())
            if not mode_row:
                continue
            mode_val = mode_row[c]
            col_type = df.schema[c].dataType
            if isinstance(col_type, (DoubleType, LongType, IntegerType, NumericType)):
                try:
                    mode_val = float(mode_val)
                except:
                    continue
            else:
                mode_val = str(mode_val)
            df_out = df_out.fillna({c: mode_val})
        return df_out, "✅ Filled with mode"

    def drop_rows(self, df: DataFrame, cols: Optional[list] = None) -> tuple:
        before = df.count()
        df_out = df.dropna(subset=cols) if cols else df.dropna()
        after = df_out.count()
        return df_out, f"✅ Dropped {before - after} rows"


# Duplicate Handler
class DuplicateHandler:

    def check_duplicates(self, df: DataFrame, cols: Optional[list] = None) -> dict:
        total = df.count()
        subset = cols if cols else df.columns
        unique = df.dropDuplicates(subset).count()
        dup_count = total - unique
        return {
            "total_rows": total,
            "unique_rows": unique,
            "duplicate_count": dup_count,
            "duplicate_pct": round((dup_count / total) * 100, 2) if total > 0 else 0.0,
            "has_duplicates": dup_count > 0,
        }

    def drop_duplicates(self, df: DataFrame, cols: Optional[list] = None) -> tuple:
        before = df.count()
        df_out = df.dropDuplicates(cols) if cols else df.dropDuplicates()
        after = df_out.count()
        return df_out, f"✅ Removed {before - after} duplicates"


# Encoding Handler
class EncoderHandler:

    def get_categorical_cols(self, df: DataFrame) -> list:
        return [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]

    def label_encode(self, df: DataFrame, cols: list) -> tuple:
        valid = [c for c in cols if c in df.columns]
        if not valid:
            return df, "⚠️ No valid columns"
        stages = [StringIndexer(inputCol=c, outputCol=f"__tmp_{c}", handleInvalid="keep") for c in valid]
        df_out = Pipeline(stages=stages).fit(df).transform(df)
        for c in valid:
            df_out = df_out.drop(c).withColumnRenamed(f"__tmp_{c}", c)
        return df_out, f"✅ Encoded {len(valid)} columns"

    def onehot_encode(self, df: DataFrame, cols: list) -> tuple:
        valid = [c for c in cols if c in df.columns]
        if not valid:
            return df, "⚠️ No valid columns"
        df_out = df
        string_cols = [c for c in valid if isinstance(df.schema[c].dataType, StringType)]
        if string_cols:
            df_out, _ = self.label_encode(df_out, string_cols)
            indexed_cols = string_cols
        else:
            indexed_cols = valid
        for ic in indexed_cols:
            df_out = df_out.withColumn(ic, F.col(ic).cast(DoubleType()))
        ohe_stages = [OneHotEncoder(inputCol=ic, outputCol=f"{ic}_ohe") for ic in indexed_cols]
        df_out = Pipeline(stages=ohe_stages).fit(df_out).transform(df_out)
        for ic in indexed_cols:
            df_out = df_out.drop(ic).withColumnRenamed(f"{ic}_ohe", ic)
        return df_out, f"✅ Encoded {len(indexed_cols)} columns"


# Scaling Handler
class ScalerHandler:

    def _assemble_and_scale(self, df: DataFrame, cols: list, scaler, output_col: str, step_name: str) -> tuple:
        numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
        valid = [c for c in cols if c in df.columns and c in numeric_cols]

        if not valid:
            return df, "⚠️ No valid numeric columns"

        assembler = VectorAssembler(inputCols=valid, outputCol="_features_vec")
        scaler_model = scaler.setInputCol("_features_vec").setOutputCol("_scaled_vec")

        pipeline = Pipeline(stages=[assembler, scaler_model])
        df_out = pipeline.fit(df).transform(df)

        df_out = df_out.withColumn("_scaled_array", vector_to_array(col("_scaled_vec")))

        for i, c in enumerate(valid):
            df_out = df_out.withColumn(c, col("_scaled_array")[i])

        df_out = df_out.drop("_features_vec", "_scaled_vec", "_scaled_array")

        return df_out, f"✅ {step_name} - Scaled {len(valid)} columns"

    def standard_scale(self, df: DataFrame, cols: list) -> tuple:
        scaler = StandardScaler(withMean=True, withStd=True)
        return self._assemble_and_scale(df, cols, scaler, "scaled_standard", "Standard Scaling")

    def minmax_scale(self, df: DataFrame, cols: list) -> tuple:
        scaler = MinMaxScaler(min=0.0, max=1.0)
        return self._assemble_and_scale(df, cols, scaler, "scaled_minmax", "Min-Max Scaling")


# Main Processing Class
class ProcessingSpark:

    def __init__(self):
        self.missing = MissingValueHandler()
        self.duplicates = DuplicateHandler()
        self.encoder = EncoderHandler()
        self.scaler = ScalerHandler()

    @staticmethod
    def _num_cols(df: DataFrame) -> list:
        return [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

    @staticmethod
    def _cat_cols(df: DataFrame) -> list:
        return [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]

    @staticmethod
    def _null_counts(df: DataFrame) -> dict:
        expr = [F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in df.columns]
        return df.select(expr).collect()[0].asDict()

    # Fill Numerical NaN
    def fill_num_nan(self, df: DataFrame, strategy: str, col: str = None):
        nulls = self._null_counts(df)
        num_cols = self._num_cols(df)
        cols = [col] if col else [c for c in num_cols if nulls.get(c, 0) > 0]
        if not cols:
            raise ValueError("No numeric columns with NaN")
        if strategy == "mean":
            df2, msg = self.missing.fill_mean(df, cols)
        elif strategy == "median":
            df2, msg = self.missing.fill_median(df, cols)
        else:
            df2, msg = self.missing.fill_mode(df, cols)
        return df2, msg

    # Fill Categorical NaN
    def fill_cat_nan(self, df: DataFrame, strategy: str, col: str = None):
        nulls = self._null_counts(df)
        cat_cols = self._cat_cols(df)
        cols = [col] if col else [c for c in cat_cols if nulls.get(c, 0) > 0]
        if not cols:
            raise ValueError("No categorical columns with NaN")
        df2, msg = self.missing.fill_mode(df, cols)
        return df2, msg

    # Encoding
    def encode(self, df: DataFrame, method: str, col: str = None):
        cols = [col] if col else self.encoder.get_categorical_cols(df)
        if not cols:
            raise ValueError("No categorical columns")
        if method == "label":
            return self.encoder.label_encode(df, cols)
        else:
            return self.encoder.onehot_encode(df, cols)

    # Normalization
    def normalize(self, df: DataFrame, method: str, col: str = None):
        cols = [col] if col else self._num_cols(df)
        if not cols:
            raise ValueError("No numeric columns")
        if method == "minmax":
            return self.scaler.minmax_scale(df, cols)
        else:
            return self.scaler.standard_scale(df, cols)

    # Remove Duplicates
    def remove_duplicates(self, df: DataFrame):
        df2, msg = self.duplicates.drop_duplicates(df)
        return df2, msg

    # Handle Imbalance
    @staticmethod
    def handle_imbalance(df: DataFrame, method: str, target_col: str):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        counts = {r[target_col]: r["count"] for r in df.groupBy(target_col).count().collect()}
        max_count = max(counts.values())
        min_count = min(counts.values())
        dfs = []
        if method == "oversample":
            for val, cnt in counts.items():
                sub = df.filter(F.col(target_col) == val)
                if cnt < max_count:
                    fraction = float(max_count) / float(cnt)
                    sub = sub.sample(withReplacement=True, fraction=fraction, seed=42)
                dfs.append(sub)
            msg = f"Oversampling applied (max={max_count})"
        else:
            for val in counts:
                dfs.append(df.filter(F.col(target_col) == val).limit(min_count))
            msg = f"Undersampling applied (min={min_count})"
        result = dfs[0]
        for d in dfs[1:]:
            result = result.union(d)
        return result.orderBy(F.rand(seed=42)), msg

    # VectorAssembler (Final Step Before ML)
    @staticmethod
    def vector_assemble(df: DataFrame, feature_cols: list, target_col: str):
        if not feature_cols:
            raise ValueError("Select feature columns")
        if not target_col:
            raise ValueError("Select target column")

        schema_map = {f.name: f.dataType for f in df.schema.fields}
        missing = [c for c in feature_cols + [target_col] if c not in schema_map]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        non_numeric = [c for c in feature_cols if not isinstance(schema_map[c], NumericType)]
        if non_numeric:
            raise ValueError(f"Non-numeric features: {non_numeric}. Apply Label Encoding first!")

        df2 = df.fillna(0, subset=feature_cols)

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
        df2 = assembler.transform(df2)

        if target_col != "label":
            df2 = df2.withColumnRenamed(target_col, "label")

        return df2.select(["features", "label"]), f"VectorAssembler done — {len(feature_cols)} features"