from flask import Flask, request, jsonify, render_template

from config import MAX_CONTENT_LEN, SAMPLE_ROWS
from core.app_state     import AppState
from core.spark_manager import SparkManager
from core.data_loader   import DataLoader
from core.data_info     import DataInfoSpark
from core.processing    import ProcessingSpark
from core.visualization import VisualizationSpark
from core.ml_models     import MLModels

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LEN

state  = AppState()
loader = DataLoader()
ml     = MLModels()

# Helpers
def err(msg, code=400):
    return jsonify({"error": str(msg)}), code

def ok(**kwargs):
    return jsonify({"ok": True, **kwargs})

# Pages
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/deploy")
def deploy_page():
    return render_template("deploy.html")

# Upload
@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return err("No file provided")
    try:
        df, path = loader.load_file(file, file.filename)

        state.reset()
        state.df_original = df
        state.csv_path    = path
        state.data_loaded = True

        meta = DataLoader.get_col_meta(df)
        state.update_orig_meta(meta)

        dup = DataInfoSpark.get_duplicates(df)
        state.has_duplicates = dup["has_duplicates"]

        return ok(
            shape = [df.count(), len(df.columns)],
            table = DataLoader.df_to_sample(df, 20),
            **state.to_payload()
        )
    except Exception as e:
        return err(e)

# Data Info
@app.route("/api/data_info", methods=["POST"])
def data_info():
    if not state.data_loaded:
        return err("Upload data first")

    body   = request.json or {}
    action = body.get("action", "head")
    processed = body.get("processed", False)

    if processed and not state.processing_done:
        return err("No processing applied yet")

    df = state.df_processed if (processed and state.df_processed is not None) \
         else state.df_original

    try:
        if action == "head":
            return ok(table=DataInfoSpark.get_head(df, 10))
        elif action == "columns_info":
            return ok(table=DataInfoSpark.get_columns_info(df))
        elif action == "describe":
            return ok(table=DataInfoSpark.get_describe(df))
        elif action == "missing":
            return ok(table=DataInfoSpark.get_missing(df))
        elif action == "duplicates":
            info = DataInfoSpark.get_duplicates(df)
            state.has_duplicates = info["has_duplicates"]
            return ok(message=info["message"], has_duplicates=info["has_duplicates"])
        elif action == "value_counts":
            col = body.get("column")
            res = DataInfoSpark.get_value_counts(df, col, top_n=10)
            if not res:
                return err("Column not found")
            return jsonify(res)
        return err(f"Unknown action: {action}")
    except Exception as e:
        return err(e)

# Processing
def _get_nan_columns_after_processing(df):
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType, StringType
    
    num_nan_cols = []
    cat_nan_cols = []
    numeric_cols = []
    
    for field in df.schema.fields:
        col_name = field.name
        null_count = df.filter(F.col(col_name).isNull()).count()
        if null_count > 0:
            if isinstance(field.dataType, NumericType):
                num_nan_cols.append(col_name)
            elif isinstance(field.dataType, StringType):
                cat_nan_cols.append(col_name)
        if isinstance(field.dataType, NumericType):
            numeric_cols.append(col_name)
    
    return num_nan_cols, cat_nan_cols, numeric_cols

@app.route("/api/processing", methods=["POST"])
def processing():
    if not state.data_loaded:
        return err("Upload data first")

    body   = request.json or {}
    action = body.get("action")
    col    = body.get("column") or None
    df     = state.get_active_df()

    try:
        proc = ProcessingSpark()
        df2  = None
        msg  = None

        if action in ("fill_num_mean", "fill_num_median", "fill_num_mode"):
            df2, msg = proc.fill_num_nan(df, action.replace("fill_num_", ""), col)

        elif action in ("fill_cat_mode", "fill_cat_unknown"):
            df2, msg = proc.fill_cat_nan(df, action.replace("fill_cat_", ""), col)

        elif action in ("encode_label", "encode_onehot"):
            df2, msg = proc.encode(df, action.replace("encode_", ""), col)

        elif action in ("normalize_minmax", "normalize_standard"):
            df2, msg = proc.normalize(df, action.replace("normalize_", ""), col)

        elif action == "remove_duplicates":
            df2, msg = proc.remove_duplicates(df)
            state.has_duplicates = False

        elif action in ("handle_imbalance_oversample", "handle_imbalance_undersample"):
            method = "oversample" if "oversample" in action else "undersample"
            if not state.target_col:
                return err("Set target column first! Go to ML Models tab → Set Target")
            df2, msg = proc.handle_imbalance(df, method, state.target_col)

        else:
            return err(f"Unknown action: {action}")

        df2.cache()
        state.df_processed    = df2
        state.processing_done = True
        state.vector_ready    = False

        if action not in state.applied_steps:
            state.applied_steps.append(action)

        num_nan_cols, cat_nan_cols, numeric_cols = _get_nan_columns_after_processing(df2)
        state.update_processed_nan_cols(num_nan_cols, cat_nan_cols, numeric_cols)

        meta = DataLoader.get_col_meta(df2)
        state.update_proc_meta(meta)

        payload = state.to_payload()
        
        return jsonify({
            "ok": True,
            "message": msg,
            "table": DataLoader.df_to_sample(df2, SAMPLE_ROWS),
            "shape": [df2.count(), len(df2.columns)],
            "processed_num_nan_cols": num_nan_cols,
            "processed_cat_nan_cols": cat_nan_cols,
            "processed_num_cols": numeric_cols,
            **payload
        })
    except Exception as e:
        return err(e)

# VectorAssembler
@app.route("/api/vector_assemble", methods=["POST"])
def vector_assemble():
    if not state.data_loaded:
        return err("Upload data first")

    body         = request.json or {}
    feature_cols = body.get("feature_cols", [])
    target_col   = state.target_col

    if not feature_cols:
        return err("Select feature columns")
    
    if not target_col:
        return err("⚠️ Select target column first (Processing tab → Step 1️⃣)")

    df = state.get_active_df()

    if target_col not in df.columns:
        return err(f"Target column '{target_col}' not found in dataframe. Available columns: {df.columns}")

    try:
        proc   = ProcessingSpark()
        vec_df, msg = proc.vector_assemble(df, feature_cols, target_col)

        vec_df.cache()
        state.vector_df    = vec_df
        state.feature_cols = feature_cols
        state.vector_ready = True

        payload = state.to_payload()
        return jsonify({
            "ok":     True,
            "message": msg,
            "shape":  [vec_df.count(), len(vec_df.columns)],
            **payload
        })
    except Exception as e:
        return err(e)

# Set Target Column
@app.route('/api/set_target', methods=['POST'])
def set_target():
    if not state.data_loaded:
        return err("Upload data first")

    data = request.get_json()
    target = data.get('target')

    if not target:
        return err("Target is required")

    df = state.get_active_df()

    if target not in df.columns:
        return err(f"Column '{target}' not found")

    state.target_col = target

    return jsonify({
        "ok": True,
        "target_col": target
    })

# Visualization
@app.route("/api/visualization", methods=["POST"])
def visualization():
    if not state.data_loaded:
        return err("Upload data first")

    body       = request.json or {}
    chart_type = body.get("chart_type")
    col        = body.get("column")
    col2       = body.get("column2")
    use_proc   = body.get("processed", False)

    if use_proc and not state.processing_done:
        return err("No processing applied yet")

    if use_proc and state.df_processed is not None:
        df = state.df_processed
    else:
        df = state.df_original

    try:
        img = VisualizationSpark.generate(df, chart_type, col, col2)
        return ok(image=img)
    except Exception as e:
        return err(e)

# ML - Train
@app.route("/api/train", methods=["POST"])
def train():
    if not state.vector_ready or state.vector_df is None:
        return err("Run VectorAssembler first")

    body     = request.json or {}
    model_id = body.get("model_id", "lr")

    vec_df = state.vector_df
    if "label" not in vec_df.columns:
        return err("Vector DataFrame missing 'label' column. VectorAssembler must rename target to 'label'.")

    try:
        result = ml.train(vec_df, model_id)
        state.current_model_path = result["model_path"]
        state.current_model_id   = model_id
        return ok(**{k: v for k, v in result.items() if k != "ok"})
    except Exception as e:
        return err(e)

# Scores History
@app.route("/api/scores_history", methods=["GET"])
def scores_history():
    return jsonify(ml.load_scores_history())

# Save Model
@app.route("/api/save_model", methods=["POST"])
def save_model():
    if not state.current_model_path:
        return err("No trained model — train first")
    return ok(
        message    = "Model saved ✅",
        model_path = state.current_model_path,
        model_id   = state.current_model_id,
        can_deploy = True,
    )

# Predict (Deploy)
@app.route("/api/predict", methods=["POST"])
def predict():
    body       = request.json or {}
    model_path = body.get("model_path") or state.current_model_path
    model_id   = body.get("model_id")   or state.current_model_id
    input_data = body.get("input_data", {})

    if not model_path or not model_id:
        return err("No saved model — train first")
    if not state.feature_cols:
        return err("Feature columns not found — re-run VectorAssembler")

    try:
        spark = SparkManager.get_session()
        model = MLModels.load_model(model_path, model_id)

        row = {c: float(input_data.get(c, 0)) for c in state.feature_cols}
        df  = spark.createDataFrame([row])

        from pyspark.ml.feature import VectorAssembler
        df2 = VectorAssembler(inputCols=state.feature_cols,
                              outputCol="features").transform(df)

        pred     = MLModels.predict(model, df2, model_id)
        pred_val = pred["prediction"] if isinstance(pred, dict) else pred.select("prediction").collect()[0]["prediction"]
        return ok(prediction=float(pred_val), features_used=state.feature_cols)
    except Exception as e:
        return err(e)

# Columns Endpoint
@app.route("/api/columns", methods=["GET"])
def get_columns():
    return jsonify(state.to_payload())

# Col Names with Before/After Support
@app.route("/api/col_names", methods=["GET"])
def col_names():
    return jsonify({
        "orig_columns":  state.orig_columns,
        "orig_cat_cols": state.orig_cat_cols,
        "orig_num_cols": state.orig_num_cols,
        "proc_columns":  state.proc_columns,
        "proc_cat_cols": state.proc_cat_cols,
        "proc_num_cols": state.proc_num_cols,
        "columns":       state.proc_columns if state.processing_done else state.orig_columns,
        "cat_cols":      state.proc_cat_cols if state.processing_done else state.orig_cat_cols,
        "num_cols":      state.proc_num_cols if state.processing_done else state.orig_num_cols,
        "num_nan_cols":  state.processed_num_nan_cols if state.processing_done else state.orig_num_nan_cols,
        "cat_nan_cols":  state.processed_cat_nan_cols if state.processing_done else state.orig_cat_nan_cols,
        "processed_num_nan_cols": state.processed_num_nan_cols,
        "processed_cat_nan_cols": state.processed_cat_nan_cols,
        "processed_num_cols": state.processed_num_cols,
        "applied_steps":   state.applied_steps,
        "has_duplicates":  state.has_duplicates,
        "vector_ready":    state.vector_ready,
        "processing_done": state.processing_done,
        "data_loaded":     state.data_loaded,
        "feature_cols":    state.feature_cols,
        "target_col":      state.target_col,
    })

# Models List
@app.route("/api/models_list", methods=["GET"])
def models_list():
    return jsonify({"models": ml.load_scores_history()})

if __name__ == "__main__":
    app.run(debug=True, port=5000)