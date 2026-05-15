import os, json, time, io, base64, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from pyspark.sql import DataFrame
from pyspark.ml.classification import (
    LogisticRegression as SparkLR,
    RandomForestClassifier as SparkRF,
    LinearSVC as SparkSVM,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from config import MODELS_FOLDER, TEST_SIZE, RANDOM_STATE

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def _fig_to_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close("all")
    return img


# Model Classes
class LogisticRegressionModel:
    name = "Logistic Regression"
    model_id = "lr"
    uses_spark = True

    @staticmethod
    def build():
        return SparkLR(featuresCol="features", labelCol="label", maxIter=100, regParam=0.01)


class RandomForestModel:
    name = "Random Forest"
    model_id = "rf"
    uses_spark = True

    @staticmethod
    def build():
        return SparkRF(featuresCol="features", labelCol="label", numTrees=100, seed=RANDOM_STATE)


class SVMModel:
    name = "SVM (LinearSVC)"
    model_id = "svm"
    uses_spark = True

    @staticmethod
    def build():
        return SparkSVM(featuresCol="features", labelCol="label", maxIter=100)


class XGBoostModel:
    name = "XGBoost"
    model_id = "xgb"
    uses_spark = False

    @staticmethod
    def build():
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost not installed. Run: pip install xgboost")
        return xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE
        )


# Main MLModels Class
class MLModels:

    MODEL_MAP = {
        "lr": LogisticRegressionModel,
        "rf": RandomForestModel,
        "svm": SVMModel,
        "xgb": XGBoostModel,
    }

    def __init__(self):
        self.scores_file = os.path.join(MODELS_FOLDER, "scores_history.json")
        self._ensure_scores_file()

    def _ensure_scores_file(self):
        if not os.path.exists(self.scores_file):
            with open(self.scores_file, "w") as f:
                json.dump([], f)

    def load_scores_history(self) -> list:
        with open(self.scores_file, "r") as f:
            return json.load(f)

    def _save_score(self, entry: dict):
        history = self.load_scores_history()
        history.append(entry)
        with open(self.scores_file, "w") as f:
            json.dump(history, f, indent=2)

    # Generate Confusion Matrix Image
    @staticmethod
    def _confusion_matrix_img(y_true, y_pred, labels) -> str:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        return _fig_to_b64()

    # Generate ROC Curve Image
    @staticmethod
    def _roc_curve_img(y_true, y_proba, n_classes) -> str:
        classes = list(range(n_classes))
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[1], lw=2, label=f"AUC = {roc_auc_val:.3f}")
        else:
            y_bin = label_binarize(y_true, classes=classes)
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc_val = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"Class {i} (AUC = {roc_auc_val:.3f})")

        plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (One-vs-Rest)")
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        return _fig_to_b64()

    # Train Spark MLlib Models (LR, RF, SVM)
    def _train_spark(self, vector_df: DataFrame, model_class, start_time: float) -> dict:
        train_df, test_df = vector_df.randomSplit([1 - TEST_SIZE, TEST_SIZE], seed=RANDOM_STATE)
        train_df.cache()
        test_df.cache()

        clf = model_class.build()
        fitted_model = clf.fit(train_df)
        predictions = fitted_model.transform(test_df)

        mc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        accuracy = round(float(mc_eval.setMetricName("accuracy").evaluate(predictions)), 4)
        f1_score = round(float(mc_eval.setMetricName("f1").evaluate(predictions)), 4)

        has_prob = model_class.model_id != "svm"

        if has_prob:
            pred_pdf = predictions.select("label", "prediction", "probability").toPandas()
            y_proba = np.array(pred_pdf["probability"].apply(lambda v: list(v)).tolist())
        else:
            pred_pdf = predictions.select("label", "prediction").toPandas()
            n_cls = len(pred_pdf["label"].unique())
            y_proba = np.eye(n_cls)[pred_pdf["prediction"].values.astype(int)]

        y_true = pred_pdf["label"].values.astype(int)
        y_pred = pred_pdf["prediction"].values.astype(int)
        labels = sorted(np.unique(y_true).tolist())
        n_classes = len(labels)

        train_df.unpersist()
        test_df.unpersist()

        return {
            "fitted_model": fitted_model, "y_true": y_true, "y_pred": y_pred,
            "y_proba": y_proba, "labels": labels, "n_classes": n_classes,
            "accuracy": accuracy, "f1_score": f1_score,
            "train_size": train_df.count(), "test_size": test_df.count(),
        }

    # Train XGBoost on Driver (pandas bridge)
    def _train_xgboost(self, vector_df: DataFrame, start_time: float) -> dict:
        from sklearn.metrics import accuracy_score, f1_score as sk_f1

        train_df, test_df = vector_df.randomSplit([1 - TEST_SIZE, TEST_SIZE], seed=RANDOM_STATE)

        def spark_to_numpy(df):
            rows = df.select("features", "label").collect()
            X = np.array([row["features"].toArray() for row in rows])
            y = np.array([int(row["label"]) for row in rows])
            return X, y

        X_train, y_train = spark_to_numpy(train_df)
        X_test, y_test = spark_to_numpy(test_df)

        clf = XGBoostModel.build()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        n_classes = len(np.unique(y_train))
        avg = "binary" if n_classes == 2 else "weighted"

        accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
        f1 = round(float(sk_f1(y_test, y_pred, average=avg, zero_division=0)), 4)
        labels = sorted(np.unique(y_test).tolist())

        return {
            "fitted_model": clf, "y_true": y_test, "y_pred": y_pred,
            "y_proba": y_proba, "labels": labels, "n_classes": n_classes,
            "accuracy": accuracy, "f1_score": f1,
            "train_size": len(X_train), "test_size": len(X_test),
        }

    # Save Model to Disk
    @staticmethod
    def _save_model(fitted_model, model_id: str) -> str | None:
        save_path = os.path.join(MODELS_FOLDER, f"model_{model_id}")

        try:
            if model_id == "xgb":
                import joblib
                os.makedirs(save_path, exist_ok=True)
                joblib_path = os.path.join(save_path, "xgb_model.joblib")
                joblib.dump(fitted_model, joblib_path)
            else:
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                fitted_model.write().overwrite().save(save_path)
            print(f"✅ Model saved → {save_path}")
            return save_path
        except Exception as e:
            print(f"⚠️ Save failed: {e}")
            return None

    # Main Train Entry Point
    def train(self, vector_df: DataFrame, model_id: str) -> dict:
        if model_id not in self.MODEL_MAP:
            raise ValueError(f"Unknown model: {model_id}")

        if "label" not in vector_df.columns:
            raise ValueError("Vector DataFrame must contain 'label' column.")
        if "features" not in vector_df.columns:
            raise ValueError("Vector DataFrame must contain 'features' column.")

        start_time = time.time()
        model_class = self.MODEL_MAP[model_id]

        if model_class.uses_spark:
            result = self._train_spark(vector_df, model_class, start_time)
        else:
            result = self._train_xgboost(vector_df, start_time)

        cm_img = self._confusion_matrix_img(result["y_true"], result["y_pred"], result["labels"])
        roc_img = self._roc_curve_img(result["y_true"], result["y_proba"], result["n_classes"])

        from sklearn.metrics import classification_report
        report = classification_report(result["y_true"], result["y_pred"], zero_division=0)

        elapsed = round(time.time() - start_time, 1)
        model_path = self._save_model(result["fitted_model"], model_id)

        score_entry = {
            "id": int(time.time()), "model_id": model_id, "model_name": model_class.name,
            "accuracy": result["accuracy"], "f1_score": result["f1_score"],
            "model_path": model_path, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": elapsed,
        }
        self._save_score(score_entry)

        return {
            "ok": True, "model": model_class.name, "model_id": model_id,
            "accuracy": result["accuracy"], "f1_score": result["f1_score"],
            "report": report, "confusion_matrix": cm_img, "roc_curve": roc_img,
            "model_path": model_path, "elapsed_s": elapsed, "score_entry": score_entry,
        }

    # Load Saved Model
    @staticmethod
    def load_model(model_path: str, model_id: str):
        if model_id == "xgb":
            import joblib
            joblib_path = os.path.join(model_path, "xgb_model.joblib")
            if not os.path.exists(joblib_path):
                raise FileNotFoundError(f"XGBoost model not found at: {joblib_path}")
            return joblib.load(joblib_path)

        from pyspark.ml import PipelineModel
        try:
            return PipelineModel.load(model_path)
        except Exception:
            if model_id == "lr":
                from pyspark.ml.classification import LogisticRegressionModel
                return LogisticRegressionModel.load(model_path)
            elif model_id == "rf":
                from pyspark.ml.classification import RandomForestClassificationModel
                return RandomForestClassificationModel.load(model_path)
            elif model_id == "svm":
                from pyspark.ml.classification import LinearSVCModel
                return LinearSVCModel.load(model_path)
            raise ValueError(f"Cannot load model: {model_id}")

    # Predict on New Data
    @staticmethod
    def predict(model, df: DataFrame, model_id: str = None):
        if model_id == "xgb" or not hasattr(model, "transform"):
            rows = df.select("features").collect()
            X = np.array([row["features"].toArray() for row in rows])
            pred = model.predict(X)
            proba = model.predict_proba(X)
            return {"prediction": float(pred[0]), "probability": proba[0].tolist()}
        return model.transform(df)