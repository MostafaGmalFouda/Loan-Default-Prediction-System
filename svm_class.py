from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class SVMModel:
    """
    Support Vector Machine Model for Loan Prediction
    Mirrors the structure of RandomForestModel using PySpark LinearSVC (One-vs-Rest)
    """

    def __init__(self):
        self.model = None
        self.scaler_model = None
        self.feature_names = [
            "Age", "Income", "LoanAmount", "Credit_Score",
            "Employment_Years", "Credit_History", "Has_Defaulted",
            "Dependents", "Gender", "Education_Level",
            "Married", "Job_Type", "Property_Area"
        ]

    def build_features(self, df):
        """
        Build feature vector from raw DataFrame
        """
        # Rename target column
        df = df.withColumnRenamed("Loan_Status", "label")

        # Define feature columns
        feature_cols = [
            "Age", "Income", "LoanAmount", "Credit_Score",
            "Employment_Years", "Credit_History", "Has_Defaulted",
            "Dependents"
        ]
        categorical_cols = ["Gender", "Education_Level", "Married", "Job_Type", "Property_Area"]

        # Combine all features
        all_features = feature_cols + categorical_cols

        # Create vector assembler
        assembler = VectorAssembler(
            inputCols=all_features,
            outputCol="raw_features",
            handleInvalid="skip"
        )

        # Transform DataFrame
        df = assembler.transform(df)
        df = df.dropna()

        # Scale features — important for SVM (unlike RF, SVM is sensitive to scale)
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withMean=True,
            withStd=True
        )
        self.scaler_model = scaler.fit(df)
        df = self.scaler_model.transform(df)

        print("Features ready")
        print(f"Total rows: {df.count()}")

        return df

    def split_data(self, df, ratio=0.8):
        """
        Split data into train and test sets
        """
        train_df, test_df = df.randomSplit([ratio, 1 - ratio], seed=42)
        print(f"Training data: {train_df.count()} rows")
        print(f"Testing data: {test_df.count()} rows")
        return train_df, test_df

    def create_model(self):
        """
        Create SVM classifier (LinearSVC wrapped in OneVsRest for binary/multi-class support)
        """
        lsvc = LinearSVC(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            maxIter=100,
            regParam=0.1,
            tol=1e-6,
            fitIntercept=True,
            standardization=False   # already scaled manually above
        )

        # OneVsRest wrapper — supports multi-class and produces probability column
        ovr = OneVsRest(
            classifier=lsvc,
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction"
        )

        return ovr

    def train_model(self, train_df):
        """
        Train SVM model
        """
        svm = self.create_model()
        print("Training SVM...")
        self.model = svm.fit(train_df)
        print("Training complete!")
        return self.model

    def predict(self, test_df):
        """
        Make predictions using trained model
        """
        print("Making predictions...")
        predictions = self.model.transform(test_df)
        print("Predictions complete")
        return predictions

    def evaluate(self, predictions):
        """
        Calculate all evaluation metrics
        """
        # Accuracy
        acc = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        ).evaluate(predictions)

        # F1 Score
        f1 = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        ).evaluate(predictions)

        # AUC — uses rawPrediction from the first OvR sub-model for binary case
        try:
            auc_score = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
            ).evaluate(predictions)
        except Exception:
            auc_score = float("nan")

        # Precision
        precision = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction",
            metricName="precisionByLabel", metricLabel=1.0
        ).evaluate(predictions)

        # Recall
        recall = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction",
            metricName="recallByLabel", metricLabel=1.0
        ).evaluate(predictions)

        return acc, f1, auc_score, precision, recall

    def print_evaluation(self, predictions):
        """
        Print all evaluation metrics
        """
        acc, f1, auc_score, precision, recall = self.evaluate(predictions)
        print(" SVM - EVALUATION RESULTS")
        print(f"Accuracy:   {acc:.4f}")
        print(f"F1 Score:   {f1:.4f}")
        print(f"AUC:        {auc_score:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")

        return acc, f1, auc_score, precision, recall

    def plot_confusion_matrix(self, predictions):
        """
        Plot confusion matrix heatmap
        """
        pred_pd = predictions.select("label", "prediction").toPandas()
        cm = pd.crosstab(pred_pd["label"], pred_pd["prediction"])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("SVM - Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.tight_layout()
        plt.show()

        return cm

    def plot_roc_curve(self, predictions):
        """
        Plot ROC curve
        SVM doesn't output probabilities natively; we use the decision function
        score (rawPrediction[:, 1]) as the ranking score for ROC.
        """
        pred_pd = predictions.select("label", "rawPrediction").toPandas()
        y_true  = pred_pd["label"]
        # rawPrediction for OvR binary: index 1 = score for positive class
        y_score = pred_pd["rawPrediction"].apply(lambda x: float(x[1]))

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, "b-", lw=2, label=f"SVM (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "r--", lw=2, label="Random Classifier")
        plt.xlabel("False Positive Rate (FPR)", fontsize=12)
        plt.ylabel("True Positive Rate (TPR)", fontsize=12)
        plt.title("ROC Curve - SVM", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return roc_auc

    def plot_precision_recall_curve(self, predictions):
        """
        Plot Precision-Recall curve
        """
        pred_pd = predictions.select("label", "rawPrediction").toPandas()
        y_true  = pred_pd["label"]
        y_score = pred_pd["rawPrediction"].apply(lambda x: float(x[1]))

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)

        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, "b-", linewidth=2)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve - SVM", fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        """
        Plot feature importance bar chart.
        For SVM, importance = mean of |coefficients| across OvR sub-classifiers.
        """
        classifiers = self.model.models          # list of LinearSVCModel (one per class)
        importances = None
        for clf in classifiers:
            coef = abs(clf.coefficients.toArray())
            importances = coef if importances is None else importances + coef
        importances /= len(classifiers)

        fi_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="Blues_r")
        plt.title("SVM - Feature Importance (|Coefficients|)", fontsize=14, fontweight="bold")
        plt.xlabel("Mean |Coefficient|", fontsize=12)
        plt.tight_layout()
        plt.show()

        # Print ranking
        print("\nFeature Importance Ranking (SVM):")
        print("=" * 45)
        for _, row in fi_df.iterrows():
            print(f"{row['Feature']:20} : {row['Importance']:.4f}")

        return fi_df

    def plot_performance_metrics(self, predictions):
        """
        Plot all performance metrics as bar chart
        """
        acc, f1, auc_score, precision, recall = self.evaluate(predictions)

        metrics = ["Accuracy", "F1 Score", "AUC", "Precision", "Recall"]
        scores  = [acc, f1, auc_score, precision, recall]
        colors  = ["#2980b9", "#1a6fa3", "#1a5f8f", "#154f7a", "#103f64"]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, scores, color=colors)
        plt.ylim(0, 1)
        plt.ylabel("Score", fontsize=12)
        plt.title("SVM - Performance Metrics", fontsize=14, fontweight="bold")

        # Add values on top of bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{score:.3f}", ha="center", fontsize=10)

        plt.tight_layout()
        plt.show()

    def save_model(self, path="svm_model"):
        """
        Save trained model
        """
        if self.model:
            self.model.save(path)
            print(f" Model saved to {path}")
        else:
            print("No model to save. Train the model first.")

    def get_feature_importance_df(self):
        """
        Get feature importance as DataFrame
        """
        classifiers = self.model.models
        importances = None
        for clf in classifiers:
            coef = abs(clf.coefficients.toArray())
            importances = coef if importances is None else importances + coef
        importances /= len(classifiers)

        fi_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        return fi_df


# ── Usage ──────────────────────────────────────────────────────────────────────
"""
# Initialize SVM Model
svm_model = SVMModel()

# Build features
df_ready = svm_model.build_features(df)

# Split data
train_df, test_df = svm_model.split_data(df_ready)

# Train model
svm_model.train_model(train_df)

# Make predictions
predictions = svm_model.predict(test_df)

# Evaluate and print results
svm_model.print_evaluation(predictions)

# Visualizations
svm_model.plot_confusion_matrix(predictions)
svm_model.plot_roc_curve(predictions)
svm_model.plot_precision_recall_curve(predictions)
svm_model.plot_feature_importance()
svm_model.plot_performance_metrics(predictions)

# Save model
svm_model.save_model()
"""
