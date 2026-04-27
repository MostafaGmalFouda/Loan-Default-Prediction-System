from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType, FloatType, LongType
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    """
    Exploratory Data Analysis class using PySpark.
    Designed for loan dataset analysis but applicable to any structured CSV dataset.
    """

    def __init__(self, csv_path: str, app_name: str = "EDA"):
        """
        Initialize SparkSession and load the dataset.

        Args:
            csv_path  : Path to the CSV file.
            app_name  : Name for the Spark application.
        """
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()

        self.df = self.spark.read.csv(csv_path, header=True, inferSchema=True)
        self._object_cols = [f.name for f in self.df.schema.fields if isinstance(f.dataType, StringType)]
        self._numeric_cols = [
            f.name for f in self.df.schema.fields
            if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType))
        ]
        print(f"[EDA] Dataset loaded — Rows: {self.df.count()}, Columns: {len(self.df.columns)}")

    # ------------------------------------------------------------------
    # 1. Basic info
    # ------------------------------------------------------------------

    def show_shape(self):
        """Print the number of rows and columns."""
        print(f"Rows: {self.df.count()}, Columns: {len(self.df.columns)}")

    def show_schema(self):
        """Print the schema (column names and data types)."""
        self.df.printSchema()

    def show_sample(self, n: int = 5):
        """Display the first n rows."""
        self.df.show(n)

    # ------------------------------------------------------------------
    # 2. Statistical summary
    # ------------------------------------------------------------------

    def describe_numeric(self):
        """Statistical summary (count, mean, std, min, max) for numerical columns."""
        self.df.select(self._numeric_cols).describe().show()

    def describe_categorical(self):
        """Statistical summary for categorical (string) columns."""
        if self._object_cols:
            self.df.select(self._object_cols).describe().show()
        else:
            print("[EDA] No categorical columns found.")

    # ------------------------------------------------------------------
    # 3. Missing values
    # ------------------------------------------------------------------

    def missing_counts(self):
        """Show the number of null values per column."""
        missing = self.df.select(
            [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in self.df.columns]
        )
        missing.show()

    def missing_percentage(self):
        """Print the percentage of null values per column."""
        total = self.df.count()
        for col in self.df.columns:
            null_count = self.df.filter(F.col(col).isNull()).count()
            perc = null_count / total * 100
            print(f"{col:<25} {perc:>8.2f}%")

    # ------------------------------------------------------------------
    # 4. Duplicates
    # ------------------------------------------------------------------

    def duplicate_count(self):
        """Print the number of fully duplicated rows."""
        count = self.df.count() - self.df.dropDuplicates().count()
        print(f"Duplicated rows: {count}")

    # ------------------------------------------------------------------
    # 5. Categorical column analysis
    # ------------------------------------------------------------------

    def plot_categorical(self, cols: list = None, subplot_cols: int = 2, figsize: tuple = (15, 10)):
        """
        Count plots for categorical columns.

        Args:
            cols         : List of column names. Defaults to all string columns.
            subplot_cols : Number of subplot columns in the grid.
            figsize      : Figure size tuple.
        """
        cols = cols or self._object_cols
        if not cols:
            print("[EDA] No categorical columns to plot.")
            return

        pdf = self.df.select(cols).toPandas()
        n_rows = -(-len(cols) // subplot_cols)          # ceiling division

        plt.figure(figsize=figsize)
        for i, col in enumerate(cols, 1):
            ax = plt.subplot(n_rows, subplot_cols, i)
            sns.countplot(x=col, data=pdf, ax=ax)
            for container in ax.containers:
                ax.bar_label(container)
            ax.margins(y=0.15)
            plt.xticks(rotation=45)
            plt.title(col)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 6. Numerical column analysis
    # ------------------------------------------------------------------

    def plot_numeric_distributions(self, cols: list = None, subplot_cols: int = 3,
                                   bins: int = 30, color: str = "red", figsize: tuple = (15, 10)):
        """
        Histogram + KDE plots for numerical columns.

        Args:
            cols         : List of column names. Defaults to all numeric columns.
            subplot_cols : Number of subplot columns in the grid.
            bins         : Number of histogram bins.
            color        : Bar color.
            figsize      : Figure size tuple.
        """
        cols = cols or self._numeric_cols
        if not cols:
            print("[EDA] No numerical columns to plot.")
            return

        pdf = self.df.select(cols).toPandas()
        n_rows = -(-len(cols) // subplot_cols)

        plt.figure(figsize=figsize)
        for i, col in enumerate(cols, 1):
            ax = plt.subplot(n_rows, subplot_cols, i)
            sns.histplot(x=col, data=pdf, ax=ax, kde=True, bins=bins, color=color)
            plt.xticks(rotation=45)
            plt.title(col)
        plt.tight_layout()
        plt.show()

    def plot_continuous_distributions(self, cols: list = None, bins: int = 30,
                                      color: str = "red", figsize: tuple = (15, 10)):
        """
        Histogram + KDE for a specific list of continuous columns.
        Defaults to ['Age', 'Income', 'LoanAmount', 'Credit_Score', 'Employment_Years'].
        """
        default_continuous = ['Age', 'Income', 'LoanAmount', 'Credit_Score', 'Employment_Years']
        cols = cols or [c for c in default_continuous if c in self.df.columns]
        self.plot_numeric_distributions(cols=cols, bins=bins, color=color, figsize=figsize)

    # ------------------------------------------------------------------
    # 7. Discrete numerical columns
    # ------------------------------------------------------------------

    def value_counts_discrete(self, cols: list = None):
        """
        Print value counts for discrete numerical columns.

        Args:
            cols : List of column names. Defaults to ['Credit_History', 'Has_Defaulted',
                   'Dependents', 'Loan_Status'] if they exist.
        """
        default_discrete = ['Credit_History', 'Has_Defaulted', 'Dependents', 'Loan_Status']
        cols = cols or [c for c in default_discrete if c in self.df.columns]

        for col in cols:
            print(f"\n{'=' * 45}\n{col}")
            self.df.groupBy(col).count().orderBy(col).show()

    def plot_discrete(self, cols: list = None, subplot_cols: int = 2, figsize: tuple = (15, 10)):
        """
        Count plots for discrete numerical columns.
        """
        default_discrete = ['Credit_History', 'Has_Defaulted', 'Dependents', 'Loan_Status']
        cols = cols or [c for c in default_discrete if c in self.df.columns]
        if not cols:
            print("[EDA] No discrete columns to plot.")
            return

        pdf = self.df.select(cols).toPandas()
        n_rows = -(-len(cols) // subplot_cols)

        plt.figure(figsize=figsize)
        for i, col in enumerate(cols, 1):
            ax = plt.subplot(n_rows, subplot_cols, i)
            sns.countplot(x=pdf[col].dropna(), ax=ax)
            for container in ax.containers:
                ax.bar_label(container)
            ax.margins(y=0.15)
            plt.xticks(rotation=45)
            plt.title(col)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 8. Bivariate / relationship plots
    # ------------------------------------------------------------------

    def plot_countplot_by_target(self, x_col: str, target_col: str = "Loan_Status"):
        """
        Count plot of x_col grouped by target_col (hue).

        Args:
            x_col      : Feature column name.
            target_col : Target/label column name.
        """
        pdf = self.df.select(x_col, target_col).toPandas()
        sns.countplot(x=x_col, hue=target_col, data=pdf)
        plt.title(f"{x_col} vs {target_col}")
        plt.show()

    def plot_barplot_by_hue(self, x_col: str, y_col: str, hue_col: str):
        """
        Bar plot: mean of y_col per x_col, split by hue_col.

        Args:
            x_col   : X-axis column.
            y_col   : Y-axis column (numeric).
            hue_col : Hue / grouping column.
        """
        pdf = self.df.select(x_col, y_col, hue_col).toPandas()
        sns.barplot(x=x_col, y=y_col, data=pdf, hue=hue_col)
        plt.title(f"{x_col} vs {y_col} by {hue_col}")
        plt.show()

    def plot_boxplot(self, x_col: str, y_col: str):
        """
        Box plot of y_col grouped by x_col.

        Args:
            x_col : Grouping column (usually the target).
            y_col : Numeric column.
        """
        pdf = self.df.select(x_col, y_col).toPandas()
        sns.boxplot(x=x_col, y=y_col, data=pdf)
        plt.title(f"{y_col} by {x_col}")
        plt.show()

    # ------------------------------------------------------------------
    # 9. Correlation heatmap
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(self, cols: list = None, figsize: tuple = (10, 8)):
        """
        Heatmap of Pearson correlations for numerical columns.

        Args:
            cols    : List of column names. Defaults to all numeric columns.
            figsize : Figure size tuple.
        """
        cols = cols or self._numeric_cols
        pdf = self.df.select(cols).toPandas()
        corr = pdf.corr(numeric_only=True)

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    # ------------------------------------------------------------------
    # 10. Full EDA pipeline (run everything at once)
    # ------------------------------------------------------------------

    def run_full_eda(self):
        """Run the complete EDA pipeline in one call."""
        print("\n===== SHAPE =====")
        self.show_shape()

        print("\n===== SCHEMA =====")
        self.show_schema()

        print("\n===== SAMPLE (5 rows) =====")
        self.show_sample()

        print("\n===== NUMERIC SUMMARY =====")
        self.describe_numeric()

        print("\n===== CATEGORICAL SUMMARY =====")
        self.describe_categorical()

        print("\n===== MISSING VALUES — COUNT =====")
        self.missing_counts()

        print("\n===== MISSING VALUES — PERCENTAGE =====")
        self.missing_percentage()

        print("\n===== DUPLICATED ROWS =====")
        self.duplicate_count()

        print("\n===== CATEGORICAL DISTRIBUTIONS =====")
        self.plot_categorical()

        print("\n===== ALL NUMERIC DISTRIBUTIONS =====")
        self.plot_numeric_distributions()

        print("\n===== CONTINUOUS DISTRIBUTIONS =====")
        self.plot_continuous_distributions()

        print("\n===== DISCRETE VALUE COUNTS =====")
        self.value_counts_discrete()

        print("\n===== DISCRETE DISTRIBUTIONS =====")
        self.plot_discrete()

        print("\n===== Credit_History vs Loan_Status =====")
        if 'Credit_History' in self.df.columns and 'Loan_Status' in self.df.columns:
            self.plot_countplot_by_target('Credit_History', 'Loan_Status')

        print("\n===== Job_Type vs Loan_Status =====")
        if 'Job_Type' in self.df.columns and 'Loan_Status' in self.df.columns:
            self.plot_countplot_by_target('Job_Type', 'Loan_Status')

        print("\n===== Married vs Loan_Status =====")
        if 'Married' in self.df.columns and 'Loan_Status' in self.df.columns:
            self.plot_countplot_by_target('Married', 'Loan_Status')

        print("\n===== Credit_History vs Loan_Status by Education_Level =====")
        if all(c in self.df.columns for c in ['Credit_History', 'Loan_Status', 'Education_Level']):
            self.plot_barplot_by_hue('Credit_History', 'Loan_Status', 'Education_Level')

        print("\n===== Income by Loan_Status =====")
        if 'Income' in self.df.columns and 'Loan_Status' in self.df.columns:
            self.plot_boxplot('Loan_Status', 'Income')

        print("\n===== CORRELATION HEATMAP =====")
        self.plot_correlation_heatmap()

        print("\n[EDA] Full pipeline complete.")


# ------------------------------------------------------------------
# Usage example
# ------------------------------------------------------------------
if __name__ == "__main__":
    eda = EDA(csv_path="loan_data_final .csv")
    eda.run_full_eda()
