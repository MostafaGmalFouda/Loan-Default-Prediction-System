from pyspark.sql import DataFrame
class AppState:

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
            cls._inst._reset()
        return cls._inst

    def _reset(self):
        # DataFrames
        self.df_original:  DataFrame = None
        self.df_processed: DataFrame = None
        self.vector_df:    DataFrame = None
        self.csv_path:     str       = None

        # Original columns meta
        self.orig_columns:      list = []
        self.orig_num_cols:     list = []
        self.orig_cat_cols:     list = []
        self.orig_num_nan_cols: list = []
        self.orig_cat_nan_cols: list = []

        # Processed columns meta
        self.proc_columns:      list = []
        self.proc_num_cols:     list = []
        self.proc_cat_cols:     list = []
        self.proc_num_nan_cols: list = []
        self.proc_cat_nan_cols: list = []

        # Columns still with NaN after processing
        self.processed_num_nan_cols: list = []
        self.processed_cat_nan_cols: list = []
        self.processed_num_cols:     list = []

        # Processing tracking
        self.applied_steps:   list = []
        self.has_duplicates:  bool = False
        self.vector_ready:    bool = False
        self.processing_done: bool = False

        # ML
        self.feature_cols: list = []
        self.target_col:   str  = None
        self.data_loaded:  bool = False

        # Saved models
        self.current_model_path: str = None
        self.current_model_id:   str = None

    def reset(self):
        self._reset()

    # Get active DataFrame
    def get_active_df(self) -> DataFrame:
        return self.df_processed if self.df_processed is not None else self.df_original

    # Update original columns meta
    def update_orig_meta(self, meta: dict):
        self.orig_columns      = meta.get("columns",      [])
        self.orig_num_cols     = meta.get("num_cols",      [])
        self.orig_cat_cols     = meta.get("cat_cols",      [])
        self.orig_num_nan_cols = meta.get("num_nan_cols",  [])
        self.orig_cat_nan_cols = meta.get("cat_nan_cols",  [])
        self.proc_columns      = self.orig_columns[:]
        self.proc_num_cols     = self.orig_num_cols[:]
        self.proc_cat_cols     = self.orig_cat_cols[:]
        self.proc_num_nan_cols = self.orig_num_nan_cols[:]
        self.proc_cat_nan_cols = self.orig_cat_nan_cols[:]
        self.processed_num_nan_cols = self.orig_num_nan_cols[:]
        self.processed_cat_nan_cols = self.orig_cat_nan_cols[:]
        self.processed_num_cols     = self.orig_num_cols[:]

    # Update processed columns meta
    def update_proc_meta(self, meta: dict):
        self.proc_columns      = meta.get("columns",      [])
        self.proc_num_cols     = meta.get("num_cols",      [])
        self.proc_cat_cols     = meta.get("cat_cols",      [])
        self.proc_num_nan_cols = meta.get("num_nan_cols",  [])
        self.proc_cat_nan_cols = meta.get("cat_nan_cols",  [])

    # Update columns still with NaN after fill
    def update_processed_nan_cols(self, num_nan_cols: list, cat_nan_cols: list, num_cols: list):
        self.processed_num_nan_cols = num_nan_cols
        self.processed_cat_nan_cols = cat_nan_cols
        self.processed_num_cols     = num_cols

    # Backward compatibility
    def update_col_meta(self, meta: dict):
        self.update_proc_meta(meta)

    # Build payload for frontend
    def to_payload(self, use_processed: bool = False) -> dict:
        if use_processed and self.processing_done:
            cols     = self.proc_columns
            num_cols = self.proc_num_cols
            cat_cols = self.proc_cat_cols
            num_nan  = self.proc_num_nan_cols
            cat_nan  = self.proc_cat_nan_cols
        else:
            cols     = self.orig_columns
            num_cols = self.orig_num_cols
            cat_cols = self.orig_cat_cols
            num_nan  = self.orig_num_nan_cols
            cat_nan  = self.orig_cat_nan_cols

        return {
            "columns":         self.proc_columns if self.processing_done else self.orig_columns,
            "num_cols":        num_cols,
            "cat_cols":        cat_cols,
            "num_nan_cols":    num_nan,
            "cat_nan_cols":    cat_nan,
            "orig_columns":    self.orig_columns,
            "orig_cat_cols":   self.orig_cat_cols,
            "orig_num_cols":   self.orig_num_cols,
            "proc_columns":    self.proc_columns,
            "proc_cat_cols":   self.proc_cat_cols,
            "proc_num_cols":   self.proc_num_cols,
            "processed_num_nan_cols": self.processed_num_nan_cols,
            "processed_cat_nan_cols": self.processed_cat_nan_cols,
            "processed_num_cols":     self.processed_num_cols,
            "applied_steps":   self.applied_steps,
            "has_duplicates":  self.has_duplicates,
            "vector_ready":    self.vector_ready,
            "processing_done": self.processing_done,
            "data_loaded":     self.data_loaded,
            "feature_cols":    self.feature_cols,
            "target_col":      self.target_col,
        }