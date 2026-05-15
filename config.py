import os

# Spark Settings
SPARK_APP_NAME   = "ML_Dashboard"
SPARK_MASTER     = "local[*]"
SPARK_DRIVER_MEM = "4g"
SPARK_EXEC_MEM   = "4g"

# App Settings
UPLOAD_FOLDER    = "/tmp/ml_dashboard_uploads"
MODELS_FOLDER    = os.path.join(os.path.dirname(__file__), "models_store")
SAMPLE_ROWS      = 500
MAX_FILE_MB      = 500
MAX_CONTENT_LEN  = MAX_FILE_MB * 1024 * 1024

# Hero Image Path (set to None to use default SVG)
HERO_IMAGE_PATH  = None

# ML Settings
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
CV_FOLDS         = 3

# Create Required Folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)