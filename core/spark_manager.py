from pyspark.sql import SparkSession
from config import SPARK_APP_NAME, SPARK_MASTER, SPARK_DRIVER_MEM, SPARK_EXEC_MEM


class SparkManager:

    _instance: SparkSession = None

    @classmethod
    def get_session(cls) -> SparkSession:
        if cls._instance is None or cls._instance._sc._jsc.sc().isStopped():
            cls._instance = (
                SparkSession.builder
                .appName(SPARK_APP_NAME)
                .master(SPARK_MASTER)
                .config("spark.driver.memory",   SPARK_DRIVER_MEM)
                .config("spark.executor.memory", SPARK_EXEC_MEM)
                .config("spark.sql.shuffle.partitions", "8")
                .config("spark.default.parallelism",    "8")
                .config("spark.ui.showConsoleProgress", "false")
                .getOrCreate()
            )
            cls._instance.sparkContext.setLogLevel("ERROR")
        return cls._instance

    @classmethod
    def stop(cls):
        if cls._instance:
            cls._instance.stop()
            cls._instance = None