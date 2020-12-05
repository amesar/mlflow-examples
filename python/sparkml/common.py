
default_data_path = "../../data/train/wine-quality-white.csv"

def read_data(spark, data_path):
    if data_path.startswith("http"):
        import pandas as pd
        pdf = pd.read_csv(data_path)
        return spark.createDataFrame(pdf)
    else:
        return spark.read.csv(data_path, header="true", inferSchema="true")

colLabel = "quality"
colPrediction = "prediction"
colFeatures = "features"

def show_versions(spark):
    import platform
    import mlflow
    import pyspark
    print("Versions:")
    print("  Operating System:",platform.version()+" - "+platform.release())
    print("  Spark Version:", spark.version)
    print("  PySpark Version:", pyspark.__version__)
    print("  MLflow Version:", mlflow.__version__)
    print("  MLflow Tracking URI:", mlflow.tracking.get_tracking_uri())
