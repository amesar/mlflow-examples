
default_data_path = "../../data/wine-quality-white.csv"

def read_data(spark, data_path):
    return spark.read.csv(data_path, header="true", inferSchema="true")

colLabel = "quality"
colPrediction = "prediction"
colFeatures = "features"
