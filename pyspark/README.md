# mlflow-examples - pyspark

## Overview

* PySpark Spark ML Decision Tree Classification example
* Saves model in SparkML and MLeap format
* Source: [train.py](train.py) and [predict.py](predict.py)
* Experiment name: pypark
* Data: [../data/wine-quality-white.csv](../data/wine-quality-white.csv)

## Train

### Unmanaged without mlflow run

To run with standard main function
```
spark-submit --master local[2] \
  --packages com.databricks:spark-avro_2.11:3.0.1,ml.combust.mleap:mleap-spark_2.11:0.12.0 \
  train.py --max_depth 16 --max_bins 32 
```

### Using mlflow run

These runs use the [MLproject](MLproject) file. For more details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Note that `mlflow run` ignores the `set_experiment()` function so you must specify the experiment with the  `--experiment-id` argument.

**mlflow run local**
```
mlflow run . \
  -P max_depth=3 -P max_bins=24 \
  --experiment-name=pyspark
```

**mlflow run github**
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/pyspark \
   -P max_depth=3 -P max_bins=24 \
  --experiment-name=pyspark
```

## Predict

See [predict.py](predict.py).

```
run_id=7b951173284249f7a3b27746450ac7b0
spark-submit --master local[2] predict.py $run_id
```

```
Spark ML predictions
+-----------------+-------+--------------------------------------------------------+
|prediction       |quality|features                                                |
+-----------------+-------+--------------------------------------------------------+
|5.470588235294118|6      |[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]|
|5.470588235294118|6      |[6.3,0.3,0.34,1.6,0.049,14.0,132.0,0.994,3.3,0.49,9.5]  |
|5.769607843137255|6      |[8.1,0.28,0.4,6.9,0.05,30.0,97.0,0.9951,3.26,0.44,10.1] |
|5.877049180327869|6      |[7.2,0.23,0.32,8.5,0.058,47.0,186.0,0.9956,3.19,0.4,9.9]|
|5.877049180327869|6      |[7.2,0.23,0.32,8.5,0.058,47.0,186.0,0.9956,3.19,0.4,9.9]|
+-----------------+-------+--------------------------------------------------------+
only showing top 5 rows

model_uri: runs:/ffd36a96dd204ac38a58a00c94390649/mleap-model
MLeap ML predictions
+-----------------+-------+--------------------------------------------------------+
. . . 
```

