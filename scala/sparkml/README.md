# mlflow-examples - scala-spark

Scala examples using the MLflow Java client:
* Hello World - Simple MLflow example with no training.
* Spark - Uses wine quality dataset, saves and predicts SparkML and MLeap model formats.
  * Spark ML - DecisionTreeRegressor 
  * XGBoost4j Spark - XGBoostRegressor

## Setup

You must install Python MLflow: `pip install mlflow==1.14.0`.

The build supports either Scala 2.11 wth Spark 2.x or Scala 2.12 with Spark 3.x.

The Maven pom.xml two profiles are:
* spark-3x
  * Scala 2.12
  * Spark 3.0.0
* spark-2x
  * Scala 2.11
  * Spark 2.4.5 

## Build

Default profile is spark-3x.
```
mvn clean package
```

To build explicitly with the Spark 2.x profile.
```
mvn clean package -P spark-2x
```


## Hello World Sample
### Run
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.hello.HelloWorld \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  http://localhost:5000
```
```
Experiment name: scala_HelloWorld
Experiment ID: 3
Run ID: 81cc7941adae4860899ad5449df52802
```

### Source

Source: [HelloWorld.scala](src/main/scala/org/andre/mlflow/examples/hello/HelloWorld.scala).
```
// Create client
val trackingUri = args(0)
val mlflowClient = new MlflowClient(trackingUri)

// Create or get existing experiment
val expName = "scala/HelloWorld"
val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)
println("Experiment name: "+expName)
println("Experiment ID: "+expId)

// Create run
val runInfo = mlflowClient.createRun(expId);
val runId = runInfo.getRunUuid()

// Log params and metrics
mlflowClient.logParam(runId, "p1","hi")
mlflowClient.logMetric(runId, "m1",0.123F)

// Close run
mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
```

## Spark ML Wine Quality DecisionTreeRegressor Sample

Sample demonstrating:
*  Trains a model with DecisionTreeRegressor algorithm
*  Logs the model in Spark ML and MLeap flavors (MLeap bundle)
*  Predicts model with Spark ML flavor and MLeap flavor as SparkBundle
*  Predicts model with MLeap flavor as MLeapBundle (no Spark dependencies)

### Train

Saves model as Spark ML and MLeap artifacts in MLflow.


#### Source

Source: [TrainWine.scala](src/main/scala/org/andre/mlflow/examples/wine/sparkml/TrainWine.scala).

### Run against local Spark and local MLflow tracking server

```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.wine.sparkml.TrainWine \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --experimentName scala_sparkml \
  --dataPath ../../data/wine-quality-white.csv \
  --modelPath model_sample --maxDepth 5 --maxBins 5
```

### Run against local Spark and Databricks hosted tracking server

```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.wine.sparkml.TrainWine \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri https://acme.cloud.databricks.com --token MY_TOKEN \
  --experimentName scala_sparkml \
  --dataPath ../../data/wine-quality-white.csv \
  --modelPath model_sample --maxDepth 5 --maxBins 5
```

### Run against Databricks Cluster with Databricks REST API

You can also run your jar in a Databricks cluster with the standard Databricks REST API run endpoints.
See [runs submit](https://docs.databricks.com/api/latest/jobs.html#runs-submit), [run now](https://docs.databricks.com/api/latest/jobs.html#run-now) and [spark_jar_task](https://docs.databricks.com/api/latest/jobs.html#jobssparkjartask).
In this example we showcase runs_submit.

#### Setup

Upload the data file and jar to your Databricks cluster.
```
databricks fs cp ../../data/wine-quality-white.csv \
  dbfs:/tmp/jobs/spark-scala-example/wine-quality-white.csv

databricks fs cp target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  dbfs:/tmp/jobs/spark-scala-example/mlflow-spark-examples-1.0-SNAPSHOT.jar
```

Here is a snippet from
[run_submit_new_cluster.json](run_submit_new_cluster.json) or
[run_submit_existing_cluster.json](run_submit_existing_cluster.json).
```
  "libraries": [
    { "pypi": { "package": "mlflow" } },
    { "jar": "dbfs:/tmp/jobs/spark-scala-example/mlflow-spark-examples-1.0-SNAPSHOT.jar" }
  ],
  "spark_jar_task": {
    "main_class_name": "org.andre.mlflow.examples.wine.sparkml.TrainWine",
    "parameters": [ 
      "--dataPath",  "dbfs:/tmp/jobs/spark-scala-example/wine-quality-white.csv",
      "--modelPath", "/dbfs/tmp/jobs/spark-scala-example/models",
      "--runOrigin", "run_submit_new_cluster.json"
    ]
  }
```

#### Run with new cluster

Create [run_submit_new_cluster.json](run_submit_new_cluster.json) and launch the run.
```
curl -X POST -H "Authorization: Bearer MY_TOKEN" \
  -d @run_submit_new_cluster.json  \
  https://acme.cloud.databricks.com/api/2.0/jobs/runs/submit
```

#### Run with existing cluster

Every time you build a new jar, you need to upload (as described above) it to DBFS and restart the cluster.
```
databricks clusters restart --cluster-id 0113-005848-about166
```

Create [run_submit_existing_cluster.json](run_submit_existing_cluster.json) and launch the run.
```
curl -X POST -H "Authorization: Bearer MY_TOKEN" \
  -d @run_submit_existing_cluster.json  \
  https://acme.cloud.databricks.com/api/2.0/jobs/runs/submit
```

#### Run jar from Databricks notebook

Create a notebook with the following cell. Attach it to the existing cluster described above.
```
import org.andre.mlflow.examples.wine.sparkml.TrainWine
val dataPath = "dbfs:/tmp/jobs/spark-scala-example/wine-quality-white.csv"
val modelPath = "/dbfs/tmp/jobs/spark-scala-example/models"
val runOrigin = "run_from_jar_Notebook"
TrainWine.train(client, experimentId, modelPath, 5, 32, "my_run", dataPath)
```

### Predict as Spark ML and MLeap SparkBundle

Source: [PredictAsSpark.scala](src/main/scala/org/andre/mlflow/examples/wine/PredictAsSpark.scala).

Predicts from Spark ML and MLeap models.
Reads from artifacts `spark-model` and `mleap-model/mleap/model`.

#### Run
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.PredictAsSpark \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --dataPath ../data/wine-quality-white.csv \
  --runId 3e422c4736a34046a74795384741ac33
```

```
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       0.0|  0.0|(692,[127,128,129...|
|       1.0|  1.0|(692,[158,159,160...|
|       1.0|  1.0|(692,[124,125,126...|
|       1.0|  1.0|(692,[152,153,154...|
+----------+-----+--------------------+
```

### Predict as MLeapBundle with no Spark

Source: [PredictAsMLeapBundle.scala](src/main/scala/org/andre/mlflow/examples/wine/PredictAsMLeapBundle.scala).

No Spark dependencies involved.
Reads from artifacts `mleap-model/schema.json` and `mleap-model/mleap/model`.

#### Run
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.PredictAsMLeapBundle \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --dataPath ../data/wine-quality-white.csv \
  --runId 3e422c4736a34046a74795384741ac33
```

```
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       0.0|  0.0|(692,[127,128,129...|
|       1.0|  1.0|(692,[158,159,160...|
|       1.0|  1.0|(692,[124,125,126...|
|       1.0|  1.0|(692,[152,153,154...|
+----------+-----+--------------------+
```

## XGBoost4j-Spark Wine Quality XGBoostRegressor Sample

### Train

Source: [Train.scala](src/main/scala/org/andre/mlflow/examples/wine/xgboost/Train.scala).
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.wine.xgboost.Train \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --experimentName scala_xgboost \
  --dataPath ../data/wine-quality-white.csv \
  --modelPath model_sample \
  --maxDepth 5 \
  --objective reg:squarederror
```

### Predict

Source: [Predict.scala](src/main/scala/org/andre/mlflow/examples/wine/xgboost/Predict.scala).
```
spark-submit --master local[2] \
  --class org.andre.mlflow.examples.wine.xgboost.Predict \
  target/mlflow-spark-examples-1.0-SNAPSHOT.jar \
  --trackingUri http://localhost:5000 \
  --experimentName scala_xgboost \
  --dataPath ../../data/wine-quality-white.csv  \
  --runId 3e422c4736a34046a74795384741ac33
```
