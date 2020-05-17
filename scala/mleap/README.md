# mlflow-examples - MLeap

Example shows how to download an MLeap model from MLflow and score it with MLeap runtime with no Spark dependencies.

## Setup and Build

* Install Python MLflow library: `pip install mlflow==1.8.0`
* Build the jar: `mvn clean package`

## Assumptions

The expected run artifacts hierarchy is shown below and is produced by the `python/sparkml` and `scala/sparkml` trainers.
`schema.json` is emitted byt the trainers and is used to create an input LeapFrame from the data.

```
+-mleap-model/
| +-schema.json
| +-mleap/
| | +-model/
| |   +-root/
| |   +-bundle.json
```

## Run predictions

```
scala -cp target/mlflow-mleap-examples-1.0-SNAPSHOT.jar \
  org.andre.mlflow.examples.wine.PredictWine \
  --dataPath ../../data/train/wine-quality-white.csv \
  --runId 7b951173284249f7a3b27746450ac7b0
```

```
Prediction sum: 28767.070

Prediction Counts:
  prediction    count
       6.063      731
       5.471      583
       6.770      566
       5.169      559
       5.877      517
       . . .

4898 Predictions:
    5.471
    5.471
    5.770
    5.877
    5.877
    . . .

```

