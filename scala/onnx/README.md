# mlflow-examples - ONNX Java Scoring

Score an ONNX model (that was created in Scikit-learn) in Java.

We download a ONNX model from MLflow and score it with Microsoft's [ONNX Runtime](https://github.com/microsoft/onnxruntime).

Files:
* [OnnxScorer.scala](src/main/scala/org/andre/onnx/examples/wine/OnnxScorer.scala)
* [ScoreFromFile.scala](src/main/scala/org/andre/onnx/examples/wine/ScoreFromFile.scala)
* [ScoreFromMLflow.scala](src/main/scala/org/andre/onnx/examples/wine/ScoreFromMLflow.scala)

## Setup and Build

* Install Python MLflow library: `pip install mlflow==1.7.0`
* Build the jar: `mvn clean package`

## Create Scikit-learn model as ONNX flavor

Train a scikit learn model and save it as an ONNX flavor. See [sklearn](../../python/sklearn).
```
cd ../../python/sklearn
python main.py --experiment_name sklearn --max_depth 2 --log_as_onnx
```

```
cd /opt/mlflow/server/mlruns/1/7b951173284249f7a3b27746450ac7b0
tree
├── onnx-model
│   ├── MLmodel
│   ├── conda.yaml
│   └── model.onnx
├── plot.png
└── sklearn-model
    ├── MLmodel
    ├── conda.yaml
    └── model.pkl
```


## Build ONNX Runtime Jar

Microsoft has a Java-based implementation part of its overall ONNX Runtime.
Since the jar is not yet in a public repository (i.e. Maven Central), you'll have to build it from scratch.
The build takes a mere 15 minutes.

Steps:
```
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime
build.sh --build_java --config RelWithDebInfo --build_shared_lib --parallel 
ls -l onnxruntime/java/build/libs/onnxruntime-1.2.0-all.jar
```

The actual runtime is built with C# and deployed as two native libraries in the jar. The Java layer delegates low-level scoring to these C# modules.
* ai/onnxruntime/native/libonnxruntime4j_jni.dylib
* ai/onnxruntime/native/libonnxruntime.dylib

They are of file type: `Mach-O 64-bit dynamically linked shared library x86_64`.

Then add the full path to your `onnxruntime-1.2.0-all.jar` in the [pom.xml](pom.xml).
```
<systemPath>CHANGE_ME/onnxruntime-1.2.0-all.jar</systemPath>
```

From the onnxruntime github repo see:
* [java/README](https://github.com/microsoft/onnxruntime/tree/master/java/README.md)
* [Javadoc](https://microsoft.github.io/onnxruntime/java/index.html)
* Example: [ScoreMNIST.java](https://github.com/microsoft/onnxruntime/blob/master/java/src/test/java/sample/ScoreMNIST.java)

## Run predictions

**Read from file**
```
scala -cp target/mlflow-onnx-scoring-1.0-SNAPSHOT.jar \
  org.andre.onnx.examples.wine.ScoreFromFile \
  --modelPath model.onnx 
```

**MLflow server**
```
scala -cp target/mlflow-onnx-scoring-1.0-SNAPSHOT.jar \
  org.andre.onnx.examples.wine.ScoreFromMLflow \
  --runId 7b951173284249f7a3b27746450ac7b0 \
  --artifactPath onnx-model/model.onnx
```

**Databricks MLflow server**
```
scala -cp target/mlflow-onnx-scoring-1.0-SNAPSHOT.jar \
  org.andre.onnx.examples.wine.ScoreFromMLflow \
  --runId 7b951173284249f7a3b27746450ac7b0 \
  --artifactPath onnx-model/model.onnx \
  --trackingUri https://acme.cloud.databricks.com \
  --token MY_TOKEN 
```

**Output**
```
Predictions:
  5.46875
  5.171642
. . .
  6.3539824
  6.26431
```
