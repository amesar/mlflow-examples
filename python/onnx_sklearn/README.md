# mlflow-examples - ONNX/sklearn 

## Overview
* First pass exploration of [ONNX](https://onnx.ai/) and MLflow.
* Converts Sklearn DecisionTreeRegressor to ONNX model
* Batch predictions with ONNX and pyfunc flavors
* Real-time predictions with web server

## Setup

```
pip install onnx==1.4.1
pip install onnxruntime==0.3.0
pip install skl2onnx==1.6.0
pip install tf2onnx==1.5.6
```

## Training

Source: [train.py](train.py).

Run the standard main function from the command-line.
```
python train.py --experiment_name onnx_sklearn --max_depth 2 --max_leaf_nodes 32
```

##  Scoring

###  Batch Scoring

#### Score with ONNX flavor

```
python onnx_predict.py runs:/7e674524514846799310c41f10d6b99d/onnx-model
```

```
model.type: <class 'onnx.onnx_ONNX_REL_1_4_ml_pb2.ModelProto'>
data.shape: (4898, 11)
predictions.type: <class 'numpy.ndarray'>
predictions.shape: (4898, 1)

+--------------+
|   prediction |
|--------------|
|        5.941 |
|        6     |
|        5.222 |
|        6.105 |
|        6.105 |
+--------------+
```

#### Score with pyfunc

```
python pyfunc_predict.py runs:/7e674524514846799310c41f10d6b99d/onnx-model
```
```
model: <class 'mlflow.onnx._OnnxModelWrapper'>
data.shape: (4898, 11)
predictions.type: <class 'pandas.core.frame.DataFrame'>
predictions.shape: (4898, 1)

+------------+
|   variable |
|------------|
|      5.941 |
|      6     |
|      5.222 |
|      6.105 |
|      6.105 |
+------------+
```

###  Realtime Scoring

In one window launch the scoring server.
```
mlflow models serve --port 5001 --model-uri runs:/7e674524514846799310c41f10d6b99d/onnx-model
```

In another window submit data for scoring.
```
curl -X POST -H "Content-Type:application/json" \
  -d @../../data/score/wine-quality-float.json \
  http://localhost:5001/invocations
```

```
[{"variable": 5.941176414489746}, {"variable": 6.0}, {"variable": 5.222222328186035}]
```

#### ONNX Issues

The response is not the standard MLflow scoring server response which is an array of scores.

The ONNX runtime also expects all input data to be floats. It is unable to promote integers to floats.
If you submit "mixed" data you will get the following error.
```
INVALID_ARGUMENT : Unexpected input data type. Actual: (N11onnxruntime11NonOnnxTypeIdEE) , expected: (N11onnxruntime11NonOnnxTypeIfEE)\n"}
```
Full error message.
```
{
  "error_code": "BAD_REQUEST",
  "message": "Encountered an unexpected error while evaluating the model. Verify that the serialized input Dataframe is compatible with the model for inference.",
  "stack_trace": "Traceback (most recent call last):\n  File \"/Users/ander/miniconda3/envs/mlflow-eb96057dbc844afacb91aed3bab4d9bc2faeb3c4/lib/python3.6/site-packages/mlflow/pyfunc/scoring_server/__init__.py\", line 196, in transformation\n    raw_predictions = model.predict(data)\n  File \"/Users/ander/miniconda3/envs/mlflow-eb96057dbc844afacb91aed3bab4d9bc2faeb3c4/lib/python3.6/site-packages/mlflow/onnx.py\", line 174, in predict\n    predicted = self.rt.run(self.output_names, feed_dict)\n  File \"/Users/ander/miniconda3/envs/mlflow-eb96057dbc844afacb91aed3bab4d9bc2faeb3c4/lib/python3.6/site-packages/onnxruntime/capi/session.py\", line 72, in run\n    return self._sess.run(output_names, input_feed, run_options)\nRuntimeError: Method run failed due to: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (N11onnxruntime11NonOnnxTypeIdEE) , expected: (N11onnxruntime11NonOnnxTypeIfEE)\n"
}
```


