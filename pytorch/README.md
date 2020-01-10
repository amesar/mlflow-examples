# mlflow-examples - pytorch

## Overview
* Pytorch train and predict.
* Saves model as pytorch and ONNX flavor (optional).
* Simple synthetic dataset.

## Training

Source: [train.py](train.py).

To run with user logging (no autologging).
```
python main.py --experiment_name pytorch --epochs 2 
```

To log a model as ONNX flavor under the artifact path `onnx-model`.
```
python main.py --experiment_name pytorch --epochs 2 --log_as_onnx
```

## Scoring

Source: [predict.py](predict.py).
Scores using following flavors:
* pytorch
* pyfunc/pytorch
* onnx
* pyfunc/onnx

```
python predict.py --run_id 7e674524514846799310c41f10d6b99d
```

```
Torch Version: 1.0.1
MLflow Version: 1.5.0
Tracking URI: http://localhost:5000
Arguments:
  run_id: f6af844e4e0b47fea22f8edf96d37d2d
data.type: <class 'torch.Tensor'>
data.shape: torch.Size([3, 1])

==== pytorch.load_model

model_uri: runs:/f6af844e4e0b47fea22f8edf96d37d2d/pytorch-model
model.type: <class '__main__.Model'>
outputs.type: <class 'torch.Tensor'>
outputs:
           0
0  1.617840
1  2.601871
2  3.585900

==== pyfunc.load_model - pytorch

model.type: <class 'mlflow.pytorch._PyTorchWrapper'>
outputs.type: <class 'pandas.core.frame.DataFrame'>
outputs:
           0
0  1.617840
1  2.601871
2  3.585900

==== onnx.load_model - onnx

model.type: <class 'onnx.onnx_ONNX_REL_1_6_ml_pb2.ModelProto'>
outputs.type: <class 'numpy.ndarray'>
outputs:
           0
0  1.617840
1  2.601871
2  3.585900

==== pyfunc.load_model - onnx

model_uri: runs:/f6af844e4e0b47fea22f8edf96d37d2d/onnx-model
model.type: <class 'mlflow.onnx._OnnxModelWrapper'>
outputs.type: <class 'pandas.core.frame.DataFrame'>
outputs:
           3
0  1.617840
1  2.601871
2  3.585900
```
