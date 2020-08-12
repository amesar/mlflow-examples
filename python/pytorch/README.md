# mlflow-examples - pytorch

## Overview

Pytorch train and predict examples.

**MNIST Model**
* Saves model as pytorch flavor and ONNX flavor. 
* Pyfunc and ONNX scoring are not working yet.

**Simple Model**
* Simple synthetic dataset.
* Saves model as pytorch and ONNX flavor. 
* Pyfunc and ONNX scoring work.

##  MNIST Model

### Training

Source: [train_mnist.py](train_mnist.py).

```
python -u train_mnist.py --experiment_name pytorch_mnist --epochs 2 
```

```
Train Epoch: 1 [0/60000 (0%)]       Loss: 4.178357
Train Epoch: 1 [6400/60000 (11%)]   Loss: 3.856662
. . .
Train Epoch: 1 [51200/60000 (85%)   Loss: 2.807601
Train Epoch: 1 [57600/60000 (96%)]  Loss: 2.476638

Test set: Average loss: 2.2248, Accuracy: 9240/10000 (92%)
```
		

### Scoring

Source: [predict_mnist.py](predict_mnist.py).

TODO: pyfunc and ONNX scoring.

```
usage: predict_mnist.py --run_id RUN_ID [--score_as_pyfunc]
                        [--score_as_onnx]

optional arguments:
  --run_id RUN_ID    Run ID
  --score_as_pyfunc  Score as Pyfunc
  --score_as_onnx    Score as ONNX
```

```
python -u predict_mnist --run_id 5bd65b5fcf264e0b814c4bad7863b8e3
```

```
data.type: <class 'torch.Tensor'>
data.shape: torch.Size([10000, 1, 28, 28])

model_uri: runs:/5bd65b5fcf264e0b814c4bad7863b8e3/pytorch-model
model.type: <class '__main__.Net'>
outputs.type: <class 'torch.Tensor'>
outputs.shape: (10000, 10)

+-----------+-----------+----------+-----------+-----------+-----------+----------+-----------+----------+----------+
|         0 |         1 |        2 |         3 |         4 |         5 |        6 |         7 |        8 |        9 |
|-----------+-----------+----------+-----------+-----------+-----------+----------+-----------+----------+----------|
| -17.1916  | -18.3416  | -15.3975 | -14.6749  | -15.8925  | -16.5027  | -22.9014 |  -6.83191 | -14.5595 | -11.8467 |
| -14.9626  |  -7.08042 | -11.5769 | -10.3589  | -11.7865  | -11.3741  | -13.1928 | -10.8136  | -10.3093 | -10.8471 |
| -11.1376  | -15.0779  | -14.5858 |  -9.66218 | -13.5638  |  -7.15294 | -13.4517 | -14.5061  | -11.859  | -12.0606 |
| -13.948   | -16.8267  | -12.7529 | -15.4706  |  -7.08425 | -13.3568  | -11.021  | -13.0811  | -14.517  | -10.5112 |
|  -7.04104 | -19.3001  | -14.1314 | -11.681   | -16.4093  | -10.9014  | -13.3053 | -13.7057  | -13.2701 | -14.1343 |
| -16.9276  |  -6.90742 | -11.4163 | -13.0818  | -13.1669  | -13.804   | -12.9    | -12.861   | -11.1579 | -13.7589 |
|  -6.90705 | -23.2038  | -13.7004 | -13.8274  | -21.3027  | -11.8081  | -16.8868 | -15.0107  | -17.0437 | -17.384  |
| -15.6461  |  -7.0249  | -12.5447 | -12.6407  | -13.0735  | -12.9183  | -12.4864 | -13.3597  | -11.33   | -12.5555 |
| -17.5718  |  -6.95123 | -12.4324 | -13.1505  | -14.0267  | -14.3623  | -13.9538 | -13.106   | -11.5788 | -13.4045 |
|  -6.92842 | -18.5234  | -11.5097 | -12.0417  | -19.6198  | -11.8229  | -14.8703 | -13.6569  | -13.1004 | -15.835  |
+-----------+-----------+----------+-----------+-----------+-----------+----------+-----------+----------+----------+
```


##  Simple Model

* Simple synthetic dataset

### Training

Source: [train_simple.py](train_simple.py).

To run with user logging (no autologging).
```
python -u train_simple.py --experiment_name pytorch_simple --epochs 2 
```

To log a model as ONNX flavor under the artifact path `onnx-model`.
```
python -u train_simple.py --experiment_name pytorch --epochs 2 --log_as_onnx
```

### Scoring

Source: [predict_simple.py](predict_simple.py).
Scores using following flavors:
* pytorch
* pyfunc/pytorch
* onnx
* pyfunc/onnx

```
python -u predict.py --run_id 7e674524514846799310c41f10d6b99d
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
