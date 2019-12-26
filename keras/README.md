# mlflow-examples - keras

## Overview
* Keras/TensorFlow train and predict.
* Saves model as keras flavor.
* MNIST dataset.
* Setup: `pip install tensorflow==1.15.0`.

## Training

Source: [train.py](train.py).
Run the standard main function from the command-line.
```
python main.py --experiment_name sklearn --epochs 3 --batch_size 128
```

## Predictions

Source: [predict.py](predict.py).
You can either use a `runs` or `models` URI.
```
python predict.py runs:/7e674524514846799310c41f10d6b99d/keras-model
```

```
python predict.py models:/keras-wine/production
```

```
predictions: [7 2 1 ... 4 5 6]
```


