
from keras.datasets import mnist
from keras.utils import to_categorical

def reshape(x, n):
    x = x.reshape((n, 28 * 28))
    x = x.astype('float32') / 255
    return x

def build_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Data:")
    print("  x_train.shape:", x_train.shape)
    print("  y_train.shape:", y_train.shape)
    print("  x_test.shape:", x_test.shape)
    print("  y_test.shape:", y_test.shape)

    x_train = reshape(x_train, 60000)
    x_test = reshape(x_test, 10000)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print("Data after reshape:")
    print("  x_test.shape:", x_test.shape)
    print("  y_test.shape:", y_test.shape)

    return x_train, y_train, x_test, y_test
