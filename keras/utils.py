
from keras.datasets import mnist
from keras.utils import to_categorical

def build_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Data:")
    print("  x_train.shape:",x_train.shape)
    print("  y_train.shape:",y_train.shape)
    print("  y_train:",y_train)
    print("  x_test.shape:",x_test.shape)
    print("  y_test.shape:",y_test.shape)
    print("  y_test:",y_test)

    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255

    x_test = x_test.reshape((10000, 28 * 28))
    x_test = x_test.astype('float32') / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test
