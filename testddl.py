from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('Elephas_App1').setMaster('local[6]')
#sc = SparkContext(conf=conf)
#sc.stop()
sc = SparkContext.getOrCreate(conf=conf)
import os
os.environ['TF_KERAS'] = '1'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import tensorflow as tf

model = Sequential()
model.add(Dense(128,input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate = 0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

epochs = 1

from elephas.utils.rdd_utils import to_simple_rdd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("length =  ( ", len(x_train), ", ", len(y_train), " )")
print("shape of the dataset = ", tf.shape(y_train))

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

nb_classes = 10
# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

rdd = to_simple_rdd(sc, x_train, y_train)
print("rdd = ", rdd)

from elephas.spark_model import SparkModel
spark_model = SparkModel(model, frequency = 'epoch', mode='asynchronous', num_workers=2)
spark_model.fit(rdd, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
score = spark_model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', score)
