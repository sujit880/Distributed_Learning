#this is for distributed scenario
import random
import os
import gym
import numpy as np
from collections import deque
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
print(state_size)
action_size = env.action_space.n
print(action_size)
batch_size = 32
n_episodes = 1001
output_dir = '/home/sujit/Github_Project/output'
class DQNAgent:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=2000)

    self.gamma = 0.95

    self.epsilon = 1.0
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.01

    self.learning_rate = 0.001

    self.model = self._build_model()

  def _build_model(self):
    model = Sequential()

    model.add(Dense(24,input_dim = self.state_size, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(self.action_size, activation = 'linear'))

    model.compile(loss='mse', optimizer=Adam(lr = self.learning_rate))

    return model
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    print("act_values : ", act_values)
    print("act_values[0] : ", np.argmax(act_values[0]))
    return np.argmax(act_values[0])
  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

      target_f = self.model.predict(state)
      print("target_f : ", target_f)
      print("pre target_f[0][action] : ", target_f[0][action])
      target_f[0][action] = target
      print("target_f[0][action] : ", target_f[0][action])

      self.model.fit(state, target_f, epochs=1, verbose=0)
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)

agent = DQNAgent(state_size, action_size)
done = False
for e in range(n_episodes):
  state = env.reset()
  print("initial state :",state)
  state = np.reshape(state, [1, state_size])
  for time in range(5000):
    #env.render()
    action = agent.act(state)
    print("action : ", action)
    next_state, reward, done, _ = env.step(action)
    print("next state :",next_state)
    reward = reward if not done else -10
    print("reward : ", reward)
    next_state = np.reshape(next_state, [1, state_size])
    agent.remember(state, action, reward, next_state, done)
    state = next_state
    if done:
      print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes, time, agent.epsilon))
      break
  if len(agent.memory) > batch_size:
    agent.replay(batch_size)
  if e%50 == 0:
    agent.save(output_dir + "/weights_"+ "{:04d}".format(e)+".hdf5")


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
