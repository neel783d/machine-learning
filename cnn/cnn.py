import numpy as np
from tensorflow import keras
from tensorflow.keras import Model, layers

num_class = 10
input_shape = (28, 28, 1)


def preprocess(x, y):
  x = x.astype("float32") / 255.
  x = np.expand_dims(x, -1)
  y = keras.utils.to_categorical(y, num_class)
  return x, y


def prepare_data():
  # Loading Data
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  print(f'shapes: \ntrain: {x_train.shape}, {y_train.shape}'
        f'\ntest: {x_test.shape}, {y_test.shape}'
        f'\ntype: {type(x_train)}')

  # preprocess
  x_train, y_train = preprocess(x_train, y_train)
  x_test, y_test = preprocess(x_test, y_test)
  print(f'\npreprocessed shapes: \ntrain: {x_train.shape}, {y_train.shape}'
        f'\ntest: {x_test.shape}, {y_test.shape}'
        f'\ntype: {type(x_train)}')
  return [(x_train, y_train), (x_test, y_test)]


class MyCnn:
  def __init__(
      self,
      loss="categorical_crossentropy",
      optimizer="adam",
      metrics=["accuracy"],
      batch_size=128,
      epochs=5,
      validation_split=0.1
  ):
    self.loss = loss
    self.optimizer = optimizer
    self.metrics = metrics
    self.epochs = epochs
    self.batch_size = batch_size
    self.validation_split = validation_split

    # Parameters
    self.model = None

  def build_model(self):
    self.model = self.model_layers()
    self.model.compile(loss=self.loss,
                       optimizer=self.optimizer,
                       metrics=self.metrics)

  @staticmethod
  def model_layers() -> Model:
    model = keras.Sequential(
        [
          layers.Input(shape=input_shape),
          layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
          layers.MaxPool2D(pool_size=(2, 2)),
          layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
          layers.MaxPool2D(pool_size=(2, 2)),
          layers.Flatten(),
          layers.Dropout(0.5),
          layers.Dense(num_class, activation="softmax")
        ]
    )
    model.summary()
    return model

  def train(self, x_train, y_train):
    self.model.fit(
        x_train,
        y_train,
        batch_size=self.batch_size,
        epochs=self.epochs,
        validation_split=self.validation_split)

  def evaluate(self, x_test, y_test):
    score = self.model.evaluate(x_test, y_test, verbose=0)
    return score


def main():
  # Preprocess data
  [train_data, test_data] = prepare_data()

  # Model
  cnn = MyCnn()

  # Compiling Building Model
  cnn.build_model()

  # Training
  cnn.train(*train_data)

  # Score
  score = cnn.evaluate(*test_data)
  print("Test loss:", score[0])
  print("Test accuracy:", score[1])


if __name__ == '__main__':
  main()
