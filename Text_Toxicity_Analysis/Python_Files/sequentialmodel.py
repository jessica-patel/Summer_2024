from tensorflow.keras.models import Sequential  # API to group linear stack of layers into a model
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, \
    Embedding
import matplotlib.pyplot as plt
import pandas as pd


class SequentialModel():
    def __init__(self, epoch_values, loss_values, val_loss_values,
                 prediction_arrays, precision_values, recall_values,
                 accuracy_values, val_set, train_set, chosen_epoch):

        self.epoch_values = epoch_values
        self.loss_values = loss_values
        self.val_loss_values = val_loss_values
        self.prediction_arrays = prediction_arrays
        self.precision_values = precision_values
        self.recall_values = recall_values
        self.accuracy_values = accuracy_values
        self.val_set = val_set
        self.train_set = train_set
        self.chosen_epoch = chosen_epoch

        self.model = Sequential()
        self.max_features = 200000

    def add_layers(self, model):

        # Create the embedding laye
        # number of words + 1, 1 embedding per word, embeddings are 32 values in len
        model.add(Embedding(self.max_features+1, 32))
        # Bidirectional LSTM Layer
        # LSTM needs GPU acceleration of tanh (dictated by TensorFlow)
        model.add(Bidirectional(LSTM(32, activation='tanh')))
        # Feature extractor Fully connected layers
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        # Final layer
        # output 6 layers for 6 labels, sigmoid converts relu into 0 to 1 values
        model.add(Dense(6, activation='sigmoid'))

        return None

    def run_model(self, model):
        ####
        # BinaryCrossentropy is reducing loss for binary outputs
        model.compile(loss='BinaryCrossentropy', optimizer='Adam')
        # model.summary()
        # epoch = 10, how long to train for, how many passes through validation data
        history = model.fit(self.train_set, epochs=self.chosen_epoch,
                            validation_data=self.val_set)
        self.epoch_values.append(self.chosen_epoch)
        self.loss_values.append(history.history['loss'])
        self.val_loss_values.append(history.history['val_loss'])

        return history
        ####

    def iplot(self, history):
        # Comparing loss and val_loss over epochs
        plt.figure(figsize=(8, 5))
        pd.DataFrame(history.history).plot()
        plt.title("For 10 epochs")
        plt.show()

        return None
