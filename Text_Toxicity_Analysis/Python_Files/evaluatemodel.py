from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
import matplotlib.pyplot as plt
import numpy as np


class EvaluateModel():
    def __init__(self):
        self.pre = Precision()
        self.re = Recall()
        self.acc = CategoricalAccuracy()

    def batch_loop(self, pre_process, model, dataset, sequential_model):
        _, _, test = pre_process.train_test_split(dataset)
        # Loop through each batch in pipeline
        for batch in test.as_numpy_iterator():
            # Unpack the batch
            X_true, y_true = batch
            # Make a prediction
            yhat = model.predict(X_true)

            # Flatten the predictions, making one big vector
            y_true = y_true.flatten()
            yhat = yhat.flatten()

            # grabbing precision metric and updating the metric based on current batch of data
            self.pre.update_state(y_true, yhat)
            self.re.update_state(y_true, yhat)
            self.acc.update_state(y_true, yhat)

        print(f'Precision: {self.pre.result().numpy()}, Recall:{self.re.result().numpy()}, Accuracy:{self.acc.result().numpy()}')

        sequential_model.precision_values.append(self.pre.result().numpy())
        sequential_model.recall_values.append(self.re.result().numpy())
        sequential_model.accuracy_values.append(self.acc.result().numpy())

        return None

    def iplot(self, model): # here model is really sequential_model
        # Plotting Metrics
        plt.figure(figsize=(8, 5))
        plt.plot(model.epoch_values, model.precision_values,
                 label='Precision', marker='o')
        plt.plot(model.epoch_values, model.recall_values,
                 label='Recall', marker='o')
        plt.plot(model.epoch_values, model.accuracy_values,
                 label='Accuracy', marker='o')
        plt.title("Evaluation Metrics")
        plt.xlabel("Epochs Used During Training")
        plt.ylabel("Metric Value")
        plt.legend()

        avg_loss_values = []
        avg_val_loss_values = []

        for i in range(len(model.epoch_values)):
            avg_loss_values.append(np.average(model.loss_values[i]))
            avg_val_loss_values.append(np.average(model.val_loss_values[i]))

        plt.figure(figsize=(8, 5))
        plt.plot(model.epoch_values, avg_loss_values, label='loss',
                 marker='o')  # training loss
        plt.plot(model.epoch_values, avg_val_loss_values,
                 label='val_loss', marker='o')  # validation loss
        plt.title("Averaged Loss Values")
        plt.xlabel("Epochs Used During Training")
        plt.ylabel("Loss Value")
        plt.legend()
        plt.show()
