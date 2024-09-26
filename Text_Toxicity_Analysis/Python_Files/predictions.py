import numpy as np

class Predictions():

    def input_phrase(self, vectorizer, input_phrase, model):
        input_vec = vectorizer(input_phrase) # for testing
        res = model.predict(np.expand_dims(input_vec, 0))
        # model.prediction_arrays.append(res)
        print(f"Prediction for test: {res}\n")

        print(f"Positives for threshold of 0.5: {(res > 0.5).astype(int)}")

    def print_batches(self, pre_process, model, dataset):
        _, _, test = pre_process.train_test_split(dataset)
        batch_X, _ = test.as_numpy_iterator().next()  # grabbing another batch to run prediction again
        print((model.predict(batch_X) > 0.5).astype(int))  # seeing which ones pass (>0.5) and putting them into binary form
