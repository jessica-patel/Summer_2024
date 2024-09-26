import numpy as np
import tensorflow as tf


class Testing():
    def __init__(self, sequential_model):
        self.model = sequential_model

    def save_model(self):
        self.model.save('toxicity.h5')

    def new_model(self, vectorizer, testing_phrase):
        self.model = tf.keras.models.load_model('toxicity.h5')

        input_str = vectorizer(testing_phrase)
        res = self.model.predict(np.expand_dims(input_str, 0))
        print("New final test: ", res)

    def score_comment(self, dataset, vectorizer, comment):
        vectorized_comment = vectorizer(comment)
        results = self.model.predict(vectorized_comment)

        text = ''
        for idx, col in enumerate(dataset.columns[2:]):
            text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

        return text
