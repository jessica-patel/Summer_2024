import tensorflow as tf
from tensorflow.keras.layers import TextVectorization  # this import makes tokenization of NLP easier
# to convert words into numeric codes


class PreProcess():
    def __init__(self, dataset):
        self.dataset = dataset
        self.max_features = 200000
        self.vectorizer = TextVectorization(max_tokens=self.max_features,
                                            output_sequence_length=1800,  # max length of sentence in words
                                            output_mode='int')

    def split_data(self):
        """
        Want to split data into comments and features
        """
        X = self.dataset['comment_text']
        y = self.dataset[self.dataset.columns[2:]].values

        return X, y

    def vectorize(self, X):
        """ 
        Vectorizer lowers and strips punctuation, given in documentation
        """
        self.vectorizer.adapt(X.values)  # learning words, X.values converts panda series to array of strings
        vectorized_text = self.vectorizer(X.values)

        return vectorized_text

    def data_pipeline(self, vectorized_text, y, shuffle=16000, batch=16,
                      prefetch=8):
        """
        Creating a TensorFlow dataset pipeline, useful for data
        that doesn't fit in memory
        """
        # MCSHBAP - map, chache, shuffle, batch, prefetch
        # mapping: applying transformation to each element
        dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
        dataset = dataset.cache()  # caching: saves time by storing transformed data
        dataset = dataset.shuffle(shuffle)  # shuffling: stops model from learning order of the data
        dataset = dataset.batch(batch)  # groups data into batches for efficiency
        dataset = dataset.prefetch(prefetch)
        # helps bottlenecks, overlaps preprocessing/model execution of batch
        # with data loading of next batch using idle GPU/CPU, decrs loading
        # latency (loading into memory and turnover),
        # incrs throughput (data processed/unit time)

        return dataset

    def train_test_split(self, dataset):
        # Assigning dataset, 70% to training
        train = dataset.take(int(len(dataset)*.7))
        val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
        test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

        return train, val, test

    def get_vectorizer(self):
        return self.vectorizer
