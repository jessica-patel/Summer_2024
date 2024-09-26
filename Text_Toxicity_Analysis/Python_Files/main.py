import os  # for working with file paths
import pandas as pd  # to read in tabular data (csvs)

from preprocess import PreProcess
from sequentialmodel import SequentialModel
from predictions import Predictions
from evaluatemodel import EvaluateModel
# from testing import Testing

# Alternative for CUDA: Apple's Metal Performance Shaders (MPS) framework for
# GPU acceleration on macOS
# import torch
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("MPS backend is available. Using MPS.")
# else:
#     device = torch.device("cpu")
#     print("MPS backend is not available. Using CPU.")

if __name__ == '__main__':
    # read csv into dataframe
    df = pd.read_csv(os.path.join('jigsaw_challenge_data', 'train.csv'))
    # df_train = pd.read_csv('jigsaw_challenge_data/train.csv')
    # df_test = pd.read_csv('jigsaw_challenge_data/test.csv')
    # df_test_labels = pd.read_csv('jigsaw_challenge_data/test_labels.csv')

    # preprocessing
    pre_process = PreProcess(df)
    X, y = pre_process.split_data()

    vectorized_text = pre_process.vectorize(X)
    dataset = pre_process.data_pipeline(vectorized_text, y)

    train, val, test = pre_process.train_test_split(dataset)

    print("Update: preprocessing done")

    # creating sequential model
    epoch_values = []
    loss_values = []
    val_loss_values = []
    prediction_arrays = [[]]
    precision_values = []
    recall_values = []
    accuracy_values = []

    chosen_epoch = 10

    sequential_model = SequentialModel(epoch_values, loss_values,
                                       val_loss_values, prediction_arrays,
                                       precision_values, recall_values,
                                       accuracy_values, val, train,
                                       chosen_epoch)

    sequential_model.add_layers(sequential_model.model)

    s_model_history = sequential_model.run_model(sequential_model.model)

    sequential_model.iplot(s_model_history)

    print("Update: model building done")

    # making predictions
    input_phrase = 'You freaking suck! I am going to hit you'
    threshold = 0.5
    
    predictions = Predictions()
    input_text = predictions.input_phrase(pre_process.get_vectorizer(), input_phrase, sequential_model.model)
    predictions.print_batches(pre_process, sequential_model.model, dataset)

    print("Update: predictions done")

    # evaluating model
    evaluate_model = EvaluateModel()
    evaluate_model.batch_loop(pre_process, sequential_model.model, dataset, sequential_model)
    evaluate_model.iplot(sequential_model)
    
    print("Update: evaluation complete")

    # testing model
    testing_phrase = 'hey i freaken hate you!'
    scoring_comment = ''
    test_model = Testing(sequential_model.model)
    test_model.save_model()
    test_model.new_model(pre_process.get_vectorizer(), testing_phrase)
    test_model.score_comment(df, pre_process.get_vectorizer(), scoring_comment)
