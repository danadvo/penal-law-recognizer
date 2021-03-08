import sys
from bag_of_words import *
from ml import train_model, split_train_and_set, use_model


def calculate(content):

    df = pd.read_csv("data/data_set.csv", header=0)

    # split the data to train and test set
    train_df, test_df = split_train_and_set(df)

    bag_of_words = set_vector_keyword_index(train_df)
    train_data, train_labels = get_vectors_and_labels(train_df, bag_of_words)
    classifier = train_model(train_data, train_labels)

    # # evaluate the model on the test set
    # test_data, test_labels = get_vectors_and_labels(test_df, bag_of_words)
    # use_model(classifier, test_data, test_labels)

    vector_content = [make_vector(content, bag_of_words)]
    y_predict = classifier.predict(vector_content)
    return y_predict[0]


if __name__ == '__main__':
    print (calculate(sys.argv[1]))