import string
from collections import Counter


def get_stop_words(penalty_list, laws_list):
    """ Remove common words which have no classification value """
    vocabulary_string = " ".join(laws_list)
    vocabulary_string = vocabulary_string.lower()
    vocabulary_list = tokenize(vocabulary_string)
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    vocabulary_list = [w.translate(table) for w in vocabulary_list]
    # remove remaining words that are not alphabetic
    vocabulary_list = [word for word in vocabulary_list if word.isalpha()]
    # filter out short words
    vocabulary_list = [word for word in vocabulary_list if len(word) > 1]

    words_counter = Counter(vocabulary_list)
    penalty_words_counter = Counter(penalty_list)

    # add to stopwords words that appear a lot of times and doesn't have significance for the classification
    stopwords = [word for word, count in words_counter.items() if count > 40 and
                 (penalty_words_counter[word] / count) < 0.6]
    # add to stopwords words that appear a few times and doesn't have significance for the classification
    stopwords.extend([word for word, count in penalty_words_counter.items() if
                      (count < 10 and words_counter[word] > 20)])
    return stopwords


def set_vector_keyword_index(df):

    df_penalty = df.loc[df['tag'] == 1.0]
    penalty_list = df_penalty['content'].tolist()
    lows_list = df['content'].tolist()

    # Mapped documents into a single word string
    vocabulary_string = " ".join(penalty_list)
    vocabulary_string = vocabulary_string.lower()

    # break string up into tokens and clean words from punctuation
    vocabulary_list = tokenize(vocabulary_string)
    # remove remaining words that are not alphabetic
    vocabulary_list = [word for word in vocabulary_list if word.isalpha()]
    # filter out short words
    vocabulary_list = [word for word in vocabulary_list if len(word) > 1]
    # filter out stopwords
    stopwords = get_stop_words(vocabulary_list, lows_list)
    vocabulary_list = [word for word in vocabulary_list if word not in stopwords]
    # remove duplicate
    vocabulary_list = set(vocabulary_list)

    offset = 0
    vector_key_word_index = {}
    # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
    for word in vocabulary_list:
        vector_key_word_index[word] = offset
        offset += 1
    return vector_key_word_index


def tokenize(content):
    """ break string up into tokens and clean words from punctuation """
    words = content.split(" ")
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]
    return words


def make_vector(word_string, bag_of_word):

    # Initialise vector with 0's
    vector = [0] * len(bag_of_word)
    word_list = tokenize(word_string)

    for word in word_list:
        if word in bag_of_word:
            vector[bag_of_word[word]] += 1  # Use simple Term Count Model
    return vector


def get_vectors_and_labels(df, bag_of_words):
    data, labels = [], []
    for index, row in df.iterrows():
        data.append(make_vector(row[1], bag_of_words))
        labels.append(int(row[0]))
    return data, labels


