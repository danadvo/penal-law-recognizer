from collections import Counter
import pandas as pd


# class Bag:
#     def __init__(self, df):
#         self.vector_key_word_index = {}
#         self.stemmer = None
#         self.stop_word = None
#         self.df = df

def remove_stop_words(list, stopwords):
    """ Remove common words which have no search value """
    return [word for word in list if word not in stopwords]


def set_vector_keyword_index(df):

    df_penalty = df.loc[df['tag'] == 1.0]
    penalty_list = df_penalty['content'].tolist()
    lows_list = df['content'].tolist()

    # Mapped documents into a single word string
    vocabulary_string = " ".join(lows_list)
    vocabulary_string = vocabulary_string.lower()
    vocabulary_list = tokenize(vocabulary_string)
    words_counter = Counter(vocabulary_list)

    p_vocabulary_string = " ".join(penalty_list)
    p_vocabulary_string = p_vocabulary_string.lower()
    p_vocabulary_list = tokenize(p_vocabulary_string)
    p_words_counter = Counter(p_vocabulary_list)

    stopwords_counter = {word: counter for word, counter in words_counter.items() if
                         counter > 40 and p_words_counter[word] / counter < 0.6}
    new_p_counter = {word: count for word, count in p_words_counter.items() if word not in stopwords_counter}
    stam_words_counter = [word for word, count in new_p_counter.items() if
                          (count < 10 and words_counter[word] > 20)]
    new_p_counter = {word: count for word, count in new_p_counter.items() if word not in stam_words_counter}

    new_p_counter = {word: count for word, count in new_p_counter.items()
                     if new_p_counter[word]/words_counter[word] > 0.95}

    vocabulary_list = [w for w in new_p_counter]

    offset = 0
    vector_key_word_index = {}
    # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
    for word in vocabulary_list:
        vector_key_word_index[word] = offset
        # offset += 1
    return vector_key_word_index

    # data = {'word': [word for word in new_p_counter],
    #         'counter in all laws': [words_counter[word] for word in new_p_counter],
    # 'penalty rate': [new_p_counter[word]/words_counter[word] for word in new_p_counter],
    #         'is_good': [None for word in new_p_counter]}
    # new_df = pd.DataFrame(data=data)
    # new_df.to_excel("data/output.xlsx")

    # unique_vocabulary_list = set(vocabulary_list)
    # important_words = self.get_important_words(unique_vocabulary_list, self.df)


def tokenize(string):
    """ break string up into tokens and stem words """
    # string = self.clean(string)
    words = string.split(" ")
    return words
    # return [self.stemmer.stem(word, 0, len(word) - 1) for word in words]


def analysis(all_lows_content, penalty_lows_content):
    pass

# def get_vector_keyword_index(self, lows_list):
#     """ create the keyword associated to the position of the elements within the document vectors """
#
#     # Mapped documents into a single word string
#     vocabulary_string = " ".join(lows_list)
#     vocabulary_string = vocabulary_string.lower()
#     vocabulary_list = self.tokenize(vocabulary_string)
#     # Remove common words which have no search value
#     vocabulary_list = set(vocabulary_list)
#     vocabulary_list = self.get_important_words(vocabulary_list)
#     # self.remove_stop_words(vocabulary_list)
#
#     offset = 0
#     # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
#     for word in vocabulary_list:
#         self.vector_key_word_index[word] = offset
#         offset += 1
#     return self.vector_key_word_index  # (keyword:position)


def make_vector(word_string, bag_of_word):
    """ @pre: unique(vectorIndex) """

    # Initialise vector with 0's
    vector = [0] * len(bag_of_word)
    word_list = tokenize(word_string)
    # word_list = self.remove_stop_words(word_list)
    for word in word_list:
        if word in bag_of_word:
            idx = bag_of_word[word]
            vector[bag_of_word[word]] += 1  # Use simple Term Count Model
    return vector


def get_vectors_and_labels(df, bag_of_words):
    data, labels = [], []
    for index, row in df.iterrows():
        data.append(make_vector(row[1], bag_of_words))
        labels.append(int(row[0]))
    return data, labels


def get_important_words(words, df):
    frequency_in_penalty = {word: 0 for word in words}
    good_list = []

    for _, row in df.iterrows():
        for word in words:
            if word in row[1]:
                frequency_in_penalty[word] += 1

    for word in words:
        if frequency_in_penalty[word] > 5:
            good_list.append(word)

    return good_list

