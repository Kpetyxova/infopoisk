import os
import re
from pymorphy2 import MorphAnalyzer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='word')
import numpy as np
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
from nltk.corpus import stopwords



def parse_files(path):
    filepaths = []
    texts = []
    curr_dir = os.getcwd()
    data_path = os.path.join(curr_dir, path)
    for root, dirs, files in os.walk(data_path):
        for name in files:
            if name[0] != '.':
                filepaths.append(os.path.join(root, name))

    for filepath in filepaths:
        with open(filepath, 'r', errors='ignore') as f:
            text = f.read()
        texts.append(text)

    return texts


def preprocessing():
    morph = MorphAnalyzer()
    russian_stopwords = stopwords.words("russian")
    texts = parse_files('friends-data')
    preprocessed_texts = []
    for text in texts:
        text = ''.join([i for i in text if not i.isdigit()])
        text = re.sub(r'[a-zA-Z]', '', text)
        preprocessed_texts.append(' '.join(morph.parse(w.strip(punctuation))[0].normal_form for w in text.split()
                                           if (w not in russian_stopwords) and (w.isdigit() is False)))

    return preprocessed_texts


def reverse_index():
    corpus = preprocessing()
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    return X, feature_names

def answers():
    X, feature_names = reverse_index()
    matrix_freq = np.array(X.sum(axis=0)).ravel()
    max_index = np.argmax(matrix_freq)
    most_freq_word = feature_names[max_index]
    print("Самое частотное слово: ", most_freq_word)
    min_index = np.argmin(matrix_freq)
    most_rare_word = feature_names[min_index]
    print("Самое редкое слово: ", most_rare_word)
    arr = X.toarray()
    list_matrix = arr.tolist()
    all_texts_words = []
    for i in range(len(list_matrix[0])):
        flag = True
        for l in list_matrix:
            if l[i] == 0:
                flag = False
                break
        if flag:
            all_texts_words.append(feature_names[i])

    str_all_t = ', '.join(all_texts_words)
    print("Слова, которые встречаются во всех текстах: ", str_all_t)
    characters = {'Моника': ['моника', 'мон'],
                  'Рэйчел': ['рэйчел', 'рейч'],
                  'Чендлер': ['чендлер', 'чэндлер', 'чен'],
                  'Фиби': ['фиби', 'фибс'],
                  'Росс': ['росс'],
                  'Джоуи': ['джоуи', 'джои', 'джо']}

    most_popular = ''
    freq_pop = 0
    for key, values in characters.items():
        counter = 0
        for v in values:
            index_v = vectorizer.vocabulary_.get(v)
            if index_v is not None:
                freq = int(matrix_freq[index_v])
                counter += freq
        if counter > freq_pop:
            freq_pop = counter
            most_popular = key

    print("Самый упоминаемый персонаж: ", most_popular)


def main():
    answers()

if __name__ == "__main__":
    main()







