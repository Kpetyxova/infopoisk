import os
import re
from pymorphy2 import MorphAnalyzer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
import numpy as np
import nltk
import ssl
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
from nltk.corpus import stopwords
morph = MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

# функция, которая возвразает список названий файлов и список всех текстов
def parse_files():
    filepaths = []
    filenames = []
    texts = []
    data_path = input('Укажите путь до папки friends-data: ')
    for root, dirs, files in os.walk(data_path):
        for name in files:
            if name[0] != '.':
                filepaths.append(os.path.join(root, name))
                filenames.append(name)
    for filepath in filepaths:
        with open(filepath, 'r', errors='ignore') as f:
            text = f.read()
        texts.append(text)

    return texts, filenames

# функция препрозессинта: лемматизация, удаление пунктуации и стоп слов
def preprocessing(texts):
    preprocessed_texts = []
    for text in tqdm(texts):
        preprocessed_texts.append(' '.join(morph.parse(w.strip(punctuation))[0].normal_form for w in text.split()
                                           if w not in russian_stopwords))

    return preprocessed_texts

# функция, возвразающая обратный индекс с использованием тф-идф
def reverse_index():
    texts, filenames = parse_files()
    corpus = preprocessing(texts)
    X = vectorizer.fit_transform(corpus)
    return X, filenames

# функция, которая делает препроцессинг запроса пользователя и возвращает вектор
def vec_request(request):
    prep_req = preprocessing([request])
    return vectorizer.transform(prep_req)

# функция, которая считает косинусную близость
def cos_sim(x, vector):
    cos_sim = cosine_similarity(x, vector)
    return cos_sim.reshape(-1)

# главная функция, которая в необходимом порядке вызывает все остальные и
# возвращает 5 самых близких к запросу серий
def main():
    x, filenames = reverse_index()
    while True:
        request = input("""Введите ваш запрос (или скажите "стоп"): """)
        if "стоп" not in request:
            names_sorted = []
            vec_rec = vec_request(request)
            list_cos = cos_sim(x, vec_rec)
            id_sort = np.argsort(list_cos)[::-1]
            id_sort = id_sort.tolist()
            for i in range(5):
                names_sorted.append(filenames[id_sort[i]])
            print("5 наиболее соответствующих запросу серий: ")
            print('\n'.join(names_sorted))
        else:
            break


if __name__ == "__main__":
    main()


