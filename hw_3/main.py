import os
from pymorphy2 import MorphAnalyzer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
import ssl
from tqdm import tqdm
from nltk.corpus import stopwords
import json
from scipy import sparse

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
nltk.download("stopwords")
morph = MorphAnalyzer()
russian_stopwords = stopwords.words("russian")


# функция, которая возвращает список ответов: для каждого вопроса по одному ответу с самым высоким значением value
def parse_files():
    data_path = input("Укажите путь до папки, где лежит ваш файл 'questions_about_love.jsonl' (если он "
                      "лежит в текущей папке, то нажмите Enter):  ")
    filepath = os.path.join(data_path, 'questions_about_love.jsonl')
    with open(filepath, 'r') as f:
        corpus = list(f)[:50000]

    answers = []
    for i in tqdm(range(50000)):
        question = json.loads(corpus[i])
        answers_to_q = question['answers']
        if answers_to_q != []:
            answers_values = []
            for ans in answers_to_q:
                answer = ans['text']
                if ans['author_rating']['value'] != '':
                    value = int(ans['author_rating']['value'])
                else:
                    value = 0
                answers_values.append((answer, value))

            answers_values.sort(key=lambda x: x[1])
            answers.append(answers_values[-1][0])

    return answers


# функция препроцессинга: лемматизация, удаление пунктуации и стоп слов
def preprocessing(text):
    prep_text = []
    for word in text.split():
        word = morph.parse(word.strip(punctuation))[0].normal_form
        if word not in russian_stopwords:
            prep_text.append(word)

    return ' '.join(prep_text)


# функция, возвращающая матрицу bm25
def bm_25(texts):
    corpus = [preprocessing(text) for text in texts]
    x_count_vec = count_vectorizer.fit_transform(corpus)
    x_tf_vec = tf_vectorizer.fit_transform(corpus)
    x_idf_vec = tfidf_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_
    k = 2
    b = 0.75
    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()
    B_1 = (k * (1 - b + b * len_d / avdl))
    rows = []
    cols = []
    bm25 = []
    for i, j in tqdm(zip(*x_tf_vec.nonzero())):
        rows.append(i)
        cols.append(j)
        A = idf[j] * x_tf_vec[i, j] * (k + 1)
        B = x_tf_vec[i, j] + B_1[i]
        bm25.append(float(A / B))

    matrix = sparse.csr_matrix((bm25, (rows, cols)))
    return matrix


# функция, которая делает препроцессинг запроса пользователя и возвращает вектор
def vec_request(request):
    prep_req = preprocessing(request)
    return count_vectorizer.transform([prep_req])


# функция, которая считает близость с помощью bm25
def bm25_sim(request, matrix):
    return matrix.dot(request.T)


# главная функция, которая в необходимом порядке вызывает все остальные и
# возвращает 5 самых близких к запросу ответов
def main():
    answers = parse_files()
    matrix = bm_25(answers)
    while True:
        request = input("""Введите ваш запрос (или скажите "стоп"): """)
        if request != "стоп":
            vec_rec = vec_request(request)
            sim_req = bm25_sim(vec_rec, matrix)
            id_sort = np.argsort(sim_req.toarray(), axis=0)[::-1]
            print("5 наиболее соответствующих запросу серий: ")
            print('\n'.join(np.array(answers)[id_sort.ravel()][:5]))
        else:
            break


if __name__ == "__main__":
    main()
