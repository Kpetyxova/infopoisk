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
import fasttext
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
import torch

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

count_vectorizer = CountVectorizer(analyzer='word')
tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
nltk.download("stopwords")
morph = MorphAnalyzer()
russian_stopwords = stopwords.words("russian")
fasttext_model = KeyedVectors.load("araneum_none_fasttextcbow_300_5_2018.model")
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")


# функция, которая возвращает список ответов: для каждого вопроса по одному ответу с самым высоким значением value
def parse_files():
    data_path = input("Укажите путь до папки, где лежит ваш файл 'questions_about_love.jsonl' (если он "
                      "лежит в текущей папке, то нажмите Enter):  ")
    filepath = os.path.join(data_path, 'questions_about_love.jsonl')
    with open(filepath, 'r') as f:
        corpus = list(f)[:10000]

    answers = []
    questions = []
    for i in tqdm(range(10000)):
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
            questions.append(question['question'])

    return answers, questions


# функция препрозессинга: лемматизация, удаление пунктуации и стоп слов
def preprocessing(text):
    prep_text = []
    for word in text.split():
        word = morph.parse(word.strip(punctuation))[0].normal_form
        if word not in russian_stopwords:
            prep_text.append(word)

    return ' '.join(prep_text)


# функция индексации корпуса с помощью countvectorizer
def countvectorizer_index(prep_answers):
    return normalize(count_vectorizer.fit_transform(prep_answers))


# функция индексации запроса с помощью countvectorizer
def vec_request_countvectorizer(prep_req):
    return normalize(count_vectorizer.transform(prep_req))


# функция индексации корпуса с помощью TFIDF
def tfidf_vectorizer_index(prep_answers):
    return tfidf_vectorizer.fit_transform(prep_answers)


# функция индексации запроса с помощью TFIDF
def vec_request_tfidf_vectorizer(prep_req):
    return tfidf_vectorizer.transform(prep_req)


# функция индексации корпуса с помощью bm25
def bm_25(corpus):
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


# функция индексации запроса для bm25
def vec_request_bm25(prep_req):
    return count_vectorizer.transform(prep_req)


# функция индексации корпуса с помощью fasttext
def fasttext_index(texts):
    corpus = [preprocessing(text) for text in texts]
    matrix = np.zeros((len(corpus), 300))
    for i in tqdm(range(len(corpus))):
        lemmas = corpus[i].split()
        lemmas_matrix = np.zeros((len(lemmas), 300))
        for j in range(len(lemmas)):
            if (lemmas[j] in fasttext_model) and (lemmas[j] != ''):
                lemmas_matrix[j] = fasttext_model[lemmas[j]]

        vec = np.mean(lemmas_matrix, axis=0)
        matrix[i] = vec
        matrix = np.nan_to_num(matrix)

    return normalize(matrix)


# функция индексации запроса fasttext-ом
def vec_request_fasttext(request):
    lemmas = request.split()
    request_vec = np.zeros((len(lemmas), 300))
    for i in range(len(lemmas)):
        if (lemmas[i] in fasttext_model) and (lemmas[i] != ''):
            request_vec[i] = fasttext_model[lemmas[i]]

    request_vec = np.mean(request_vec, axis=0)
    request_vec = np.nan_to_num(request_vec)
    request_vec = request_vec.reshape(1, -1)
    return normalize(request_vec)


# функция индексации корпуса с помощью sbert
def sbert_index(answer):
    encoded_input = tokenizer(answer, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
    return normalize(sentence_embeddings)


# функция индексации запроса sbert-ом
def vec_request_sbert(request):
    encoded_input = tokenizer(request, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    request_embedding = cls_pooling(model_output, encoded_input['attention_mask'])
    return normalize(request_embedding)


# функция, которая находит cls-токены
def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]


# функция подсчета близости
def similarity(request, matrix):
    return matrix.dot(request.T)


# функция для подсчета метрики
def get_score(answers_matrix, questions_matrix):
    sim = similarity(questions_matrix, answers_matrix)
    counter = 0
    for i in range(sim.shape[0]):
        id_sort = np.argsort(sim[i], axis=0)[::-1]
        if i in id_sort[:5]:
            counter += 1

    return counter / questions_matrix.shape[0]


# главная функция, объединяющая все предыдущие
def main():
    answers, questions = parse_files()
    prep_answers = [preprocessing(a) for a in answers]
    prep_questions = [preprocessing(q) for q in questions]
    countvectorizer_matrix = countvectorizer_index(prep_answers)
    req_countvectorizer = vec_request_countvectorizer(prep_questions)
    tfidf_matrix = tfidf_vectorizer_index(prep_answers)
    req_tfidf = vec_request_tfidf_vectorizer(prep_questions)
    bm25_matrix = bm_25(answers)
    req_bm25 = vec_request_bm25(prep_questions)
    # fasttext_matrix = fasttext_index(answers)
    fasttext_matrix = np.load('fasttext_matrix.npy') # мы уже проиндексировали корпус в прошлом задании
    req_fasttext = np.zeros((len(prep_questions), 300))
    # sbert_matrix = np.zeros((len(answers), 1024))
    sbert_matrix = np.load('sbert_matrix.npy')
    req_sbert = np.zeros((len(questions), 1024))
    for i in tqdm(range(len(answers))):
        # matrix = sbert_index(answers[i])
        # sbert_matrix[i] = matrix[0]
        reqs = vec_request_sbert(questions[i])
        req_sbert[i] = reqs[0]
        req_fast = vec_request_fasttext(prep_questions[i])
        req_fasttext[i] = req_fast[0]

    countvectorizer_score = get_score(req_countvectorizer, countvectorizer_matrix)
    tfidf_score = get_score(req_tfidf, tfidf_matrix)
    bm25_score = get_score(req_bm25, bm25_matrix)
    fasttext_score = get_score(req_fasttext, fasttext_matrix)
    sbert_score = get_score(req_sbert, sbert_matrix)
    print("CountVectorizer: ", countvectorizer_score)
    print("TF-IFD: ", tfidf_score)
    print("bm25: ", bm25_score)
    print("Fasttext: ", fasttext_score)
    print("Sbert: ", sbert_score)


if __name__ == "__main__":
    main()
