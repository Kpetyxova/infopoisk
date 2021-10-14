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
import tensorflow as tf

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

count_vectorizer = CountVectorizer()
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

    return answers


# функция препрозессинта: лемматизация, удаление пунктуации и стоп слов
def preprocessing(text):
    prep_text = []
    for word in text.split():
        word = morph.parse(word.strip(punctuation))[0].normal_form
        if word not in russian_stopwords:
            prep_text.append(word)

    return ' '.join(prep_text)


# функция, которая индексирует корпус с помощью fasttext
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


# функция, которая векторизует запрос с помощью fasttext
def vec_request_fasttext(request):
    prep_req = preprocessing(request)
    lemmas = prep_req.split()
    request_vec = np.zeros((len(lemmas), 300))
    for i in range(len(lemmas)):
        if (lemmas[i] in fasttext_model) and (lemmas[i] != ''):
            request_vec[i] = fasttext_model[lemmas[i]]

    request_vec = np.mean(request_vec, axis=0)
    request_vec = request_vec.reshape(1, -1)
    return normalize(request_vec)


# функция, которая индексирует корпус с помощью модели sbert
def sbert_index(answer):
    encoded_input = tokenizer(answer, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
    return normalize(sentence_embeddings)


# функция, которая индексирует запрос с помощью модели sbert
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


def main():
    answers = parse_files()
    fasttext_matrix = fasttext_index(answers)
    np.save('fasttext_matrix.npy', fasttext_matrix)
    sbert_matrix = np.zeros((len(answers), 1024))
    for i in range(len(answers)):
        matrix = sbert_index(answers[i])
        sbert_matrix[i] = matrix[0]
    np.save('sbert_matrix.npy', sbert_matrix)
    while True:
        request = input("""Введите ваш запрос (или скажите "стоп"): """)
        if request != "стоп":
            vec_rec_fast = vec_request_fasttext(request)
            sim_req_fast = similarity(vec_rec_fast, fasttext_matrix)
            id_sort_fast = np.argsort(sim_req_fast, axis=0)[::-1]
            vec_rec_sbert = vec_request_sbert(request)
            sim_req_sbert = similarity(vec_rec_sbert, sbert_matrix)
            id_sort_sbert = np.argsort(sim_req_sbert, axis=0)[::-1]
            print('\n')
            print("FASTTEXT: ")
            print("5 наиболее соответствующих запросу ответов: ")
            print('\n'.join(np.array(answers)[id_sort_fast.ravel()][:5]))
            print('\n')
            print("SBERT: ")
            print('\n'.join(np.array(answers)[id_sort_sbert.ravel()][:5]))
            print('\n')
        else:
            break

if __name__ == "__main__":
    main()








