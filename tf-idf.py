# coding=utf-8
import csv
import heapq

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist


def get_movie_data():
    f = open('./datasets/movies.csv', 'r')
    reader = csv.DictReader(f)
    movies_data = []
    map_id = {}
    map_g = {}
    for row in reader:
        movieId, title, genres = int(row['movieId']), row['title'], row['genres'].split('|')
        new_genres = []
        for genre in genres:
            if genre not in map_g.values():
                new_genres.append(len(map_g))
                map_g[len(map_g)] = genre
            else:
                new_genres.append(list(map_g.keys())[list(map_g.values()).index(genre)])

        if movieId not in map_id.values():
            new_movieId = len(map_id)
            map_id[len(map_id)] = movieId
        else:
            new_movieId = list(map_id.keys())[list(map_id.values()).index(movieId)]

        movies_data.append([new_movieId, title, new_genres])
    f.close()
    np.save('./movies_data.npy', movies_data)
    np.save('./map_id.npy', map_id)
    np.save('./map_g.npy', map_g)
    # data = np.array(movies_data)
    # data = data[np.argsort(data[:, 0])]

    return movies_data, map_id, map_g


def get_TFIDF_Matrix(movies_data, num_genres):
    TFIDF_Matrix = np.zeros(shape=(len(movies_data), num_genres), dtype=float)
    genres_occur_num = np.zeros(num_genres, dtype=int)

    for index in range(len(movies_data)):
        genres = movies_data[index][2]
        for genre in genres:
            genres_occur_num[genre] += 1

    for index in range(len(movies_data)):
        movieId, genres = movies_data[index][0], movies_data[index][2]
        TF = 1 / len(genres)
        for genre in genres:
            IDF = np.log10(len(movies_data) / genres_occur_num[genre])
            TFIDF_Matrix[index][genre] = TF * IDF

    return TFIDF_Matrix


def get_train_data(map_id):
    f = open('./datasets/train_set.csv', 'r')  # userId, movieId, rating,timestamp
    reader = csv.DictReader(f)
    train_data = []
    num_file = sum([1 for _ in open('./datasets/train_set.csv', 'r')])
    for row in tqdm(reader, total=num_file):
        userId, movieId, rating = int(row['userId']), int(row['movieId']), float(row['rating'])

        movieId_new = list(map_id.keys())[list(map_id.values()).index(movieId)]

        train_data.append([userId, movieId_new, rating])
    f.close()

    Utility_Matrix = np.zeros(shape=(train_data[-1][0], len(map_id)), dtype=np.float16)
    # csv用户编号[1,671]-----> Utility_Matrix编号[0,670]
    for item in train_data:
        Utility_Matrix[int(item[0]) - 1][int(item[1])] = item[2]

    np.save('./Utility_Matrix_tfidf.npy', Utility_Matrix)
    return Utility_Matrix


def get_test_set(map):
    f = open('./datasets/test_set.csv', 'r')  # userId, movieId, rating,timestamp
    reader = csv.DictReader(f)
    test_data = []
    for row in reader:
        userId, movieId, rating = int(row['userId']), int(row['movieId']), float(row['rating'])

        movieId_new = list(map.keys())[list(map.values()).index(movieId)]

        test_data.append([userId, movieId_new, rating])
    f.close()
    return test_data


def get_predict_rating(test_data, Utility_Matrix, TFIDF_Matrix):
    predict_rating = []

    for test_item in test_data:
        test_userId, test_movieId = test_item[0], test_item[1]

        rating = calculate_rating(test_userId, test_movieId, Utility_Matrix, TFIDF_Matrix)

        predict_rating.append(rating)

    return predict_rating


def calculate_rating(now_userId, now_movieId, Utility_Matrix, TFIDF_Matrix):
    watched_movies = np.nonzero(Utility_Matrix[now_userId - 1])[0]

    sim = []
    rating = []
    for movie in watched_movies:
        cosine = calculate_cosine(TFIDF_Matrix[movie], TFIDF_Matrix[now_movieId])
        if cosine > 0:
            sim.append(cosine)
            rating.append(Utility_Matrix[now_userId - 1][movie])

    if len(sim) > 0:
        predict_rating = (np.array(rating) * np.array(sim)).sum() / np.array(sim).sum()

    else:
        for movie in watched_movies:
            rating.append(Utility_Matrix[now_userId - 1][movie])

        predict_rating = np.array(rating).sum() / len(rating)

    return predict_rating


def calculate_cosine(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def calculate_sse(data_rating, predict_rating):
    norm_2 = np.linalg.norm((np.array(data_rating) - np.array(predict_rating)), axis=0, keepdims=True)[0]
    return norm_2 ** 2


def Recommendation_Systems(userId, map_id, Utility_Matrix, TFIDF_Matrix, K):
    rating_list = []
    watched_movies = np.nonzero(Utility_Matrix[userId - 1])[0]
    no_watched_movies = list(set([i for i in range(Utility_Matrix.shape[1])]) - set(watched_movies))

    for movieId in tqdm(no_watched_movies):
        rating = calculate_rating(userId, movieId, Utility_Matrix, TFIDF_Matrix)
        rating_list.append(rating)

    # get top-n index
    recommendation_movies = heapq.nlargest(K, range(len(rating_list)), rating_list.__getitem__)
    recommendation_movies_index = [no_watched_movies[i] for i in recommendation_movies]

    return recommendation_movies_index


if __name__ == '__main__':
    movies_data, map_id, map_g = get_movie_data()

    # map_id : <new_movie_Id,movie_Id>     map_g :<new_genre,genre>
    TFIDF_Matrix = get_TFIDF_Matrix(movies_data, len(map_g))

    # Utility_Matrix = get_train_data(map_id)
    Utility_Matrix = np.load('./Utility_Matrix_tfidf.npy')
    """test_data = get_test_set(map_id)

    predict_rating = get_predict_rating(test_data, Utility_Matrix, TFIDF_Matrix)

    data_rating = np.array(test_data)[:, 2].reshape(-1)

    sse = calculate_sse(data_rating, predict_rating)
    print("SSE = %f" % sse)"""
    userId = 10
    K = 3
    recommend_movies_index = Recommendation_Systems(userId, map_id, Utility_Matrix, TFIDF_Matrix, K)
    recommend_movies_raw = [map_id[i] for i in recommend_movies_index]
    print("user_%d recommend list:" % userId)
    for item in recommend_movies_index:
        print("movie_Id:%d" % map_id[item], end="\t")
        print(movies_data[item], end="\t")
        for g in movies_data[item][2]:
            print(map_g[g], end=" ")
        print("")
