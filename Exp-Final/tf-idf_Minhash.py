# coding=utf-8
import csv
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


def get_predict_rating(test_data, Utility_Matrix, signature_matrix):
    predict_rating = []

    for test_item in test_data:
        test_userId, test_movieId = test_item[0], test_item[1]

        rating = calculate_rating(test_userId, test_movieId, Utility_Matrix, signature_matrix)

        predict_rating.append(rating)

    return predict_rating


def calculate_rating(now_userId, now_movieId, Utility_Matrix, signature_matrix):
    watched_movies = np.nonzero(Utility_Matrix[now_userId - 1])[0]

    sim = []
    rating = []
    for movie in watched_movies:
        sim_jaccard = cal_sim_jaccard(signature_matrix[movie], signature_matrix[now_movieId])
        sim.append(sim_jaccard)
        rating.append(Utility_Matrix[now_userId - 1][movie])
    predict_rating = (np.array(rating) * np.array(sim)).sum() / np.array(sim).sum()
    """cosine = calculate_cosine(TFIDF_Matrix[movie], TFIDF_Matrix[now_movieId])
        if cosine > 0:
            sim.append(cosine)
            rating.append(Utility_Matrix[now_userId - 1][movie])

    if len(sim) > 0:
        predict_rating = (np.array(rating) * np.array(sim)).sum() / np.array(sim).sum()

    else:
        for movie in watched_movies:
            rating.append(Utility_Matrix[now_userId - 1][movie])

        predict_rating = np.array(rating).sum() / len(rating)"""
    return predict_rating


def calculate_cosine(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def calculate_sse(data_rating, predict_rating):
    norm_2 = np.linalg.norm((np.array(data_rating) - np.array(predict_rating)), axis=0, keepdims=True)[0]
    return norm_2 ** 2


def Zero_One(Utility_Matrix):
    return np.where(Utility_Matrix > 0, 1, 0)


def min_hash(TFIDF_Matrix, n_hash):
    Featurre_Matrix = Zero_One(TFIDF_Matrix)
    hash_matrix = np.zeros(shape=(n_hash, Featurre_Matrix.shape[1]))
    for i in range(n_hash):
        hash_matrix[i] = get_hash_function(Featurre_Matrix.shape[1])
    signature_matrix = get_signature_matrix(hash_matrix, Featurre_Matrix)

    # Sim_Matrix = get_Sim_Matrix(signature_matrix)
    return signature_matrix.T


def get_hash_function(length):
    index = np.arange(length)
    np.random.shuffle(index)
    return index


def get_signature_matrix(hash_matrix, Feature_Matrix):
    print("generate signature_matrix...")
    signature_matrix = np.zeros(shape=(len(hash_matrix), len(Feature_Matrix)), dtype=int)
    for i_hash in tqdm(range(len(hash_matrix))):
        for j_movie in range(len(Feature_Matrix)):
            min_hash_value = get_min_hash_value(hash_matrix[i_hash], Feature_Matrix[j_movie])
            signature_matrix[i_hash][j_movie] = min_hash_value

    return signature_matrix


def get_min_hash_value(hash_function, Feature_movie):
    for i in range(len(hash_function)):
        index = np.argwhere(hash_function == i)
        if Feature_movie[index] == 1:
            return i


def get_Sim_Matrix(signature_matrix):  # 未使用
    print("generate Sim_Matrix...")
    signature_matrix = signature_matrix.T
    Sim_Matrix = np.zeros(shape=(len(signature_matrix), len(signature_matrix)), dtype=float)
    for i in tqdm(range(len(signature_matrix))):
        for j in range(i + 1, len(signature_matrix)):
            sim_jaccard = cal_sim_jaccard(signature_matrix[i], signature_matrix[j])
            Sim_Matrix[i][j] = Sim_Matrix[j][i] = sim_jaccard

    return Sim_Matrix


def cal_sim_jaccard(A, B):
    # return len(set(A) & set(B)) / len(set(A))
    equal = (A == B)
    return equal.sum() / len(A)


if __name__ == '__main__':
    movies_data, map_id, map_g = get_movie_data()
    n_hash = 10
    # map_id : <new_movie_Id,movie_Id>     map_g :<new_genre,genre>
    TFIDF_Matrix = get_TFIDF_Matrix(movies_data, len(map_g))

    signature_matrix = min_hash(TFIDF_Matrix, n_hash)
    # Utility_Matrix = get_train_data(map_id)
    Utility_Matrix = np.load('./Utility_Matrix_tfidf.npy')
    test_data = get_test_set(map_id)

    predict_rating = get_predict_rating(test_data, Utility_Matrix, signature_matrix)

    data_rating = np.array(test_data)[:, 2].reshape(-1)

    sse = calculate_sse(data_rating, predict_rating)
    print("hash number = %d" % n_hash)
    print("SSE = %f" % sse)
