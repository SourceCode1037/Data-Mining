# coding=utf-8
import csv
import numpy as np
from tqdm import tqdm
import heapq
import scipy.spatial.distance as dist


def get_train_set():
    print("Process train_set.csv ...")
    f = open('./datasets/train_set.csv', 'r')  # userId, movieId, rating,timestamp
    reader = csv.DictReader(f)
    train_data = []
    map = {}
    num_file = sum([1 for _ in open('./datasets/train_set.csv', 'r')])
    for row in tqdm(reader, total=num_file):
        userId, movieId, rating = int(row['userId']), int(row['movieId']), float(row['rating'])

        if movieId not in map.values():
            map[len(map)] = movieId
            movieId_new = len(map) - 1
        else:
            movieId_new = list(map.keys())[list(map.values()).index(movieId)]

        train_data.append([userId, movieId_new, rating])
    f.close()

    Utility_Matrix = np.zeros(shape=(train_data[-1][0], len(map)), dtype=np.float16)
    # csv用户编号[1,671]-----> Utility_Matrix编号[0,670]
    for item in train_data:
        Utility_Matrix[int(item[0]) - 1][int(item[1])] = item[2]

    np.save('./Utility_Matrix.npy', Utility_Matrix)
    np.save('./map.npy', map)
    return train_data, Utility_Matrix, map


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


def get_predict_rating(test_data, Utility_Matrix, Sim_Matrix, K):
    predict_rating = []
    for test_item in test_data:
        test_userId, test_movieId = test_item[0], test_item[1]
        top_k_userId, top_k_sim = find_top_k_user(test_userId, test_movieId, Utility_Matrix, Sim_Matrix, K)
        top_k_rating = [Utility_Matrix[Id][test_movieId] for Id in top_k_userId]
        rating = (np.array(top_k_rating) * np.array(top_k_sim)).sum() / np.array(top_k_sim).sum()
        predict_rating.append(rating)
    return predict_rating


def find_top_k_user(now_userId, now_movieId, Utility_Matrix, Sim_Matrix, K):
    top_k_userId = []
    top_k_sim = []

    userId_watched_movie = np.nonzero(Utility_Matrix[:, now_movieId])[0]

    now_userId_sim = Sim_Matrix[now_userId]
    userId_sim = np.concatenate((np.arange(len(now_userId_sim)), now_userId_sim)).reshape((2, len(now_userId_sim)))
    top_userId_sim = userId_sim[:, userId_sim[1].argsort()]

    for index in range(len(now_userId_sim) - 1, -1, -1):  # Reverse order
        userId = top_userId_sim[0][index]
        if userId in userId_watched_movie:
            top_k_userId.append(int(userId))
            top_k_sim.append(top_userId_sim[1][index])
            if len(top_k_userId) == K:
                break
    return top_k_userId, top_k_sim


def calculate_sse(data_rating, predict_rating):
    norm_2 = np.linalg.norm((np.array(data_rating) - np.array(predict_rating)), axis=0, keepdims=True)[0]
    return norm_2 ** 2


def Recommendation_Systems(userId, map, Utility_Matrix, Sim_Matrix, K, n):
    rating_list = []
    for movieId in tqdm(range(len(map))):
        top_k_userId, top_k_sim = find_top_k_user(userId, movieId, Utility_Matrix, Sim_Matrix, K)

        top_k_rating = [Utility_Matrix[Id][movieId] for Id in top_k_userId]
        rating = (np.array(top_k_rating) * np.array(top_k_sim)).sum() / np.array(top_k_sim).sum()
        rating_list.append(rating)

    # get top-n index
    recommendation_movies = heapq.nlargest(n, range(len(rating_list)), rating_list.__getitem__)
    recommendation_movies_raw = [map[i] for i in recommendation_movies]
    return recommendation_movies_raw


def Zero_One(Utility_Matrix):
    return np.where(Utility_Matrix >= 3, 1, 0)


def min_hash(Utility_Matrix, n_hash):
    Utility_Matrix = Zero_One(Utility_Matrix)
    hash_matrix = np.zeros(shape=(n_hash, Utility_Matrix.shape[1]))
    for i in range(n_hash):
        hash_matrix[i] = get_hash_function(Utility_Matrix.shape[1])
    signature_matrix = get_signature_matrix(hash_matrix, Utility_Matrix)

    Sim_Matrix = get_Sim_Matrix(signature_matrix)
    return Sim_Matrix


def get_hash_function(length):
    index = np.arange(length)
    np.random.shuffle(index)
    return index


def get_signature_matrix(hash_matrix, Utility_Matrix):
    print("generate signature_matrix...")
    signature_matrix = np.zeros(shape=(len(hash_matrix), len(Utility_Matrix)), dtype=int)
    for i_hash in tqdm(range(len(hash_matrix))):
        for j_user in range(len(Utility_Matrix)):
            min_hash_value = get_min_hash_value(hash_matrix[i_hash], Utility_Matrix[j_user])
            signature_matrix[i_hash][j_user] = min_hash_value

    return signature_matrix


def get_min_hash_value(hash_function, Utility_user):
    for i in range(len(hash_function)):
        index = np.argwhere(hash_function == i)
        if Utility_user[index] == 1:
            return i


def get_Sim_Matrix(signature_matrix):
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
    K = 22
    N = 3
    userId = 1
    n_hash = 50
    # train_data, Utility_Matrix, map = get_train_set()
    # Sim_Matrix = get_Sim_Matrix(Utility_Matrix)
    # Sim_Matrix = np.load('./Sim_Matrix.npy')
    Utility_Matrix = np.load('./Utility_Matrix.npy')
    map = np.load('./map.npy', allow_pickle=True).item()

    # recommend_list = Recommendation_Systems(userId, map, Utility_Matrix, Sim_Matrix, K, N)
    # print(recommend_list)
    Sim_Matrix = min_hash(Utility_Matrix, n_hash)
    test_data = get_test_set(map)
    predict_rating = get_predict_rating(test_data, Utility_Matrix, Sim_Matrix, K)

    data_rating = np.array(test_data)[:, 2].reshape(-1)
    sse = calculate_sse(data_rating, predict_rating)
    print("most similar user number = %d" % K)
    print("hash number = %d" % n_hash)
    print("SSE = %f" % sse)
