import csv
import random
import numpy as np
import matplotlib.pyplot as plt


def get_data():
    f = open("./NormalizedData.csv")
    reader = csv.reader(f)
    Y = []
    X = []
    for line in reader:
        X.append(list(map(float, line[1:])))
        Y.append(int(line[0]))
    return X, Y


def K_means(X):
    X = np.array(X)
    Y = np.zeros(len(X))
    new_Y = np.zeros(len(X))
    centers = []
    center_indexs = random.sample(list(range(len(X))), 3)
    for index in center_indexs:
        centers.append(list(X[index]))

    # iteration
    for iteration in range(100):
        for i in range(len(X)):
            new_Y[i] = get_cluster(X[i], centers)

        new_centers = get_new_centers(X, new_Y, centers)

        # error = np.where(np.abs(new_Y - Y) >= 1, 1, 0).sum()
        error = np.linalg.norm((np.array(new_centers) - np.array(centers)), axis=1, keepdims=True).sum()
        print("iteration_%d: error=%f" % (iteration, error))

        # if error==0:
        if error < 10e-6:
            break
        else:
            Y = new_Y
            centers = new_centers

    return Y, centers


def get_cluster(x, centers):
    min_distance = 10e6
    cluster = 0
    for i in range(len(centers)):
        distance = np.linalg.norm((np.array(x) - np.array(centers[i])), axis=0, keepdims=True).item()  # default 2-norm
        if distance < min_distance:
            cluster = i + 1
            min_distance = distance

    return cluster


def get_new_centers(X, Y, centers):
    new_centers = []
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []

    for i in range(len(X)):
        if Y[i] == 1:
            cluster_1.append(list(X[i]))
        elif Y[i] == 2:
            cluster_2.append(list(X[i]))
        else:
            cluster_3.append(list(X[i]))

    new_centers.append(list(np.array(cluster_1).sum(axis=0) / len(cluster_1)))
    new_centers.append(list(np.array(cluster_2).sum(axis=0) / len(cluster_2)))
    new_centers.append(list(np.array(cluster_3).sum(axis=0) / len(cluster_3)))

    return new_centers


def draw(X, label, select_dim, SSE, Acc, centers):
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []
    for i in range(len(X)):
        if label[i] == 1:
            cluster_1.append(list(X[i]))
        elif label[i] == 2:
            cluster_2.append(list(X[i]))
        else:
            cluster_3.append(list(X[i]))

    X0_axis = np.array(cluster_1)[:, select_dim[0]]  # 0-th column ->X
    Y0_axis = np.array(cluster_1)[:, select_dim[1]]  # 1-th column ->Y
    X1_axis = np.array(cluster_2)[:, select_dim[0]]  # 0-th column ->X
    Y1_axis = np.array(cluster_2)[:, select_dim[1]]  # 1-th column ->Y
    X2_axis = np.array(cluster_3)[:, select_dim[0]]  # 0-th column ->X
    Y2_axis = np.array(cluster_3)[:, select_dim[1]]  # 1-th column ->Y
    plt.title('SSE=%f Acc=%f' % (SSE, Acc))
    plt.xlabel('dimension_%d' % select_dim[0])
    plt.ylabel('dimension_%d' % select_dim[1])
    plt.xlim(xmin=-0.1, xmax=1.1)
    plt.ylim(ymin=-0.1, ymax=1.1)

    area = np.pi * 3 ** 2  # area

    plt.scatter(X0_axis, Y0_axis, s=area, c='red', alpha=0.3, label='cluster-1')
    plt.scatter(X1_axis, Y1_axis, s=area, c='blue', alpha=0.3, label='cluster-2')
    plt.scatter(X2_axis, Y2_axis, s=area, c='green', alpha=0.3, label='cluster-3')

    X_center = np.array(centers)[:, select_dim[0]]
    Y_center = np.array(centers)[:, select_dim[1]]

    plt.scatter(X_center, Y_center, marker=',', c='black', alpha=1, label='cluster-3')

    plt.show()


def get_data_centers(X):
    centers = []

    cluster_1 = X[:59]
    cluster_2 = X[59:130]
    cluster_3 = X[130:]

    centers.append(list(np.array(cluster_1).sum(axis=0) / len(cluster_1)))
    centers.append(list(np.array(cluster_2).sum(axis=0) / len(cluster_2)))
    centers.append(list(np.array(cluster_3).sum(axis=0) / len(cluster_3)))

    return centers


def match_centers(cluster_centers, Y_cluster):
    data_centers = get_data_centers(X)
    match = []
    for cluster_center in cluster_centers:
        min_distance = 10e6
        index = 0
        for i in range(3):
            distance = np.linalg.norm((np.array(cluster_center) - np.array(data_centers[i])), axis=0, keepdims=True)[0]
            if distance < min_distance:
                min_distance = distance
                index = i + 1  # match certain data-label
        match.append(index)

    for i in range(len(Y_cluster)):
        Y_cluster[i] = match[int(Y_cluster[i] - 1)]

    return Y_cluster


def get_SSE(X, centers, Y):
    distance = []
    for i in range(len(X)):
        distance.append(
            np.linalg.norm((np.array(X[i]) - np.array(centers[int(Y[i]) - 1])), axis=0, keepdims=True).item())
    distance = np.array(distance) ** 2
    sse = distance.sum()
    return sse


def get_Acc(Y_data, Y_cluster):
    right_num = 0
    for i in range(len(Y_data)):
        if Y_data[i] == Y_cluster[i]:
            right_num += 1

    Acc = right_num / len(Y_data)
    return Acc


if __name__ == "__main__":
    X, Y_data = get_data()
    select_dimension = [0, 11]
    Y_cluster, cluster_centers = K_means(X)
    SSE = get_SSE(X, cluster_centers, Y_cluster)
    Y_cluster = match_centers(cluster_centers, Y_cluster)
    Acc = get_Acc(Y_data, Y_cluster)
    print("SSE: %f" % SSE)
    print("Accuracy: %f" % (Acc * 100) + "%")
    draw(X, Y_cluster, select_dimension, SSE, Acc, cluster_centers)
