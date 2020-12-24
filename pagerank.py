import numpy as np
import csv


def create_M_R_map():
    f = open("sent_receive.csv", 'r')
    reader = csv.reader(f)
    result = list(reader)
    map = {}
    links = []
    for i in range(1, len(result)):  # neglect first line [sent_id	receive_id]
        send_id, receive_id = result[i][1], result[i][2]
        # map->reduce the matrix_M's dimension
        if send_id not in map.values():
            map[len(map)] = send_id
            send_id_new = len(map) - 1
        else:
            send_id_new = list(map.keys())[list(map.values()).index(send_id)]

        if receive_id not in map.values():
            map[len(map)] = receive_id
            receive_id_new = len(map) - 1
        else:
            receive_id_new = list(map.keys())[list(map.values()).index(receive_id)]

        # print("%s   %s" % (send_id, receive_id))
        # print("%s   %s\n" % (send_id_new, receive_id_new))
        links.append([send_id_new, receive_id_new])

    # print(len(map))

    M = np.zeros((len(map), len(map)))
    for link in links:
        M[link[1]][link[0]] = 1  # link[0]->link[1],,i->j,,M[j][i]=1/d_i

    d_i = np.sum(M, axis=0)
    d_i = np.where(d_i == 0, 1, d_i)  # avoid isolated node, whose degree=0, and M == 0
    M = M / d_i

    R = np.zeros((len(map), 1)) + 1 / len(map)

    return M, R, map


def iteration(M, R, Beta):
    for i in range(100):
        print("\niteration %d" % (i + 1))

        # R=Beta*M@R+(1-Beta)*1/N
        R_next = Beta * np.matmul(M, R) + (1 - Beta) * 1 / len(R) * np.ones((len(R), 1))

        R_next = R_next / np.sum(R_next)

        error = np.linalg.norm((R_next - R), axis=0, keepdims=True)[0][0]  # default 2-norm
        R = R_next
        print("error:%.10f\t" % error)
        if (error < 10e-8):
            print("Error < 10e-8, End!\n")
            break
        # print("max value:%f" % np.max(R_next))
        # print("sum value:%f" % np.sum(R_next))
        # print(R)

    return R


if __name__ == "__main__":
    Beta = 0.85
    M, R, map = create_M_R_map()
    R = iteration(M, R, Beta)

    index = np.arange(len(R))[np.newaxis, :]

    lis = np.concatenate((index, R.T), axis=0)

    for i in range(len(R)):
        lis[0][i] = map[i]

    node_value = lis.T[np.lexsort(lis)].T
    print("Top-10 rank value nodes(Beta=%.2f):" % Beta)
    for i in range(len(R) - 1, len(R) - 11, -1):
        print("node_%d:\t%f" % (node_value[0][i], node_value[1][i]))
