import numpy as np
import threading
import time


def map_and_combine(num):
    start_time = time.time()
    address = "./wordCount/source" + num
    f = open(address, 'r')
    s = [i.replace(",", "").split() for i in f.readlines()]
    words = np.array(s, dtype=object).reshape(-1)
    address_write = "./wordCount/map_result" + num
    word_count = {}

    for word in words:
        if word in word_count.keys():  # combine
            word_count[word] += 1
        else:
            word_count[word] = 1

    word_count = sorted(word_count.items(), key=lambda d: d[0])  # sort

    # partition
    """
    part_1: ' - number a~h A~H   (else of part_2,part_3)
    part_2: i~q I~Q
    part_3: r~z R~Z
    """

    f_write_part1 = open(address_write + '_part1', 'w')
    f_write_part2 = open(address_write + '_part2', 'w')
    f_write_part3 = open(address_write + '_part3', 'w')

    for i in range(len(word_count)):
        word = word_count[i][0]

        if ('i' <= word[0] <= 'q') or ('I' <= word[0] <= 'Q'):
            f_write_part2.write("%s\t%s\n" % (word_count[i][0], word_count[i][1]))

        elif ('r' <= word[0] <= 'z') or ('R' <= word[0] <= 'Z'):
            f_write_part3.write("%s\t%s\n" % (word_count[i][0], word_count[i][1]))

        else:
            f_write_part1.write("%s\t%s\n" % (word_count[i][0], word_count[i][1]))

    print("map_" + num + " time: %.4f\n" % (time.time() - start_time))


if __name__ == "__main__":

    map1 = threading.Thread(target=map_and_combine, args=('01',))
    map2 = threading.Thread(target=map_and_combine, args=('02',))
    map3 = threading.Thread(target=map_and_combine, args=('03',))
    map4 = threading.Thread(target=map_and_combine, args=('04',))
    map5 = threading.Thread(target=map_and_combine, args=('05',))
    map6 = threading.Thread(target=map_and_combine, args=('06',))
    map7 = threading.Thread(target=map_and_combine, args=('07',))
    map8 = threading.Thread(target=map_and_combine, args=('08',))
    map9 = threading.Thread(target=map_and_combine, args=('09',))

    map1.start()
    map2.start()
    map3.start()
    map4.start()
    map5.start()
    map6.start()
    map7.start()
    map8.start()
    map9.start()
