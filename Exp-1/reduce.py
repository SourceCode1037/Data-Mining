import threading
import time


def reduce(num):
    start_time = time.time()
    word_count = {}
    for i in range(9):
        address = "./wordCount/map_result0" + str(i+1)+"_part"+str(num)
        f = open(address, "r")
        for line in f:
            word, count = line.strip().split('\t', 1)
            count=int(count)
            if word in word_count.keys():
                word_count[word] += count
            else:
                word_count[word] = count

    address_write = "./wordCount/reduce_result" + num
    f_write = open(address_write, "w")
    for word, count in word_count.items():
        f_write.write("%s\t%s\n" % (word, count))

    print("reduce_0" + str(int(num) % 3) + " time: %.4f\n" % (time.time() - start_time))


if __name__ == "__main__":
    reduce1 = threading.Thread(target=reduce, args=('1',))  # part1
    reduce2 = threading.Thread(target=reduce, args=('2',))  # part2
    reduce3 = threading.Thread(target=reduce, args=('3',))  # part3

    reduce1.start()
    reduce2.start()
    reduce3.start()
