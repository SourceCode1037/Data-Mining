import time

def merge():  # reduce again
    # read reduce_result_i
    start_time = time.time()
    word_count = {}
    for i in range(3):
        address = "./wordCount/reduce_result" + str(i + 1)
        f = open(address, "r")
        for line in f:
            word, count = line.strip().split('\t', 1)
            count=int(count)
            if word in word_count.keys():
                word_count[word] += count
            else:
                word_count[word] = count

    address_write = "./wordCount/final_result"
    f_write = open(address_write, "w")
    for word, count in word_count.items():
        f_write.write("%s\t%s\n" % (word, count))

    print("final_reduce" + " time: %.4f\n" % (time.time() - start_time));


if __name__ == "__main__":
    merge()
