from tqdm import tqdm
import csv
from itertools import combinations


def get_baskets_and_C1():
    f = open("Groceries.csv", 'r')
    reader = csv.DictReader(f)
    baskets = [set(row['items'].replace("{", "").replace("}", "").split(",")) for row in reader]

    items = set()
    for basket in baskets:
        items = items | basket
    C1 = []
    for item in items:
        C1.append({item})
    return baskets, C1


def generate_Ck(Lk_1, k):  # return all pairs in Lk_1
    print("\ngenerate C_%d..." % k)
    Ck = []
    for i in tqdm(range(len(Lk_1))):
        for j in range(i + 1, len(Lk_1)):  # Lk_1[i]|Lk_1[j]

            if list(Lk_1[i])[:k - 2] == list(Lk_1[j])[:k - 2]:
                set_ = Lk_1[i] | Lk_1[j]
                if set_ not in Ck:
                    Ck.append(set_)

    print("C_%d.len() = %d" % (k, len(Ck)))
    return Ck


def generate_Lk(Ck, baskets, min_support, k):
    print("\ngenerate L_%d..." % k)
    path = "./L" + str(k)
    f = open(path, 'w')
    Lk = []
    for set_ in tqdm(Ck):
        count = 0
        for basket in baskets:
            if set_.issubset(basket):
                count += 1
        support = count / len(baskets)
        if support >= min_support:
            Lk.append(set_)
            f.write(str(set_) + '  ' + str(support) + '\n')

    f.close()
    print("L_%d.len() = %d" % (k, len(Lk)))
    return Lk


def generate_rules(L, baskets, min_confident, k):
    print("\ngenerate rules...")
    path = "./rules_L" + str(k)
    f = open(path, 'w')
    N_rule = 0
    for set_ in tqdm(L):
        N_set = 0
        for basket in baskets:
            if set_.issubset(basket):
                N_set += 1

        # find all subset, delete empty and full set
        all_subset_list = sum([list(map(list, combinations(set_, i))) for i in range(len(set_) + 1)], [])[1:-1]
        for subset in all_subset_list:  # {A} -> {I-A}
            A = set(subset)
            I_A = set_ - A
            N_subset = 0
            for basket in baskets:
                if A.issubset(basket):
                    N_subset += 1
            confident = N_set / N_subset

            if confident >= min_confident:
                f.write(str(A) + " -> " + str(I_A) + "  " + str(confident) + '\n')
                N_rule += 1

    f.close()
    print("rules_L%d = %d" % (k, N_rule))
    return N_rule


if __name__ == "__main__":
    min_support = 0.005
    min_confident = 0.5

    baskets, C1 = get_baskets_and_C1()
    L1 = generate_Lk(C1, baskets, min_support, 1)

    C2 = generate_Ck(L1, 2)
    L2 = generate_Lk(C2, baskets, min_support, 2)
    a = generate_rules(L2, baskets, min_confident, 2)

    C3 = generate_Ck(L2, 3)
    L3 = generate_Lk(C3, baskets, min_support, 3)

    b = generate_rules(L3, baskets, min_confident, 3)
    print("Total rules number = %d" % (a + b))
