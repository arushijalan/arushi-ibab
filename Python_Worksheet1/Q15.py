def count_word_occ():
    S = input("Enter a Sentence: ")
    W = input("Enter the word whose occurrence you want to find: ")
    # count = 0
    # for word in S:
    #     if word == W:
    #         count = count + 1
    #     else:
    #         continue
    S = S.lower()
    count = S.count(W)
    print("The number of occurrence of word", W, "is", count)

def main():
    count_word_occ()

if __name__ == "__main__":
    main()