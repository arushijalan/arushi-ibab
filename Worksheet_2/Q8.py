def extract_words_with_k():

    L = ["kite", "king", "apple", "kangaroo", "banana", "key", "cat"]
    k = "k"

    result = []

    for word in L:
        if word[0] == k: 
            result.append(word)

    print("Words starting with", k, "in list", L, "are: ", result )
    
extract_words_with_k()