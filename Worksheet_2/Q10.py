def sum_dictionary_values():

    dict = {"a": 10, "b": 20, "c": 30}

    total = 0

    for value in dict.values():
        total += value

    print("Sum of all values: ", total)
    
sum_dictionary_values()