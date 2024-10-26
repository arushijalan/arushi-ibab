def key_with_most_unique_values():
    test_dict = {"Gfg": [5, 7, 7, 7, 7], "is": [6, 7, 7, 7], "Best": [9, 9, 6, 5, 5]}

    max_unique_count = 0
    result_key = None

    for key, value_list in test_dict.items():

        unique_count_dict = {}
        unique_count = 0

        for item in value_list:
            if item not in unique_count_dict:
                unique_count_dict[item] = 1
                unique_count += 1

        if unique_count > max_unique_count:
            max_unique_count = unique_count
            result_key = key

    return result_key


print("Key with the most unique values is: ", key_with_most_unique_values())
