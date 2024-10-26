def extract_elements():
    List = [1, 2, 3, 2, 4, 5, 3, 2, 6, 7, 8, 5, 3, 2]
    k = 2
    element_count = {}
    for item in List:
        if item in element_count:
            element_count[item] += 1
        else:
            element_count[item] = 1
    result = [item for item, count in element_count.items() if count > k]

    print("Elements that occur more than", k, "times:", result)
extract_elements()
