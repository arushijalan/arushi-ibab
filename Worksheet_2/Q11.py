def find_max_min():
    my_dict = {'a': 10, 'b': 45, 'c': 5, 'd': 30}

    values = list(my_dict.values())
    
    max_value = values[0]
    min_value = values[0]

    for value in values:
        if value > max_value:
            max_value = value
        if value < min_value:
            min_value = value
    
    print("Maximum value:", max_value)
    print("Minimum value:", min_value)

find_max_min()
