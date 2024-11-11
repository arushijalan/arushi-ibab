numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares_dict = {num: num ** 2 for num in numbers if num % 2 == 0}
print(squares_dict)