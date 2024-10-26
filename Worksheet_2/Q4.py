def find_duplicates():
    
    L = [1, 2, 3, 4, 2, 5, 6, 3, 7, 8, 1]

    duplicates = []

    L.sort() 

    for i in range(1, len(L)):
        if L[i] == L[i - 1] and L[i] not in duplicates:
            duplicates.append(L[i])

    print("Duplicate elements:", duplicates)

find_duplicates()

