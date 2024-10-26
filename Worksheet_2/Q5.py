m1 = [[9, 3, 6], [5, 2, 7], [7, 5, 3]]
m2 = [[8, 4, 3], [3, 1, 4], [7, 2, 6]]

mat_diff = []
for row in range(len(m1)):
    mat_diff.append([])
    for column in range(len(m1[0])):
        mat_diff[row].append(m1[row][column] - m2[row][column])
print(mat_diff)