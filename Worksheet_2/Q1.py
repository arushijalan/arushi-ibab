import random

L = [random.randint(0, 10) for i in range(0,15)]
print(L)

sum = 0
for i in L:
    sum = sum + L[i]
print("The sum of elements of list, L is: ", sum)

average = sum / len(L)
print("The average of elements of list, L is: ", average)