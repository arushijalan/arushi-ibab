import random

L = [random.randint(0, 10) for i in range(0,15)]
print(L)

even = []
for i in L:
    if L[i] % 2 == 0:
        even.append(L[i])
    else:
        continue
print("The even numbers are:", even)