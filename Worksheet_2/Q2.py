import random

L = [random.randint(0, 10) for i in range(0,15)]
print(L)

maximum = 0
for i in L:
    if L[i] > maximum:
        maximum = L[i]
    else:
        continue
print("The maximum number in the list is: ", maximum)

minimum = 0
for i in L:
    if L[i] < minimum:
        minimum = L[i]
    else:
        continue 
print("The minimum number in the list is: ", minimum)