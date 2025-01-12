import random
x = [[random.randint(5,10) for i in range(0,3)] for j in range(2)]
print("x: ", x)
h=[]
for i in range(2):
    h.append((x[i][0]*5)+(3*(x[i][1]))+(2*x[i][2]))
print("Hypothesis values: ",h)
y=70
E=0
for j in range(len(h)):
    E=E+((h[j]-y)**2)
E=0.5*E
print("Error detected: ", E)