import numpy as np

A = np.matrix([[1,2,3], [4,5,6]])
print(f'The original matrix is:\n',A)
AT = A.transpose()
print(f'The transpose of the matrix is:\n',AT)
#First way
Result = AT * A
print(f'The product of matrix and its transpose is:\n',Result)
#Second way
a = AT@A
print(f'The product of matrix and its transpose is:\n',a)
#Third way
b = AT.__matmul__(A)
print(f'The product of matrix and its transpose is:\n',b)