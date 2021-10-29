import numpy as np
import pandas as pd

x = np.array([
    [1],
    [2],
    [3]
])

print(5*x)

m = np.zeros((3,3))

print(m)

a,b = 2,3

x,y = np.array([[2],[3]]), np.array([[4],[5]])

print(a*x+y*b)

x,y = np.array([[-2],[2]]), np.array([[4],[-3]])

print(x.T @ y)


q = np.array([[1,2,3],[3,2,1],[2,3,1]])
w = np.array([[0,1,2],[2,1,0],[1,0,2]])

print(np.dot(q,w))



lst = ['Java','Python','C', 'C++', 'JavaScript', 'Swift', 'Go']
data = {'Name':['Tom','Joseph','Krish','John'],'Age':[20,21,19,18]}
df = pd.DataFrame(lst);
df = pd.DataFrame(data)

X = np.array(df)
X_1 = X[:-1]
X_2 = X[-2:]
print(X)
print("--------------------------")
print(X_1)
print("--------------------------")
print(X_2)