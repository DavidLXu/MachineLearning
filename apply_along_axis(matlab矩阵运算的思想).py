# apply_along_axis的用法
import numpy as np

def add(x):
    return x[0]+x[1]
x = np.array([[1,2],[3,4]])

print(np.apply_along_axis(add,0,x)) #
print(np.apply_along_axis(add,1,x))