# %%
import numpy as np
from matplotlib import pyplot as plt
# Create array
x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array

# Array attributes
print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)

print("dtype:", x3.dtype)

print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")

# 1-d Array indexing
print(x1)
print(x1[0], x1[4])

# 2-d array indexing
print(x2)
print(x2[0, 0])

# value assgin
x2[0, 0] = 12
print(x2)

# array slicing
x1[3:]
x1[1:4]
x1[::2]
x1[1::2]
x1[::-1]
x1[4::-2]

# Multi dimensional array
x2[:2, :3]
x2[:3, ::2]
x2[::-1, ::-1]

# row column
x2[:, 0]
x2[0, :]

# Reshape
array = np.arange(64)
matrix = np.reshape(array, (8, 8))
cube = np.reshape(array, (4, 4, 4))
print(array.shape)
print(matrix.shape)
print(cube.shape)

array = np.arange(64)
matrix = np.arange(64).reshape((8, 8))
cube = np.arange(64).reshape((4, 4, 4))
print(array.shape)
print(matrix.shape)
print(cube.shape)

# Concatenate
# 1d array
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])

# 2d array
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])

np.concatenate([grid, grid])
np.concatenate([grid, grid], axis=1)

# use stack
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])

y = np.array([[99],
              [99]])
np.hstack([grid, y])

# Split
# 1d array
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

# 2d array
grid = np.arange(16).reshape((4, 4))

upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)

left, right = np.hsplit(grid, [2])
print(left)
print(right)

# np.ones(4), np.ones((4, 4))
# np.zeros(4), np.zeros((4, 4))
# np.random.randint(0, 5, 4), np.random.randint(0, 5, (4, 4))
# np.random.rand(4), np.random.rand(4, 4), plt.hist(np.random.rand(1000))
# np.random.randn(4), np.random.randn(4, 4), plt.hist(np.random.randn(1000))


# %% 
x = np.arange(10)
y = 2*x + 1
plt.plot(x, y, ".- ")
# %%
