import numpy as np

def dominant_eigenvalue(matrix, max_iterations=1000, tolerance=1e-10):
    n, m = matrix.shape
    assert n == m, "Matrix must be square"
    b_k = np.random.rand(n)
    for _ in range(max_iterations):
        b_k1 = np.dot(matrix, b_k)
        
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        if np.linalg.norm(np.dot(matrix, b_k) - b_k1_norm * b_k) < tolerance:
            break
    dominant_value = np.dot(b_k.T, np.dot(matrix, b_k)) / np.dot(b_k.T, b_k)
    return dominant_value

def inverse_matrix(matrix):
    dominant_value = dominant_eigenvalue(matrix)
    
    if dominant_value == 0:
        raise ValueError("Matrix is singular and cannot be inverted")
    
    inverse = np.linalg.inv(matrix)
    
    return inverse
matrix = np.array([[4, 2], [3, 1]])


dom_value = dominant_eigenvalue(matrix)
print("Dominant Eigenvalue:", dom_value)

inverse_mat = inverse_matrix(matrix)
print("Inverse Matrix:\n", inverse_mat)



