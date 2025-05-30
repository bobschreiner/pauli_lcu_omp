# In your Python test script
import numpy as np
from pauli_lcu_omp.decomposition import pauli_coefficients
import os
import time

dim = np.power(2,14)# num_qubits = 15
# Set the number of threads for OpenMP
num_threads = 8
os.environ["OMP_NUM_THREADS"] = str(num_threads)
test_matrix = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
test_matrix = np.ascontiguousarray(test_matrix, dtype=np.complex128)

print(f"Calling pauli_coefficients_omp with dim={dim} and OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
time_start = time.time()
# Call the C function
pauli_coefficients(test_matrix)
time_end = time.time()
print(f"Time taken: {time_end - time_start:.4f} seconds")
# Print the result
print("Ran with OMP_NUM_THREADS=", os.environ.get('OMP_NUM_THREADS'))

print("Finished call to pauli_coefficients_omp")