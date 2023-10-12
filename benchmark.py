import timeit
from enum import Enum
from typing import Tuple

import numpy as np
import scipy.linalg as la

# Set a random seed for reproducibility
np.random.seed(42)


# Enum definitions for operation and precision types
class OperationType(Enum):
    FLOAT = "float"
    INTEGER = "integer"


class PrecisionType(Enum):
    SINGLE = "single"
    DOUBLE = "double"


# Benchmarking class
class Benchmark:
    def __init__(
        self,
        size: int = 100,
        operation_type: OperationType = OperationType.FLOAT,
        precision_type: PrecisionType = PrecisionType.DOUBLE,
        scale: Tuple[int, int] = (1, 5),
        iterations=5,
    ):
        """
        Initialize a benchmark object.

        Args:
            size (int): The size of the matrix.
            operation_type (OperationType): Type of matrix values (float or integer).
            precision_type (PrecisionType): Precision of matrix values (single or double).
            scale (tuple): Range for random values generation.
            iterations (int): Number of benchmark iterations.
        """
        self.size = size
        self.operation_type = operation_type
        self.precision_type = precision_type
        self.scale = scale
        self.iterations = iterations

        # Calculate the total number of FLOPs for benchmarking
        num_flops_LU = (2 / 3) * (self.size**3)
        num_flops_other = 2 * (self.size**2)
        self.num_flops = num_flops_LU + num_flops_other

    def generate_matrix(self):
        """
        Generate random matrices based on specified parameters.

        Returns:
            tuple: A tuple containing the generated matrix A and vector b.
        """
        # Determine the data type based on precision_type
        if self.precision_type == PrecisionType.DOUBLE:
            dtype = np.float64
        else:
            dtype = np.float32

        A = np.random.randn(self.size, self.size).astype(dtype)
        A = A * (self.scale[1] - self.scale[0]) + self.scale[0]

        # If operation_type is INTEGER, round the values and cast to int
        if self.operation_type == OperationType.INTEGER:
            A = np.round(A).astype(int)
        b = np.sum(A, axis=0).reshape(-1, 1)

        return A, b

    def benchmark(self):
        """
        Perform LU decomposition benchmark on generated matrices.
        """
        # Generate matrices and perform LU decomposition
        A, b = self.generate_matrix()
        lu, piv = la.lu_factor(A)
        _ = la.lu_solve((lu, piv), b)

    def run(self):
        """
        Run the benchmark and print the results.
        """
        # Measure the execution time for the benchmark
        execution_time = timeit.timeit(self.benchmark, number=self.iterations)

        # Calculate FLOPs and print results
        flops = self.num_flops / execution_time
        print("Settings:")
        print(f"Size: {self.size}")
        print(f"Precision Type: {self.precision_type}")
        print(f"Operation Type: {self.operation_type}")
        print(f"Iterations: {self.iterations}")
        print(f"Scale: {self.scale}")
        print()

        print(f"Time spent: {execution_time:.6f} seconds")
        print(f"FLOPs: {flops:.3e}")
        print("------------------------------------------")


if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run()
    benchmark.operation_type = OperationType.INTEGER
    benchmark.run()
