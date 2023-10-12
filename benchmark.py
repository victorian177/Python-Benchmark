import timeit

import numpy as np
import scipy.linalg as la

np.random.seed(42)


class Benchmark:
    def __init__(self, size=10000):
        self.size = size
        num_flops_LU = (2 / 3) * (self.size**3)
        num_flops_other = 2 * (self.size**2)
        self.num_flops = num_flops_LU + num_flops_other

    def run(self):
        execution_time = timeit.timeit(self.benchmark, number=1)
        mflops = self.num_flops / execution_time
        print(f"Time spent: {execution_time:.6f} seconds")
        print(f"FLOPs: {mflops:.3e}")

    def benchmark(self):
        for _ in range(10):
            self.linear_system_solving()

    def generate_matrix(self):
        A = np.random.randn(self.size, self.size)
        b = np.sum(A, axis=0).reshape(-1, 1)
        return A, b

    def linear_system_solving(self):
        A, b = self.generate_matrix()
        lu, piv = la.lu_factor(A)
        _ = la.lu_solve((lu, piv), b)


if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run()
