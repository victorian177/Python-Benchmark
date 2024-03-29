import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.linalg as la
from ascii_graph import Pyasciigraph
from tabulate import SEPARATING_LINE, tabulate

np.random.seed(42)


# Options
class OperationType(Enum):
    FLOAT = "float"
    INTEGER = "integer"


class PrecisionType(Enum):
    SINGLE = np.float32
    DOUBLE = np.float64


class ConcurrencyLevel(Enum):
    ONE = 1
    TWO = 2
    FOUR = 4
    EIGHT = 8


class Benchmark:
    # Default values
    DEFAULT_SIZE = 1_000
    DEFAULT_OPERATION_TYPE = OperationType.FLOAT
    DEFAULT_PRECISION_TYPE = PrecisionType.DOUBLE
    DEFAULT_SCALE = (1, 5)
    DEFAULT_ITERATIONS = 10
    DEFAULT_RUNS = 3
    DEFAULT_CONCURRENCY = ConcurrencyLevel.ONE

    TIMEOUT = 3  # Times between benchmark runs
    SAVE_DIR = Path("saves")  # Directory to save settings and results

    def __init__(self):
        """
        Initialize a benchmark object.

        Keyword Args:
            size (int): The size of the matrix.
            operation_type (OperationType): Type of matrix values (float or integer).
            precision_type (PrecisionType): Precision of matrix values (single or double).
            scale (tuple): Range for random values generation.
            iterations (int): Number of benchmark iterations.
            concurrency (ConcurrencyLevel): Concurrency level.
            runs (int): Number of times to run benchmark.
        """

        kwargs = self.get_kwargs()
        self.size = kwargs.get("size", self.DEFAULT_SIZE)
        self.operation_type = kwargs.get("operation_type", self.DEFAULT_OPERATION_TYPE)
        self.precision_type = kwargs.get("precision_type", self.DEFAULT_PRECISION_TYPE)
        self.scale = kwargs.get("scale", self.DEFAULT_SCALE)
        self.iterations = kwargs.get("iterations", self.DEFAULT_ITERATIONS)
        self.concurrency = kwargs.get("concurrency", self.DEFAULT_CONCURRENCY)
        self.runs = kwargs.get("runs", self.DEFAULT_RUNS)

        # Calculate total operations based on matrix size and iterations
        num_ops_LU = (2 / 3) * (
            self.size**3
        )  # The upper bound on the number of operations carried out during the LU factorisation.
        num_ops_other = 2 * (
            self.size**2
        )  # The upper bound on the number of operations carried out during the solving for x.
        self.total_ops = self.iterations * (num_ops_LU + num_ops_other)

        # Create settings dictionary for display and saving
        precision_value = (
            "Single(32 Bit)"
            if self.precision_type == PrecisionType.SINGLE
            else "Double(64 Bit)"
        )
        operation_value = (
            "Float" if self.operation_type == OperationType.FLOAT else "Integer"
        )
        concurrency_value = self.concurrency.value

        self.system_information = {
            "OS": platform.system(),
            "Release": platform.release(),
            "CPU Information": platform.processor(),
            "CPU Count(Logical)": psutil.cpu_count(logical=True),
            "CPU Count(Physical)": psutil.cpu_count(logical=True),
            "RAM": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
        }

        self.settings = {
            "Size": self.size,
            "Precision Type": precision_value,
            "Operation Type": operation_value,
            "Scale": self.scale,
            "Iterations": self.iterations,
            "Concurrency": concurrency_value,
            "Runs": self.runs,
        }

        self.results = {
            "Time spent (seconds)": [],
            "[Giga] Operations per second": [],
            "[Giga] OPS per thread": [],
        }

        self.init_save_dir()

    def init_save_dir(self):
        """
        Initialize the save directory if it does not exist.
        """
        if not os.path.exists(self.SAVE_DIR):
            os.mkdir(self.SAVE_DIR)

    def generate_matrix(self):
        """
        Generate random matrices based on specified parameters.

        Returns:
            tuple: A tuple containing the generated matrix A and vector b.
        """
        # Determine the data type based on precision_type
        A = np.random.randn(self.size, self.size).astype(self.precision_type.value)
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
        self.display_system_information()
        self.display_settings()
        for _ in range(self.runs):
            # Measure the execution time for the benchmark
            start_time = time.perf_counter_ns()

            with ThreadPoolExecutor(max_workers=self.concurrency.value) as executor:
                futures = [
                    executor.submit(self.benchmark) for _ in range(self.iterations)
                ]

                # Wait for all futures to complete
                for future in futures:
                    future.result()

            end_time = time.perf_counter_ns()
            execution_time = end_time - start_time
            ops = self.total_ops / execution_time
            ops_per_thread = ops / self.concurrency.value

            time.sleep(self.TIMEOUT)

            self.results["Time spent (seconds)"].append(execution_time / 1e9)
            self.results["[Giga] Operations per second"].append(ops)
            self.results["[Giga] OPS per thread"].append(ops_per_thread)

        self.display_results()
        self.save()

    def save(self):
        """
        Save the benchmark settings and results to a specified directory.
        """
        # Create a directory based on the current timestamp
        save_dir = self.SAVE_DIR / datetime.now().strftime("%d%m%y_%H%M%S")
        os.mkdir(save_dir)

        # Save settings and results as JSON files
        with open(save_dir / "settings.json", "w") as settings_file:
            json.dump(self.settings, settings_file)

        with open(save_dir / "results.json", "w") as results_file:
            json.dump(self.results, results_file)

        with open(save_dir / "system_information.json", "w") as system_information_file:
            json.dump(self.system_information, system_information_file)

        print("\n\nData has been saved to 'saves' folder.")

    def display_system_information(self):
        """
        Display system information
        """
        print("\nSYSTEM INFORMATION\n")
        system_information_to_display = [["Information", "Value"]] + [
            [k, v] for k, v in self.system_information.items()
        ]
        print(tabulate(system_information_to_display, "firstrow"))

    def display_settings(self):
        """
        Display the benchmark settings.
        """
        print("\nSETTINGS\n")

        settings_to_display = [["Setting", "Value"]] + [
            [k, v] for k, v in self.settings.items()
        ]
        print(tabulate(settings_to_display, "firstrow"))

    def display_results(self):
        """
        Display the benchmark results including averages and standard deviations.
        """
        print("\nRESULTS\n")

        results_to_display = [["Index"] + list(self.results.keys())]
        for i in range(self.runs):
            data = [i + 1]
            for k in self.results:
                data.append(self.results[k][i])
            results_to_display.append(data)

        results_to_display.append(SEPARATING_LINE)

        means = ["Average"]
        stds = ["Standard deviation"]
        for v in self.results.values():
            means.append(np.mean(v))
            stds.append(np.std(v))

        results_to_display.append(means)
        results_to_display.append(stds)

        print(tabulate(results_to_display, headers="firstrow"))

        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        index = 0
        for k, v in self.results.items():
            x = [i + 1 for i in range(len(v))]
            axes[index].bar(
                x,
                v,
                width=0.5,
                align="center",
                color="blue",
                alpha=0.7,
                edgecolor="black",
                linewidth=1.2,
            )

            # Adding labels and a title
            axes[index].set_xlabel("Runs")
            axes[index].set_ylabel("Values")
            axes[index].set_title(k)
            
            index += 1

        plt.show()

    def get_kwargs(self):
        kwargs = {}

        print(
            f"""
    Here is an implementation of the LinPack benchmarker in Python.
    Here are the options that are to be entered:

    Size:
        - The size of the matrix to be solved. 
        - Value inputted should be greater than or equal to {self.DEFAULT_SIZE}. 
        DEFAULT: {self.DEFAULT_SIZE}
        
    Operation Type:
        - The type of operation to be carried out. 
        - Enter 'f' for float and 'i' for int. 
        DEFAULT: float

    Precision Type:
        - The level of precision that should be computed. 
        - Enter 's' for Single and 'd' for Double.
        DEFAULT: Double
        
    Scale:
        - The range of values that the elements matrix would take.
        - The format for entry is: lower, upper where (upper > lower).
        - An example input is: 10, 100.
        DEFAULT: {self.DEFAULT_SCALE}

    Iterations:
        - The number of times to run the benchmark.
        - Value should not be less than {self.DEFAULT_ITERATIONS}.
        DEFAULT: {self.DEFAULT_ITERATIONS}
        
    Concurrency:
        - The number of threads to execute the benchmark on.
        - The options available are 1, 2, 4, and 8.
        DEFAULT: 1

    Runs:
        - The number of times to run the benchmark.
        - Value should be greater than or equal to {self.DEFAULT_RUNS}.
        DEFAULT: {self.DEFAULT_RUNS}
        
    NOTE: If an invalid input is supplied, it reverts to the default.
            """
        )

        should_escape = input(
            "Press 'Enter' to proceed (or 'd' to use default settings): "
        )
        if should_escape != "d":
            try:
                size = int(input("Size: "))
                if size >= self.DEFAULT_SIZE:
                    kwargs["size"] = size
                else:
                    pass
            except ValueError:
                pass

            operation_type = input("Operation Type: ")
            if operation_type == "f":
                kwargs["operation_type"] = OperationType.FLOAT
            elif operation_type == "i":
                kwargs["operation_type"] = OperationType.INTEGER
            else:
                pass

            precision_type = input("Precision Type: ")
            if precision_type == "s":
                kwargs["precision_type"] = PrecisionType.SINGLE
            elif precision_type == "d":
                kwargs["precision_type"] = PrecisionType.DOUBLE
            else:
                pass

            try:
                scale = input("Scale: ")
                lower, upper = map(int, scale.split(","))
                if upper > lower:
                    kwargs["scale"] = (lower, upper)
                else:
                    pass
            except (ValueError, IndexError):
                pass

            try:
                iterations = int(input("Iterations: "))
                if iterations > self.DEFAULT_ITERATIONS:
                    kwargs["iterations"] = iterations
                else:
                    pass

            except ValueError:
                pass

            try:
                concurrency = int(input("Concurrency: "))
                concurrency = int(concurrency)
                if concurrency == 1:
                    kwargs["concurrency"] = ConcurrencyLevel.ONE
                elif concurrency == 2:
                    kwargs["concurrency"] = ConcurrencyLevel.TWO
                elif concurrency == 4:
                    kwargs["concurrency"] = ConcurrencyLevel.FOUR
                elif concurrency == 8:
                    kwargs["concurrency"] = ConcurrencyLevel.EIGHT
                else:
                    pass

            except ValueError:
                pass

            try:
                runs = int(input("Runs: "))
                if runs >= self.DEFAULT_RUNS:
                    kwargs["runs"] = runs
                else:
                    pass

            except ValueError:
                pass

        return kwargs


if __name__ == "__main__":
    from pyfiglet import Figlet

    f = Figlet(font="banner3-D", width=200)
    print(f.renderText("The Python LinPack-Based Benchmark"))
    b = Benchmark()
    b.run()
