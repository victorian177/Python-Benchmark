# Linpack Benchmark Implementation in Python

This repository contains a Python implementation of the Linpack benchmark, a widely-used measure of a computer's floating-point computing power. The Linpack benchmark measures the performance of a computer by solving a dense system of linear equations.

## Getting Started

To run the Linpack benchmark, follow these steps:

### Prerequisites

Before running the benchmark, make sure you have Python.

### Installation

1. Navigate to the project directory:

    ```bash
    cd Py Benchmark
    ```

2. Create a virtual environment to isolate dependencies:

    ```bash
    python -m venv .venv
    ```

3. Activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

4. Install the required packages from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Benchmark

To run the Linpack benchmark, simply execute the `benchmark.py` script:

```bash
python benchmark.py
