# Matrix Multiplication Client-Server

This project implements parallel matrix multiplication using a client-server architecture in Python.

## Structure

- **Client.py**: Generates matrices A and B, splits A into submatrices, sends them to servers, receives partial results, and concatenates them into the final matrix.
- **Server.py**: Receives a submatrix of A and the full matrix B from the client, performs the multiplication, and sends the partial result back.

## Requirements

- Python 3.x
- numpy

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Start the servers on different ports (for example 4 servers):

   ```bash
   python Server.py 5001
   python Server.py 5002
   python Server.py 5003
   python Server.py 5004
   ```

2. Run the client:

   ```bash
   python Client.py --size 4 --servers 4 --start-port 5001
   ```

The client will send each row block of matrix A to a different server, receive partial results, and compose the final matrix C.

## Configuration

- `--size`: Matrix size `n` for `n x n` random matrices.
- `--servers`: Number of servers to use.
- `--start-port`: First port number for servers.
- `--low` / `--high`: Range for random integer values in matrices.
- `--seed`: Optional random seed for reproducible results.
- `matrix_analysis.ipynb`: Use this notebook to run benchmarks, generate gráficos e exportar resultados para CSV.

## Notebook Analysis

- `matrix_analysis.ipynb`: interactive notebook for training, performance comparison, and visualization.
- The notebook includes serial vs parallel timing, charts, and tables for different matrix sizes.
- Use the notebook to compare random matrix runs and example matrix results.

## Notes

- The client performs a handshake with each server: it sends the task, receives an acknowledgement, and then receives the partial result.
- The final matrix is built by concatenating the partial results from each server.
- This structure matches a simple distributed client-server architecture for matrix multiplication.