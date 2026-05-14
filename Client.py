"""
Cliente para multiplicacao de matrizes distribuida.

O cliente gera as matrizes A e B, divide A em blocos de linhas, envia cada bloco
para um servidor worker e concatena os resultados parciais para formar C = A x B.
Tambem executa a versao serial para comparar tempo e speedup.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


HEADER_FORMAT = "!Q"  # unsigned long long em network byte order
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_START_PORT = 5000
DEFAULT_SIZES = "20,100,500,1000"
CSV_COLUMNS = [
    "matrix_size",
    "serial_time_ms",
    "parallel_time_ms",
    "speedup",
    "num_servers",
    "timestamp",
]


@dataclass(frozen=True)
class ServerSpec:
    """Representa o endereco de um servidor worker."""

    host: str
    port: int

    @property
    def label(self) -> str:
        return f"{self.host}:{self.port}"


def send_pickle(sock: socket.socket, obj: Any) -> None:
    """Serializa e envia um objeto pelo socket com cabecalho de tamanho."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(HEADER_FORMAT, len(data)))
    sock.sendall(data)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Recebe exatamente size bytes do socket."""
    chunks: list[bytes] = []
    received = 0

    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("Conexao encerrada antes de receber todos os dados.")
        chunks.append(chunk)
        received += len(chunk)

    return b"".join(chunks)


def recv_pickle(sock: socket.socket) -> Any:
    """Recebe e desserializa um objeto enviado por send_pickle."""
    header = recv_exact(sock, HEADER_SIZE)
    (payload_size,) = struct.unpack(HEADER_FORMAT, header)
    payload = recv_exact(sock, payload_size)
    return pickle.loads(payload)


def parse_matrix_sizes(size: int | None, sizes: str) -> list[int]:
    """Interpreta --size ou --sizes e devolve uma lista de tamanhos."""
    if size is not None:
        if size <= 0:
            raise ValueError("--size deve ser maior que zero.")
        return [size]

    parsed_sizes = [int(item.strip()) for item in sizes.split(",") if item.strip()]
    if not parsed_sizes or any(matrix_size <= 0 for matrix_size in parsed_sizes):
        raise ValueError("--sizes deve conter inteiros positivos separados por virgula.")
    return parsed_sizes


def parse_server_list(server_list: str | None, host: str, start_port: int, num_servers: int) -> list[ServerSpec]:
    """Monta a lista de servidores a partir de --server-list ou portas sequenciais."""
    if server_list:
        servers: list[ServerSpec] = []
        for raw_item in server_list.split(","):
            item = raw_item.strip()
            if not item:
                continue

            if ":" not in item:
                raise ValueError("Cada servidor em --server-list deve estar no formato host:porta.")

            server_host, raw_port = item.rsplit(":", 1)
            servers.append(ServerSpec(server_host.strip(), int(raw_port)))

        if not servers:
            raise ValueError("--server-list nao pode estar vazio.")
        return servers

    if num_servers < 1:
        raise ValueError("--servers deve ser maior ou igual a 1.")

    return [ServerSpec(host, start_port + index) for index in range(num_servers)]


def generate_random_matrices(matrix_size: int, low: int, high: int, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Gera A e B aleatorias usando NumPy."""
    rng = np.random.default_rng(seed)
    matrix_a = rng.integers(low, high + 1, size=(matrix_size, matrix_size), dtype=np.int64)
    matrix_b = rng.integers(low, high + 1, size=(matrix_size, matrix_size), dtype=np.int64)
    return matrix_a, matrix_b


def split_matrix(matrix_a: np.ndarray, num_parts: int) -> list[np.ndarray]:
    """Divide A verticalmente em num_parts submatrizes com tamanhos equilibrados."""
    if num_parts > matrix_a.shape[0]:
        raise ValueError("O numero de servidores nao pode ser maior que o numero de linhas de A.")
    return list(np.array_split(matrix_a, num_parts, axis=0))


def multiply_serial(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple[np.ndarray, float]:
    """Executa C = A x B localmente e mede o tempo em milissegundos."""
    started_at = time.perf_counter()
    result = matrix_a @ matrix_b
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return result, elapsed_ms


def request_partial_result(
    index: int,
    server: ServerSpec,
    submatrix_a: np.ndarray,
    matrix_b: np.ndarray,
    timeout: float,
) -> tuple[int, np.ndarray, float]:
    """Envia uma submatriz para um servidor e recebe o resultado parcial."""
    task_id = f"parte-{index + 1}"

    with socket.create_connection((server.host, server.port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        send_pickle(
            sock,
            {
                "action": "multiply",
                "task_id": task_id,
                "submatrix_a": submatrix_a,
                "matrix_b": matrix_b,
            },
        )
        response = recv_pickle(sock)

    if response.get("status") != "ok":
        message = response.get("message", "erro desconhecido")
        raise RuntimeError(f"Servidor {server.label} falhou: {message}")

    return index, response["result"], float(response.get("server_time_ms", 0.0))


def multiply_distributed(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    servers: list[ServerSpec],
    timeout: float,
) -> tuple[np.ndarray, float, list[float]]:
    """Executa a multiplicacao distribuida e aguarda todos os servidores."""
    submatrices = split_matrix(matrix_a, len(servers))
    results: list[np.ndarray | None] = [None] * len(servers)
    server_times_ms = [0.0] * len(servers)

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
        futures = []
        for index, (server, submatrix) in enumerate(zip(servers, submatrices)):
            print(
                f"[CLIENT] Enviando submatriz {index + 1}/{len(servers)} "
                f"{submatrix.shape} para Servidor {index + 1} ({server.label})..."
            )
            futures.append(executor.submit(request_partial_result, index, server, submatrix, matrix_b, timeout))

        for future in as_completed(futures):
            index, partial_result, server_time_ms = future.result()
            results[index] = partial_result
            server_times_ms[index] = server_time_ms
            print(f"[CLIENT] Resultado parcial {index + 1} recebido.")

    elapsed_ms = (time.perf_counter() - started_at) * 1000

    if any(result is None for result in results):
        raise RuntimeError("Nem todos os servidores retornaram resultado.")

    final_result = np.vstack([result for result in results if result is not None])
    return final_result, elapsed_ms, server_times_ms


def append_benchmark_result(csv_path: Path, row: dict[str, Any]) -> None:
    """Salva uma linha de benchmark no CSV, criando cabecalho quando necessario."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_benchmark_for_size(
    matrix_size: int,
    servers: list[ServerSpec],
    args: argparse.Namespace,
    seed: int | None,
) -> dict[str, Any]:
    """Executa benchmark serial e distribuido para um tamanho de matriz."""
    print(f"\n[CLIENT] Gerando matrizes A ({matrix_size}x{matrix_size}) e B ({matrix_size}x{matrix_size})...")
    matrix_a, matrix_b = generate_random_matrices(matrix_size, args.low, args.high, seed)

    if args.print_matrices and matrix_size <= 10:
        print("[CLIENT] Matriz A:")
        print(matrix_a)
        print("[CLIENT] Matriz B:")
        print(matrix_b)

    serial_result, serial_time_ms = multiply_serial(matrix_a, matrix_b)
    distributed_result, parallel_time_ms, server_times_ms = multiply_distributed(
        matrix_a,
        matrix_b,
        servers,
        args.timeout,
    )

    print("[CLIENT] Resultado recebido de todos os servidores.")

    if not args.skip_validation and not np.array_equal(distributed_result, serial_result):
        raise AssertionError("A matriz distribuida nao corresponde ao resultado serial.")

    speedup = serial_time_ms / parallel_time_ms if parallel_time_ms > 0 else 0.0
    row = {
        "matrix_size": matrix_size,
        "serial_time_ms": round(serial_time_ms, 4),
        "parallel_time_ms": round(parallel_time_ms, 4),
        "speedup": round(speedup, 4),
        "num_servers": len(servers),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    append_benchmark_result(Path(args.output), row)

    print("[CLIENT] Matriz resultante C montada com sucesso!")
    print(
        f"[CLIENT] Tempo Serial: {serial_time_ms:.2f}ms | "
        f"Tempo Distribuido: {parallel_time_ms:.2f}ms | Speedup: {speedup:.2f}x"
    )
    print(
        "[CLIENT] Tempos internos dos servidores: "
        + ", ".join(f"{time_ms:.2f}ms" for time_ms in server_times_ms)
    )
    print(f"[CLIENT] Resultados salvos em {args.output}")

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cliente para multiplicacao de matrizes distribuida.")
    parser.add_argument("--size", type=int, default=None, help="Executa apenas um tamanho n para matrizes n x n.")
    parser.add_argument(
        "--sizes",
        default=DEFAULT_SIZES,
        help=f"Tamanhos separados por virgula para benchmark. Padrao: {DEFAULT_SIZES}.",
    )
    parser.add_argument("--servers", type=int, default=2, help="Numero de servidores usando portas sequenciais.")
    parser.add_argument("--server-host", default=DEFAULT_HOST, help=f"Host dos servidores. Padrao: {DEFAULT_HOST}.")
    parser.add_argument(
        "--start-port",
        type=int,
        default=DEFAULT_START_PORT,
        help=f"Primeira porta dos servidores sequenciais. Padrao: {DEFAULT_START_PORT}.",
    )
    parser.add_argument(
        "--server-list",
        default=None,
        help="Lista customizada no formato host:porta,host:porta. Ex: 127.0.0.1:5000,127.0.0.1:5001",
    )
    parser.add_argument("--low", type=int, default=-10, help="Menor valor aleatorio das matrizes.")
    parser.add_argument("--high", type=int, default=10, help="Maior valor aleatorio das matrizes.")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade. Use vazio removendo o valor.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout em segundos para cada conexao.")
    parser.add_argument("--output", default="benchmark_results.csv", help="Arquivo CSV de saida.")
    parser.add_argument("--print-matrices", action="store_true", help="Mostra matrizes quando n <= 10.")
    parser.add_argument("--skip-validation", action="store_true", help="Nao compara o resultado distribuido com o serial.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = parse_matrix_sizes(args.size, args.sizes)
    servers = parse_server_list(args.server_list, args.server_host, args.start_port, args.servers)

    print("[CLIENT] Servidores configurados: " + ", ".join(server.label for server in servers))

    for run_index, matrix_size in enumerate(sizes):
        seed = None if args.seed is None else args.seed + run_index
        run_benchmark_for_size(matrix_size, servers, args, seed)


if __name__ == "__main__":
    main()
