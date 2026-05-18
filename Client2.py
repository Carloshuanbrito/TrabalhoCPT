# Client.py COMPLETO adaptado para matrizes não quadradas


"""
Cliente do PROJETOCP - multiplicacao de matrizes distribuida.

Fluxo:
1. Gera matrizes A e B aleatoriamente com NumPy.
2. Executa a multiplicacao serial local para comparacao.
3. Divide A em submatrizes e envia para a quantidade de servidores informada.
4. Recebe os resultados parciais, monta C e salva os tempos no CSV.
5. Salva as matrizes da ultima execucao em last_run_matrices.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# Configuracoes obrigatorias do trabalho
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_START_PORT = 5000
DEFAULT_NUM_SERVERS = 2

MATRIX_SIZES = [20, 50, 100, 200, 500, 1000]

DTYPE = np.int32
VALUE_RANGE = (1, 10)

CSV_PATH = Path("benchmark_results.csv")
LAST_RUN_JSON_PATH = Path("last_run_matrices.json")

CSV_COLUMNS = [
    "matrix_size",
    "serial_time_ms",
    "parallel_time_ms",
    "speedup",
    "num_servers",
    "timestamp",
]

HEADER_FORMAT = "!Q"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
SOCKET_TIMEOUT_SECONDS = 300
SOCKET_CHUNK_SIZE = 1024 * 1024
SERVER_REQUEST_RETRIES = 2
SEPARATOR = "=" * 60


def send_pickle(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(HEADER_FORMAT, len(payload)))

    for offset in range(0, len(payload), SOCKET_CHUNK_SIZE):
        sock.sendall(payload[offset : offset + SOCKET_CHUNK_SIZE])


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    received = 0

    while received < size:
        chunk_size = min(SOCKET_CHUNK_SIZE, size - received)
        chunk = sock.recv(chunk_size)

        if not chunk:
            raise ConnectionError(
                "A conexao foi encerrada antes do recebimento completo."
            )

        chunks.append(chunk)
        received += len(chunk)

    return b"".join(chunks)


def recv_pickle(sock: socket.socket) -> Any:
    header = recv_exact(sock, HEADER_SIZE)
    (payload_size,) = struct.unpack(HEADER_FORMAT, header)
    payload = recv_exact(sock, payload_size)
    return pickle.loads(payload)


def generate_random_matrices(
    matrix_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera:

    A = matrix_size x matrix_size
    B = matrix_size x matrix_size
    """

    matrix_a = np.random.randint(
        VALUE_RANGE[0],
        VALUE_RANGE[1],
        size=(matrix_size, matrix_size),
        dtype=DTYPE,
    )

    matrix_b = np.random.randint(
        VALUE_RANGE[0],
        VALUE_RANGE[1],
        size=(matrix_size, matrix_size),
        dtype=DTYPE,
    )

    return matrix_a, matrix_b


def should_print_full_matrix(rows: int, cols: int) -> bool:
    return rows <= 50 and cols <= 50


def matrix_to_display_text(matrix: np.ndarray) -> str:
    rows, cols = matrix.shape

    if should_print_full_matrix(rows, cols):
        return str(matrix)

    preview = matrix[:5, :5]

    return (
        f"{preview}\n"
        "... (mostrando apenas as primeiras 5 linhas e 5 colunas)"
    )


def print_matrix(title: str, matrix: np.ndarray) -> None:
    print(title)
    print(matrix_to_display_text(matrix))
    print()


def split_matrix(
    matrix_a: np.ndarray,
    num_servers: int,
) -> list[np.ndarray]:

    if num_servers < 1:
        raise ValueError(
            "A quantidade de servidores deve ser maior ou igual a 1."
        )

    if num_servers > matrix_a.shape[0]:
        raise ValueError(
            "A quantidade de servidores nao pode ser maior que o numero de linhas da matriz."
        )

    return list(np.array_split(matrix_a, num_servers, axis=0))


def multiply_serial(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
) -> tuple[np.ndarray, float]:

    start = time.perf_counter()

    result = np.dot(matrix_a, matrix_b)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return result, elapsed_ms


def request_server_multiplication(
    server_index: int,
    server_address: tuple[str, int],
    submatrix_a: np.ndarray,
    matrix_b: np.ndarray,
    results: list[np.ndarray | None],
    server_times_ms: list[float],
    errors: list[Exception],
) -> None:

    host, port = server_address

    for attempt in range(1, SERVER_REQUEST_RETRIES + 1):
        try:
            with socket.create_connection(
                (host, port),
                timeout=SOCKET_TIMEOUT_SECONDS,
            ) as sock:

                sock.settimeout(SOCKET_TIMEOUT_SECONDS)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

                send_pickle(
                    sock,
                    {
                        "action": "multiply",
                        "server_index": server_index,
                        "submatrix_a": submatrix_a,
                        "matrix_b": matrix_b,
                    },
                )

                response = recv_pickle(sock)

            if response.get("status") != "ok":
                raise RuntimeError(
                    response.get(
                        "message",
                        "Erro desconhecido no servidor.",
                    )
                )

            results[server_index] = response["result"]
            server_times_ms[server_index] = float(response["elapsed_ms"])
            return

        except Exception as exc:
            if attempt < SERVER_REQUEST_RETRIES:
                print(
                    f"[CLIENTE] Servidor {server_index + 1} ({host}:{port}) "
                    f"falhou na tentativa {attempt}; tentando novamente..."
                )
                time.sleep(1)
                continue

            errors.append(
                RuntimeError(
                    f"Servidor {server_index + 1} ({host}:{port}) falhou "
                    f"apos {SERVER_REQUEST_RETRIES} tentativa(s), "
                    f"com submatriz {submatrix_a.shape} e B {matrix_b.shape}: {exc}"
                )
            )
            return


def multiply_distributed(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    servers: list[tuple[str, int]],
) -> tuple[np.ndarray, float, list[np.ndarray], list[float]]:

    submatrices = split_matrix(matrix_a, len(servers))

    row_counts = [
        submatrix.shape[0]
        for submatrix in submatrices
    ]

    row_summary = ", ".join(
        str(row_count)
        for row_count in row_counts
    )

    print(
        f"[CLIENTE] Dividindo A em {len(servers)} submatrizes "
        f"(linhas por servidor: {row_summary})...\n"
    )

    partial_results: list[np.ndarray | None] = [None] * len(servers)
    server_times_ms = [0.0] * len(servers)

    errors: list[Exception] = []
    threads: list[threading.Thread] = []

    start = time.perf_counter()

    for index, (server, submatrix) in enumerate(
        zip(servers, submatrices)
    ):

        print(
            f"[CLIENTE] Enviando submatriz para "
            f"Servidor {index + 1} ({server[0]}:{server[1]})..."
        )

        thread = threading.Thread(
            target=request_server_multiplication,
            args=(
                index,
                server,
                submatrix,
                matrix_b,
                partial_results,
                server_times_ms,
                errors,
            ),
            daemon=False,
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    elapsed_ms = (time.perf_counter() - start) * 1000

    if errors:
        raise RuntimeError(
            f"Erro durante a execucao distribuida: {errors[0]}"
        )

    if any(result is None for result in partial_results):
        raise RuntimeError(
            "Nem todos os servidores retornaram resultados parciais."
        )

    typed_results = [
        result
        for result in partial_results
        if result is not None
    ]

    final_matrix = np.vstack(typed_results)

    return (
        final_matrix,
        elapsed_ms,
        typed_results,
        server_times_ms,
    )


def append_benchmark_row(row: dict[str, Any]) -> None:
    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0

    with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_last_run_matrices(last_run: dict[str, Any]) -> None:

    with LAST_RUN_JSON_PATH.open("w", encoding="utf-8") as json_file:
        json.dump(
            last_run,
            json_file,
            ensure_ascii=False,
            indent=2,
        )


def run_test(
    matrix_size: int,
    servers: list[tuple[str, int]],
    last_run: dict[str, Any],
) -> None:

    print(SEPARATOR)

    print(
        f" TESTE: Matriz {matrix_size}x{matrix_size}"
    )

    print(SEPARATOR)
    print()

    matrix_a, matrix_b = generate_random_matrices(
        matrix_size,
    )

    print_matrix(
        f"[CLIENTE] Matriz A gerada ({matrix_size}x{matrix_size}):",
        matrix_a,
    )

    print_matrix(
        f"[CLIENTE] Matriz B gerada ({matrix_size}x{matrix_size}):",
        matrix_b,
    )

    serial_result, serial_time_ms = multiply_serial(
        matrix_a,
        matrix_b,
    )

    (
        distributed_result,
        parallel_time_ms,
        partial_results,
        server_times_ms,
    ) = multiply_distributed(
        matrix_a,
        matrix_b,
        servers,
    )

    if not np.array_equal(serial_result, distributed_result):
        raise AssertionError(
            "Resultado distribuido diferente do resultado serial."
        )

    for index, (
        partial_result,
        server_time_ms,
    ) in enumerate(
        zip(partial_results, server_times_ms),
        start=1,
    ):

        print(
            f"[SERVIDOR {index}] Recebeu submatriz A "
            f"({partial_result.shape[0]}x{matrix_a.shape[1]}) "
            f"e matriz B ({matrix_b.shape[0]}x{matrix_b.shape[1]})"
        )

        print(
            f"[SERVIDOR {index}] Multiplicando com threading... "
            f"concluido em {server_time_ms:.0f}ms\n"
        )

        print_matrix(
            f"[CLIENTE] Resultado parcial do Servidor {index}:",
            partial_result,
        )

    print_matrix(
        f"[CLIENTE] Matriz C final "
        f"({matrix_size}x{matrix_size}) montada com sucesso:",
        distributed_result,
    )

    speedup = (
        serial_time_ms / parallel_time_ms
        if parallel_time_ms > 0
        else 0.0
    )

    row = {
        "matrix_size": matrix_size,
        "serial_time_ms": round(serial_time_ms, 2),
        "parallel_time_ms": round(parallel_time_ms, 2),
        "speedup": round(speedup, 2),
        "num_servers": len(servers),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    append_benchmark_row(row)

    matrix_key = f"{matrix_size}x{matrix_size}"

    last_run[matrix_key] = {
        "matrix_size": matrix_size,
        "A": matrix_a.tolist(),
        "B": matrix_b.tolist(),
        "partial_results": [
            {
                "server": index,
                "rows": partial_result.shape[0],
                "cols": partial_result.shape[1],
                "matrix": partial_result.tolist(),
            }
            for index, partial_result in enumerate(
                partial_results,
                start=1,
            )
        ],
        "C": distributed_result.tolist(),
    }

    print(f"[CLIENTE] Tempo Serial:      {serial_time_ms:.0f} ms")
    print(f"[CLIENTE] Tempo Distribuido: {parallel_time_ms:.0f} ms")
    print(f"[CLIENTE] Speedup:           {speedup:.2f}x")

    if speedup < 1.0:
        outcome = "Serial vence"
    elif speedup < 1.2:
        outcome = "Transicao"
    else:
        outcome = "Distribuido vence"

    print(
        f"[{matrix_size}x{matrix_size}]".ljust(12)
        + f" Serial: {serial_time_ms:.0f}ms".ljust(18)
        + f"| Distribuido: {parallel_time_ms:.0f}ms".ljust(25)
        + f"| Speedup: {speedup:.2f}x <- {outcome}"
    )

    print(SEPARATOR)
    print()


def build_servers(
    num_servers: int,
    start_port: int,
    host: str,
) -> list[tuple[str, int]]:

    if num_servers < 1:
        raise ValueError(
            "--servers deve ser maior ou igual a 1."
        )

    return [
        (host, start_port + index)
        for index in range(num_servers)
    ]


def verify_servers(servers: list[tuple[str, int]]) -> None:
    """Confirma se todos os servidores configurados respondem antes do benchmark."""
    print("[CLIENTE] Verificando conexao com os servidores...")

    for index, (host, port) in enumerate(servers, start=1):
        try:
            with socket.create_connection(
                (host, port),
                timeout=10,
            ) as sock:
                sock.settimeout(10)
                send_pickle(sock, {"action": "ping"})
                response = recv_pickle(sock)

            if response.get("status") != "ok":
                raise RuntimeError(response)

            print(f"  Servidor {index}: {host}:{port} OK")

        except Exception as exc:
            raise RuntimeError(
                f"Servidor {index} ({host}:{port}) nao respondeu ao ping. "
                "Feche os servidores antigos e abra novamente com o Server.py atualizado. "
                f"Detalhe: {exc}"
            ) from exc

    print()


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Cliente do PROJETOCP."
    )

    parser.add_argument(
        "--servers",
        type=int,
        default=DEFAULT_NUM_SERVERS,
        help="Quantidade de servidores/portas que voce abriu.",
    )

    parser.add_argument(
        "--start-port",
        type=int,
        default=DEFAULT_START_PORT,
        help="Primeira porta dos servidores.",
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_SERVER_HOST,
        help="Host onde os servidores estao rodando.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    servers = build_servers(
        args.servers,
        args.start_port,
        args.host,
    )

    np.set_printoptions(linewidth=120)

    print("[CLIENTE] Servidores configurados:")

    for index, (host, port) in enumerate(servers, start=1):
        print(f"  Servidor {index}: {host}:{port}")

    print()
    verify_servers(servers)

    last_run: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "servers": [
            f"{host}:{port}"
            for host, port in servers
        ],
        "matrices": {},
    }

    for matrix_size in MATRIX_SIZES:

        run_test(
            matrix_size,
            servers,
            last_run["matrices"],
        )

    save_last_run_matrices(last_run)

    print(f"[CLIENTE] Resultados acrescentados em {CSV_PATH}")

    print(
        f"[CLIENTE] Matrizes da ultima execucao "
        f"salvas em {LAST_RUN_JSON_PATH}"
    )


if __name__ == "__main__":
    main()


