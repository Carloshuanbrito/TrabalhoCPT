"""
Servidor para multiplicacao de matrizes distribuida.

Cada instancia deste arquivo representa um worker/servidor. O cliente envia uma
submatriz de A e a matriz B completa; o servidor multiplica as duas em paralelo
internamente e devolve o resultado parcial.
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np


HEADER_FORMAT = "!Q"  # unsigned long long em network byte order
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def send_pickle(sock: socket.socket, obj: Any) -> None:
    """Serializa um objeto com pickle e envia com cabecalho de tamanho fixo."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(HEADER_FORMAT, len(data)))
    sock.sendall(data)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Recebe exatamente size bytes ou levanta erro se a conexao fechar antes."""
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
    """Recebe um objeto serializado por send_pickle."""
    header = recv_exact(sock, HEADER_SIZE)
    (payload_size,) = struct.unpack(HEADER_FORMAT, header)
    payload = recv_exact(sock, payload_size)
    return pickle.loads(payload)


def multiply_chunk(chunk: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Multiplica um bloco de linhas de A pela matriz B."""
    return chunk @ matrix_b


def multiply_submatrix_parallel(
    submatrix_a: np.ndarray,
    matrix_b: np.ndarray,
    max_workers: int,
) -> np.ndarray:
    """Multiplica submatrix_a por matrix_b usando threads internas.

    A divisao em blocos evita criar uma tarefa por linha, o que seria custoso em
    matrizes grandes. O NumPy faz o calculo pesado em codigo nativo e as threads
    ajudam a dividir o trabalho entre blocos de linhas.
    """
    if submatrix_a.ndim != 2 or matrix_b.ndim != 2:
        raise ValueError("As matrizes recebidas devem ser bidimensionais.")

    if submatrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError(
            "Dimensoes incompatíveis: "
            f"A_sub tem {submatrix_a.shape[1]} colunas e B tem {matrix_b.shape[0]} linhas."
        )

    if submatrix_a.shape[0] == 0:
        return np.empty((0, matrix_b.shape[1]), dtype=np.result_type(submatrix_a, matrix_b))

    workers = max(1, min(max_workers, submatrix_a.shape[0]))
    chunks = [chunk for chunk in np.array_split(submatrix_a, workers, axis=0) if len(chunk) > 0]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        partial_results = list(executor.map(lambda chunk: multiply_chunk(chunk, matrix_b), chunks))

    return np.vstack(partial_results)


def handle_client(
    conn: socket.socket,
    addr: tuple[str, int],
    internal_workers: int,
    server_label: str,
) -> None:
    """Atende uma conexao de cliente e devolve o resultado parcial."""
    thread_name = threading.current_thread().name

    with conn:
        try:
            print(f"[{server_label}] Cliente {addr} conectado em {thread_name}.")
            request = recv_pickle(conn)

            if request.get("action") != "multiply":
                raise ValueError("Acao invalida. Esperado: 'multiply'.")

            task_id = request.get("task_id", "sem-id")
            submatrix_a = request["submatrix_a"]
            matrix_b = request["matrix_b"]

            print(
                f"[{server_label}] Recebendo dados da tarefa {task_id}: "
                f"A_sub={submatrix_a.shape}, B={matrix_b.shape}. Multiplicando..."
            )
            started_at = time.perf_counter()
            result = multiply_submatrix_parallel(submatrix_a, matrix_b, internal_workers)
            elapsed_ms = (time.perf_counter() - started_at) * 1000

            send_pickle(
                conn,
                {
                    "status": "ok",
                    "task_id": task_id,
                    "result": result,
                    "server_time_ms": elapsed_ms,
                    "server_label": server_label,
                },
            )
            print(
                f"[{server_label}] Tarefa {task_id} finalizada em {elapsed_ms:.2f} ms. "
                "Resultado enviado."
            )

        except Exception as exc:  # Envia erro estruturado para o cliente.
            try:
                send_pickle(conn, {"status": "error", "message": str(exc), "server_label": server_label})
            except Exception:
                pass
            print(f"[{server_label}] Erro ao atender {addr}: {exc}")


def run_server(host: str, port: int, internal_workers: int, max_clients: int, backlog: int) -> None:
    """Inicia o servidor e aceita multiplas conexoes simultaneas."""
    server_label = f"SERVER {port}"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(backlog)

        print(
            f"[{server_label}] Aguardando conexoes em {host}:{port} | "
            f"workers internos={internal_workers} | clientes simultaneos={max_clients}"
        )

        with ThreadPoolExecutor(max_workers=max_clients) as client_pool:
            while True:
                try:
                    conn, addr = server_socket.accept()
                    client_pool.submit(handle_client, conn, addr, internal_workers, server_label)
                except KeyboardInterrupt:
                    print(f"\n[{server_label}] Encerrando servidor...")
                    break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Servidor worker para multiplicacao distribuida.")
    parser.add_argument("port_positional", nargs="?", type=int, help="Porta do servidor (atalho posicional).")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host/IP para bind. Padrao: {DEFAULT_HOST}.")
    parser.add_argument("--port", type=int, default=None, help=f"Porta do servidor. Padrao: {DEFAULT_PORT}.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Numero de threads internas para multiplicar a submatriz.",
    )
    parser.add_argument(
        "--max-clients",
        type=int,
        default=8,
        help="Quantidade maxima de clientes/conexoes atendidos simultaneamente.",
    )
    parser.add_argument("--backlog", type=int, default=16, help="Tamanho da fila de conexoes pendentes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    port = args.port if args.port is not None else (args.port_positional or DEFAULT_PORT)

    if args.workers < 1:
        raise ValueError("--workers deve ser maior ou igual a 1.")
    if args.max_clients < 1:
        raise ValueError("--max-clients deve ser maior ou igual a 1.")

    run_server(args.host, port, args.workers, args.max_clients, args.backlog)


if __name__ == "__main__":
    main()
