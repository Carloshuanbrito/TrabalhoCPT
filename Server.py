"""
Servidor do PROJETOCP - worker de multiplicacao distribuida.

Uso:
    python Server.py 5000

Cada conexao recebida e atendida em uma thread separada. O cliente envia uma
submatriz de A e a matriz B completa; o servidor calcula np.dot(sub_A, B) e
devolve o resultado parcial.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import threading
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np


HOST = "127.0.0.1"
HEADER_FORMAT = "!Q"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def send_pickle(sock: socket.socket, obj: Any) -> None:
    """Serializa e envia um objeto com cabecalho de tamanho."""
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(HEADER_FORMAT, len(payload)))
    sock.sendall(payload)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Recebe exatamente size bytes do socket."""
    chunks: list[bytes] = []
    received = 0

    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("A conexao foi encerrada antes do recebimento completo.")
        chunks.append(chunk)
        received += len(chunk)

    return b"".join(chunks)


def recv_pickle(sock: socket.socket) -> Any:
    """Recebe e desserializa um objeto enviado por send_pickle."""
    header = recv_exact(sock, HEADER_SIZE)
    (payload_size,) = struct.unpack(HEADER_FORMAT, header)
    payload = recv_exact(sock, payload_size)
    return pickle.loads(payload)


def handle_client(conn: socket.socket, addr: tuple[str, int], port: int) -> None:
    """Atende uma requisicao de multiplicacao enviada pelo cliente."""
    with conn:
        try:
            print(f"[SERVIDOR porta={port}] Conexao recebida de {addr}")
            request = recv_pickle(conn)

            if request.get("action") != "multiply":
                raise ValueError("Acao invalida recebida do cliente.")

            submatrix_a = request["submatrix_a"]
            matrix_b = request["matrix_b"]

            print(f"[SERVIDOR porta={port}] Submatriz recebida: shape={submatrix_a.shape}")

            def multiply_row(row, matrix_b):
                return np.dot(row, matrix_b)
            
            start = time.perf_counter()

            with ThreadPoolExecutor() as executor:
                rows = list(executor.map(lambda row: multiply_row(row, matrix_b), submatrix_a))
            
            result = np.array(rows)
           
            elapsed_ms = (time.perf_counter() - start) * 1000

            print(
                f"[SERVIDOR porta={port}] Multiplicacao concluida em "
                f"{elapsed_ms:.0f}ms. Enviando resultado..."
            )

            send_pickle(
                conn,
                {
                    "status": "ok",
                    "result": result,
                    "elapsed_ms": elapsed_ms,
                    "server_port": port,
                },
            )

        except Exception as exc:
            print(f"[SERVIDOR porta={port}] Erro: {exc}")
            try:
                send_pickle(conn, {"status": "error", "message": str(exc), "server_port": port})
            except Exception:
                pass


def run_server(port: int) -> None:
    """Inicia o servidor e aceita conexoes indefinidamente."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, port))
        server_socket.listen()

        print(f"[SERVIDOR porta={port}] Aguardando conexoes em {HOST}:{port}")

        while True:
            try:
                conn, addr = server_socket.accept()
                thread = threading.Thread(target=handle_client, args=(conn, addr, port), daemon=True)
                thread.start()
            except KeyboardInterrupt:
                print(f"\n[SERVIDOR porta={port}] Encerrando servidor...")
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Servidor worker do PROJETOCP.")
    parser.add_argument("port", type=int, help="Porta do servidor. Exemplo: python Server.py 5000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_server(args.port)


if __name__ == "__main__":
    main()
