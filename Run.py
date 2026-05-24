"""
Orquestrador do PROJETOCP.

Execute apenas:
    python Run.py

Este arquivo inicia os servidores automaticamente, chama o Client.py no modo
interativo e encerra os servidores ao final.
"""

from __future__ import annotations

import random
import subprocess
import sys
import time
from pathlib import Path

import Client


DEFAULT_NUM_SERVERS = 4
DEFAULT_START_PORT = 5000
DEFAULT_HOST = "127.0.0.1"
SERVER_START_WAIT_SECONDS = 2

# Limites para geracao aleatoria de dimensoes
RANDOM_DIM_MIN = 50
RANDOM_DIM_MAX = 2000
RANDOM_NUM_CONFIGS_DEFAULT = 4


def read_positive_int(prompt: str, default: int) -> int:
    """Le um inteiro positivo, aceitando Enter para usar o valor padrao."""
    while True:
        try:
            raw_value = input(f"{prompt} [{default}]: ").strip()
        except EOFError:
            return default

        if not raw_value:
            return default

        try:
            value = int(raw_value)
        except ValueError:
            print("Digite um numero inteiro valido.")
            continue

        if value <= 0:
            print("Digite um numero maior que zero.")
            continue

        return value


def read_matrix_configs_random() -> list[tuple[int, int, int]]:
    """Gera configuracoes de matrizes aleatorias."""
    num_configs = read_positive_int(
        "Quantas configuracoes aleatorias deseja gerar",
        RANDOM_NUM_CONFIGS_DEFAULT,
    )

    configs: list[tuple[int, int, int]] = []
    print()
    for i in range(num_configs):
        rows_a = random.randint(RANDOM_DIM_MIN, RANDOM_DIM_MAX)
        cols_a = random.randint(RANDOM_DIM_MIN, RANDOM_DIM_MAX)
        cols_b = random.randint(RANDOM_DIM_MIN, RANDOM_DIM_MAX)
        configs.append((rows_a, cols_a, cols_b))
        print(f"  Config {i + 1}: A={rows_a}x{cols_a}  B={cols_a}x{cols_b}  -> C={rows_a}x{cols_b}")

    return configs


def read_matrix_configs_manual() -> list[tuple[int, int, int]]:
    """Solicita ao usuario uma sequencia manual de configuracoes de matrizes."""
    print()
    print("Digite cada configuracao no formato:  rows_a cols_a cols_b")
    print("Exemplo: 100 200 300  (A=100x200, B=200x300, C=100x300)")
    print("Digite 'fim' ou deixe em branco para encerrar.\n")

    configs: list[tuple[int, int, int]] = []
    index = 1

    while True:
        try:
            raw = input(f"  Config {index}: ").strip()
        except EOFError:
            break

        if not raw or raw.lower() == "fim":
            break

        parts = raw.split()
        if len(parts) != 3:
            print("  Formato invalido. Use: rows_a cols_a cols_b")
            continue

        try:
            rows_a, cols_a, cols_b = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            print("  Valores invalidos. Digite tres numeros inteiros.")
            continue

        if rows_a <= 0 or cols_a <= 0 or cols_b <= 0:
            print("  Todas as dimensoes devem ser maiores que zero.")
            continue

        configs.append((rows_a, cols_a, cols_b))
        index += 1

    return configs


def choose_matrix_configs() -> list[tuple[int, int, int]]:
    """Pergunta ao usuario se quer matrizes aleatorias ou manuais."""
    print()
    print("Como deseja definir as configuracoes de matrizes?")
    print("  [1] Gerar automaticamente (aleatorio)")
    print("  [2] Inserir manualmente")
    print()

    while True:
        try:
            choice = input("Escolha [1/2]: ").strip()
        except EOFError:
            choice = "1"

        if choice == "1":
            print("\n[RUN] Gerando configuracoes aleatorias...")
            configs = read_matrix_configs_random()
            break
        elif choice == "2":
            print("\n[RUN] Modo de entrada manual:")
            configs = read_matrix_configs_manual()
            break
        else:
            print("Digite 1 ou 2.")

    if not configs:
        print("\n[RUN] Nenhuma configuracao informada. Encerrando.")
        sys.exit(0)

    return configs


def start_servers(num_servers: int, start_port: int) -> tuple[list[subprocess.Popen], list]:
    """Inicia servidores nas portas sequenciais e retorna processos/logs."""
    processes: list[subprocess.Popen] = []
    log_files = []
    logs_dir = Path("server_logs")
    logs_dir.mkdir(exist_ok=True)

    print("\n[RUN] Iniciando servidores...")

    for index in range(num_servers):
        port = start_port + index
        log_path = logs_dir / f"server_{port}.log"
        log_file = log_path.open("w", encoding="utf-8")
        log_files.append(log_file)

        process = subprocess.Popen(
            [sys.executable, "Server.py", str(port)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        processes.append(process)
        print(f"  Servidor {index + 1}: {DEFAULT_HOST}:{port} | log: {log_path}")

    time.sleep(SERVER_START_WAIT_SECONDS)
    return processes, log_files


def stop_servers(processes: list[subprocess.Popen], log_files: list) -> None:
    """Encerra todos os servidores iniciados por este orquestrador."""
    print("\n[RUN] Encerrando servidores...")

    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    for log_file in log_files:
        log_file.close()

    print("[RUN] Servidores encerrados.")


def matrix_configs_to_argument(configs: list[tuple[int, int, int]]) -> str:
    """Converte configs para o formato aceito pelo Client.py --configs."""
    return ";".join(
        f"{rows_a},{cols_a},{cols_b}"
        for rows_a, cols_a, cols_b in configs
    )


def run_client(
    num_servers: int,
    start_port: int,
    matrix_configs: list[tuple[int, int, int]],
) -> int:
    """Executa Client.py com as configuracoes ja coletadas pelo Run.py."""
    print("\n[RUN] Iniciando cliente.\n")

    command = [
        sys.executable,
        "Client.py",
        "--servers",
        str(num_servers),
        "--start-port",
        str(start_port),
        "--host",
        DEFAULT_HOST,
        "--configs",
        matrix_configs_to_argument(matrix_configs),
    ]

    return subprocess.call(command)


def main() -> None:
    print("=" * 60)
    print(" PROJETOCP - Execucao automatica")
    print("=" * 60)
    print(f"\n[RUN] Usando {DEFAULT_NUM_SERVERS} servidores na porta inicial {DEFAULT_START_PORT}.")

    matrix_configs = choose_matrix_configs()

    processes: list[subprocess.Popen] = []
    log_files: list = []

    try:
        processes, log_files = start_servers(DEFAULT_NUM_SERVERS, DEFAULT_START_PORT)
        return_code = run_client(DEFAULT_NUM_SERVERS, DEFAULT_START_PORT, matrix_configs)

        if return_code == 0:
            print("\n[RUN] Execucao concluida com sucesso.")
        else:
            print(f"\n[RUN] Cliente terminou com codigo {return_code}.")
            print("[RUN] Verifique os logs em server_logs/ para detalhes.")

    finally:
        stop_servers(processes, log_files)


if __name__ == "__main__":
    main()
