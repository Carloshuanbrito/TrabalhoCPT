import argparse
import socket
import pickle
import struct
import numpy as np


def send_pickle(sock, obj):
    data = pickle.dumps(obj)
    header = struct.pack('!I', len(data))
    sock.sendall(header)
    sock.sendall(data)


def recv_all(sock, size):
    buffer = b''
    while len(buffer) < size:
        packet = sock.recv(size - len(buffer))
        if not packet:
            raise EOFError('Socket closed before receiving all data')
        buffer += packet
    return buffer


def recv_pickle(sock):
    header = recv_all(sock, 4)
    size = struct.unpack('!I', header)[0]
    data = recv_all(sock, size)
    return pickle.loads(data)


def generate_random_matrices(n, low=-10, high=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    a = np.random.randint(low, high + 1, size=(n, n))
    b = np.random.randint(low, high + 1, size=(n, n))
    return a, b


def split_matrix(a, num_servers):
    rows_per_server = a.shape[0] // num_servers
    submatrices = []
    for i in range(num_servers):
        start = i * rows_per_server
        end = (i + 1) * rows_per_server if i < num_servers - 1 else a.shape[0]
        submatrices.append(a[start:end])
    return submatrices


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed matrix multiplication client')
    parser.add_argument('--size', type=int, default=4, help='Matrix size n for n x n matrices')
    parser.add_argument('--servers', type=int, default=4, help='Number of servers')
    parser.add_argument('--start-port', type=int, default=5001, help='Starting port for server list')
    parser.add_argument('--low', type=int, default=-10, help='Minimum random value')
    parser.add_argument('--high', type=int, default=10, help='Maximum random value')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible runs')
    return parser.parse_args()


def main():
    args = parse_args()
    num_servers = args.servers
    ports = [args.start_port + i for i in range(num_servers)]

    if num_servers < 1:
        raise ValueError('Number of servers must be at least 1')
    if args.size < num_servers:
        raise ValueError('Matrix size must be at least the number of servers')

    a, b = generate_random_matrices(args.size, low=args.low, high=args.high, seed=args.seed)
    print('Matrix A:')
    print(a)
    print('\nMatrix B:')
    print(b)

    sub_as = split_matrix(a, num_servers)
    results = []

    for i, port in enumerate(ports):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(('localhost', port))
            print(f'Connecting to server on port {port}...')
            send_pickle(sock, {'type': 'task', 'sub_a': sub_as[i], 'b': b})

            ack = recv_pickle(sock)
            print(f'Server {port} ack:', ack)

            result = recv_pickle(sock)
            results.append(result)

    final_result = np.vstack(results)
    print('\nFinal result (A * B):')
    print(final_result)

    expected = np.dot(a, b)
    print('\nExpected result:')
    print(expected)
    print('Results match:', np.allclose(final_result, expected))


if __name__ == '__main__':
    main()
