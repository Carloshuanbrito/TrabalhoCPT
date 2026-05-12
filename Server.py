import socket
import pickle
import struct
import numpy as np
from concurrent.futures import ThreadPoolExecutor


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


def multiply_row(row, b):
    return np.dot(row, b)


def matrix_multiply(sub_a, b):
    with ThreadPoolExecutor(max_workers=min(len(sub_a), 4)) as executor:
        futures = [executor.submit(multiply_row, row, b) for row in sub_a]
        results = [future.result() for future in futures]
    return np.array(results)


def main(port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', port))
    server_socket.listen(1)
    print(f'Server listening on port {port}')

    conn, addr = server_socket.accept()
    print(f'Connected to {addr}')

    task = recv_pickle(conn)
    if task.get('type') != 'task':
        raise ValueError('Unexpected task type from client')

    send_pickle(conn, 'ACK')
    result = matrix_multiply(task['sub_a'], task['b'])
    send_pickle(conn, result)

    conn.close()
    server_socket.close()


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print('Usage: python Server.py <port>')
        sys.exit(1)

    port = int(sys.argv[1])
    main(port)
