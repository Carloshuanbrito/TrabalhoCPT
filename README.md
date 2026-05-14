# PROJETOCP - Multiplicação de Matrizes Distribuída

Projeto da disciplina de Computação Paralela e Concorrente. A aplicação simula
uma arquitetura cliente-servidor para multiplicação distribuída de matrizes com
Python, sockets e paralelismo interno nos servidores.

## Como funciona

1. O cliente gera duas matrizes aleatórias `A` e `B` com NumPy.
2. A matriz `A` é dividida verticalmente em partes, uma para cada servidor.
3. Cada servidor recebe uma submatriz de `A` e a matriz `B` completa via socket.
4. O servidor calcula `A_sub x B` usando threads internas.
5. O cliente recebe todos os resultados parciais e concatena verticalmente para
   montar a matriz final `C`.
6. O cliente também executa `A x B` de forma serial/local e registra os tempos em
   `benchmark_results.csv`.

## Estrutura

```text
PROJETOCP/
├── Client.py
├── Server.py
├── benchmark_results.csv
├── matrix_analysis.ipynb
├── requirements.txt
└── README.md
```

## Instalação

Crie e ative um ambiente virtual, se desejar:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Executando os servidores

Abra um terminal para cada servidor. Exemplo com 4 servidores locais:

```bash
python Server.py 5000
```

Em outro terminal:

```bash
python Server.py 5001
```

Em outro terminal:

```bash
python Server.py 5002
```

Em outro terminal:

```bash
python Server.py 5003
```

O servidor permanece aguardando conexões até ser encerrado com `Ctrl+C`.

## Executando o cliente

Com os servidores ativos, execute:

```bash
python Client.py --servers 4 --start-port 5000
```

O comando acima usa os 4 servidores nas portas `5000`, `5001`, `5002` e `5003`.
O cliente testa automaticamente os tamanhos `20`, `50`, `100` e `200`.

Se você abrir apenas 2 servidores, execute:

```bash
python Client.py --servers 2 --start-port 5000
```

Principais opções do cliente:

- `--servers`: quantidade de servidores em portas sequenciais.
- `--start-port`: primeira porta usada quando `--servers` é informado.
- `--host`: host onde os servidores estão rodando.

## Resultado esperado

Exemplo de saída:

```text
[CLIENTE] Servidores configurados:
  Servidor 1: localhost:5000
  Servidor 2: localhost:5001
  Servidor 3: localhost:5002
  Servidor 4: localhost:5003

============================================================
 TESTE: Matriz 20x20
============================================================

[CLIENTE] Matriz A gerada (20x20):
...
[CLIENTE] Dividindo A em 4 submatrizes (linhas por servidor: 5, 5, 5, 5)...
[CLIENTE] Tempo Serial:      1 ms
[CLIENTE] Tempo Distribuido: 8 ms
[CLIENTE] Speedup:           0.12x
```

## CSV de benchmark

O arquivo `benchmark_results.csv` usa as colunas:

```text
matrix_size,serial_time_ms,parallel_time_ms,speedup,num_servers,timestamp
```

- `matrix_size`: tamanho `n` da matriz `n x n`.
- `serial_time_ms`: tempo da multiplicação local `A @ B`.
- `parallel_time_ms`: tempo total da versão distribuída, incluindo comunicação.
- `speedup`: `serial_time_ms / parallel_time_ms`.
- `num_servers`: quantidade de servidores usados.
- `timestamp`: data e hora da execução.

## Análise dos resultados

Abra o notebook:

```bash
jupyter notebook matrix_analysis.ipynb
```

O notebook carrega `benchmark_results.csv`, consolida médias quando há múltiplas
execuções, gera gráficos comparativos e mostra o speedup para cada tamanho.

Em matrizes pequenas, a comunicação por socket e a serialização com pickle podem
pesar mais que o cálculo. Em matrizes maiores, a distribuição tende a ser mais
vantajosa porque o custo da multiplicação passa a dominar o tempo total.

## Observações técnicas

- Comunicação: módulo `socket`.
- Serialização: `pickle` com cabeçalho binário de tamanho fixo.
- Paralelismo no cliente: `ThreadPoolExecutor` para falar com servidores em paralelo.
- Paralelismo no servidor: `ThreadPoolExecutor` para multiplicar blocos de linhas.
- Validação: o cliente compara o resultado distribuído com o resultado serial.
