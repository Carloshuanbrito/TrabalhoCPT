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

Abra um terminal para cada servidor. Exemplo com 2 servidores locais:

```bash
python Server.py 5000
```

Em outro terminal:

```bash
python Server.py 5001
```

Também é possível configurar host, porta e número de threads internas:

```bash
python Server.py --host 127.0.0.1 --port 5000 --workers 4
```

O servidor permanece aguardando conexões até ser encerrado com `Ctrl+C`.

## Executando o cliente

Com os servidores ativos, execute:

```bash
python Client.py
```

Por padrão, o cliente usa 2 servidores nas portas `5000` e `5001` e testa os
tamanhos `20`, `100`, `500` e `1000`.

Para executar apenas um tamanho:

```bash
python Client.py --size 100 --servers 2 --start-port 5000
```

Para informar servidores manualmente:

```bash
python Client.py --size 100 --server-list 127.0.0.1:5000,127.0.0.1:5001
```

Principais opções do cliente:

- `--size`: executa um único tamanho `n` para matrizes `n x n`.
- `--sizes`: lista de tamanhos separados por vírgula, por exemplo `20,100,500,1000`.
- `--servers`: quantidade de servidores em portas sequenciais.
- `--start-port`: primeira porta usada quando `--servers` é informado.
- `--server-list`: lista explícita no formato `host:porta,host:porta`.
- `--output`: arquivo CSV de saída.
- `--print-matrices`: imprime as matrizes quando `n <= 10`.

## Resultado esperado

Exemplo de saída:

```text
[CLIENT] Servidores configurados: 127.0.0.1:5000, 127.0.0.1:5001
[CLIENT] Gerando matrizes A (100x100) e B (100x100)...
[CLIENT] Enviando submatriz 1/2 (50, 100) para Servidor 1 (127.0.0.1:5000)...
[CLIENT] Enviando submatriz 2/2 (50, 100) para Servidor 2 (127.0.0.1:5001)...
[CLIENT] Resultado parcial 1 recebido.
[CLIENT] Resultado parcial 2 recebido.
[CLIENT] Resultado recebido de todos os servidores.
[CLIENT] Matriz resultante C montada com sucesso!
[CLIENT] Tempo Serial: 180.00ms | Tempo Distribuido: 95.00ms | Speedup: 1.89x
[CLIENT] Resultados salvos em benchmark_results.csv
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
