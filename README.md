# PROJETOCP - Multiplicacao de Matrizes Distribuida

Projeto da disciplina de Computacao Paralela e Concorrente. A aplicacao simula
uma arquitetura cliente-servidor para multiplicacao distribuida de matrizes com
Python, sockets e NumPy.

## Como funciona

1. O cliente gera duas matrizes aleatorias `A` e `B`.
2. A matriz `A` e dividida verticalmente em partes, uma para cada servidor.
3. A matriz `B` e enviada completa para todos os servidores.
4. Cada servidor calcula `A_sub x B` e devolve um resultado parcial.
5. O cliente concatena os resultados parciais para montar a matriz final `C`.
6. O cliente tambem executa `A x B` de forma serial/local para comparar tempos.
7. Os resultados sao salvos em `benchmark_results.csv` e as matrizes da ultima
   execucao sao salvas em `last_run_matrices.json`.

## Estrutura

```text
PROJETOCP/
+-- Run.py
+-- Client.py
+-- Server.py
+-- benchmark_results.csv
+-- last_run_matrices.json
+-- matrix_analysis2.ipynb
+-- requirements.txt
+-- README.md
```

## Instalacao

Crie e ative um ambiente virtual, se desejar:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Execucao recomendada

Use o orquestrador:

```bash
python Run.py
```

O `Run.py` faz tudo em uma unica execucao:

- pergunta quantos servidores voce quer usar;
- pergunta a porta inicial;
- inicia automaticamente os servidores;
- pergunta se a matriz e quadrada ou nao;
- executa o cliente;
- salva os resultados;
- encerra os servidores no final.

Exemplo para matriz quadrada:

```text
Quantidade de servidores [4]:
Porta inicial [5000]:
A matriz sera quadrada? (s/n): s
Informe o tamanho n da matriz n x n: 100
Deseja adicionar outro teste? (s/n): n
```

Isso gera:

```text
A(100x100) x B(100x100) = C(100x100)
```

Exemplo para matriz nao quadrada:

```text
Quantidade de servidores [4]:
Porta inicial [5000]:
A matriz sera quadrada? (s/n): n
Informe a quantidade de linhas da Matriz A: 100
Informe a quantidade de colunas da Matriz A: 50
A Matriz B precisa ter 50 linhas para ser compativel com A.
Informe a quantidade de linhas da Matriz B: 50
Informe a quantidade de colunas da Matriz B: 120
Deseja adicionar outro teste? (s/n): n
```

Isso gera:

```text
A(100x50) x B(50x120) = C(100x120)
```

## Execucao manual opcional

Se quiser iniciar os 4 servidores padrao em um unico terminal, execute:

```bash
python Server.py
```

Isso abre automaticamente as portas `5000`, `5001`, `5002` e `5003`.

Depois execute o cliente em outro terminal:

```bash
python Client.py --servers 4 --start-port 5000
```

Tambem e possivel passar as configuracoes sem input:

```bash
python Client.py --servers 4 --start-port 5000 --configs "20,10,30;50,25,60;1000,2000,1000"
```

Para mudar a quantidade de servidores ou a porta inicial:

```bash
python Server.py --servers 6 --start-port 6000
python Client.py --servers 6 --start-port 6000
```

Se quiser iniciar apenas uma porta especifica, ainda funciona:

```bash
python Server.py 5000
```

## CSV de benchmark

O arquivo `benchmark_results.csv` usa as colunas:

```text
matrix_size,serial_time_ms,parallel_time_ms,speedup,num_servers,timestamp
```

- `matrix_size`: configuracao testada. Exemplo: `100x100x100` ou `100x50x120`.
- `serial_time_ms`: tempo da multiplicacao local com NumPy.
- `parallel_time_ms`: tempo total da versao distribuida, incluindo comunicacao.
- `speedup`: `serial_time_ms / parallel_time_ms`.
- `num_servers`: quantidade de servidores usados.
- `timestamp`: data e hora da execucao.

## Analise dos resultados

Abra o notebook:

```bash
jupyter notebook matrix_analysis2.ipynb
```

O notebook carrega `benchmark_results.csv`, gera graficos comparativos, mostra o
speedup e le `last_run_matrices.json` para exibir as matrizes da ultima execucao.

## Observacoes tecnicas

- Comunicacao: modulo `socket`.
- Serializacao: `pickle` com cabecalho binario de tamanho fixo.
- Distribuicao: a matriz `A` e dividida por linhas; a matriz `B` e enviada
  completa para cada servidor.
- Calculo no servidor: `np.dot(A_sub, B)`.
- Validacao: o cliente compara o resultado distribuido com o resultado serial.
- Logs dos servidores iniciados por `Run.py`: pasta `server_logs/`.
