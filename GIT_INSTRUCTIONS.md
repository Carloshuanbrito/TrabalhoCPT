# Instruções para enviar o projeto ao Git

## 1. Configurar Git (primeira vez)

Se não tiver Git instalado, baixe em: https://git-scm.com/download/win

Depois, configure suas credenciais globais:

```powershell
git config --global user.name "Seu Nome"
git config --global user.email "seu.email@example.com"
```

## 2. Inicializar repositório local

Na pasta do projeto:

```powershell
cd "c:\Users\finan\OneDrive\Desktop\projetoCP"
git init
```

Isso cria uma pasta `.git` (oculta).

## 3. Adicionar arquivos ao repositório

```powershell
git add .
```

Ou adicione arquivos específicos:

```powershell
git add Client.py Server.py matrix_analysis.ipynb README.md requirements.txt .gitignore
```

## 4. Fazer o primeiro commit

```powershell
git commit -m "Inicial: Projeto de multiplicação de matrizes distribuída"
```

Ou com mensagem mais descritiva:

```powershell
git commit -m "feat: Implementação de cliente-servidor para multiplicação de matrizes
- Cliente divide matriz A em blocos
- Servidores realizam multiplicação em paralelo
- Notebook com análise de desempenho
- Suporte a matrizes aleatórias configuráveis"
```

## 5. Criar repositório no GitHub

1. Vá em https://github.com/new
2. Crie um repositório (ex: `projetoCP`)
3. **NÃO** inicialize com README, .gitignore ou LICENSE (já temos)
4. Clique em "Create repository"

## 6. Adicionar repositório remoto

Copie a URL HTTPS do seu repositório e execute:

```powershell
git remote add origin https://github.com/seu-usuario/projetoCP.git
```

Ou com SSH (se tiver configurado):

```powershell
git remote add origin git@github.com:seu-usuario/projetoCP.git
```

## 7. Fazer push para GitHub

```powershell
git branch -M main
git push -u origin main
```

## 8. Verificar status

Para checar se tudo foi enviado:

```powershell
git status
```

Deve aparecer: `nothing to commit, working tree clean`

## Comandos úteis depois

### Fazer novos commits

Depois de fazer alterações:

```powershell
git add .
git commit -m "Descrição da mudança"
git push
```

### Atualizar do repositório remoto

```powershell
git pull origin main
```

### Ver histórico

```powershell
git log --oneline
```

### Ver diferenças

```powershell
git diff
```

## Estrutura do repositório

Após enviar, seu repositório terá:

```
projetoCP/
├── Client.py
├── Server.py
├── matrix_analysis.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── .git/ (pasta oculta)
```

## Pronto!

Agora seu projeto está versionado e salvo no GitHub. Você pode:

- Compartilhar o link com colegas
- Clonar em outro computador: `git clone https://github.com/seu-usuario/projetoCP.git`
- Colaborar com outras pessoas
- Recuperar versões antigas quando necessário
