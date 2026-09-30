# portifolio-IA

Portfólio de projetos de Inteligência Artificial, organizado por módulo. Cada módulo fica em uma pasta própria, com relatório, código e figuras, e tem o seu próprio `README.md` com as instruções de instalação e execução.

## Módulos

| Módulo                  | Pasta                                            | Conteúdo                                                                                                                                                                                         | Como rodar                                         |
| ----------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| Métodos Clássicos de IA | [`metodos-classicos-ia/`](metodos-classicos-ia/) | 6 projetos em Python (busca informada, busca não informada, busca complexa, algoritmo genético, CSP e base de conhecimento) e o relatório em [`relatorio.md`](metodos-classicos-ia/relatorio.md) | [README do módulo](metodos-classicos-ia/README.md) |

## Como rodar um módulo

Os passos são os mesmos para qualquer módulo. Recomenda-se um ambiente virtual (`venv`) por módulo, para isolar as dependências de cada um. A pasta `.venv` já está no `.gitignore`.

**1. Criar e ativar o ambiente virtual**

```
cd <pasta-do-modulo>
python -m venv .venv
```

Ativação, conforme o terminal:

| Terminal                 | Comando                                                                                |
| ------------------------ | -------------------------------------------------------------------------------------- |
| PowerShell (Windows)     | `.venv\Scripts\Activate.ps1`                                                           |
| CMD (Windows)            | `.venv\Scripts\activate.bat`                                                           |
| Git Bash / Linux / macOS | `source .venv/Scripts/activate` (Windows) ou `source .venv/bin/activate` (Linux/macOS) |

Se o PowerShell bloquear a ativação por política de execução, rode uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Com o ambiente ativo, o prompt mostra `(.venv)`.

**2. Instalar as dependências do módulo**

```
pip install -r requirements.txt     # cada módulo declara as próprias dependências
```

**3. Rodar os projetos**

Execute os scripts listados na tabela "Como rodar" do README do módulo (por exemplo, `python 01_busca_informada.py`). Para sair do ambiente virtual, use `deactivate`. Testado com Python 3.14 no Windows.

## Convenções dos módulos

Para manter os módulos consistentes, cada um segue esta estrutura:

```
<modulo>/
├─ README.md          # arquivos, instalação, comando e tempo de cada projeto
├─ requirements.txt   # dependências só deste módulo
├─ relatorio.md       # relatório do módulo
├─ NN_<projeto>.py    # um script independente por projeto, comentado
└─ figs/              # figuras geradas e saídas NN_saida.txt
```

- Cada script é independente, roda de qualquer pasta (caminhos relativos ao próprio arquivo) e usa sementes fixas.
- Os scripts imprimem as tabelas no console e reescrevem as figuras em `figs/`.
- Os resultados citados no relatório vêm dessas execuções; só os tempos variam com a máquina.

## Adicionar um novo módulo

1. Crie a pasta do módulo na raiz, seguindo a estrutura acima.
2. Escreva o `README.md` do módulo com a tabela de projetos (comando, tempo aproximado, saídas).
3. Acrescente uma linha na tabela "Módulos" deste arquivo, apontando para a pasta e para o README do módulo.
