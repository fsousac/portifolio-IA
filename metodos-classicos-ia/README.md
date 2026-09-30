# Módulo 9.2 - Métodos clássicos de IA

Seis projetos em Python que implementam, à mão, algoritmos clássicos de IA (sem bibliotecas de IA, CSP, GA ou SAT). O texto completo, com resultados e discussão, está em [`relatorio.md`](relatorio.md).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `relatorio.md` | Relatório: introdução, 6 capítulos (um por projeto) e conclusão geral |
| `01_busca_informada.py` | A*, UCS, Greedy e Weighted A* no alinhamento de DNA (verdade: Needleman-Wunsch) |
| `02_busca_nao_informada.py` | BFS, DFS, DLS, IDDFS, UCS e bidirecional no Rush Hour (10 instâncias geradas por busca reversa) |
| `03_busca_complexa.py` | Hill climbing, recozimento simulado (3 resfriamentos) e beam no layout de teclado |
| `04_algoritmo_genetico.py` | GA que aproxima uma imagem sintética com triângulos translúcidos (aceita imagem própria via argumento) |
| `05_csp.py` | CSP genérico: backtracking, MRV, grau, LCV, FC, AC-3, MAC e Min-Conflicts na cruzadinha |
| `06_base_conhecimento.py` | Campo Minado por inferência lógica: KB em CNF, DPLL próprio, regras de subconjunto e chute por probabilidade |
| `corpus_pt.txt` | Corpus em português (Machado de Assis, domínio público), lido pelo projeto 3 |
| `palavras_pt.txt` | Dicionário de palavras sem acento, lido pelo projeto 5 |
| `figs/` | Figuras (`.png`, e `04_evolucao.gif` extra) e a saída textual completa de cada script (`NN_saida.txt`) |
| `requirements.txt` | Dependências: numpy, matplotlib, pillow |

## Instalação

```
pip install -r requirements.txt
```

Testado com Python 3.14 no Windows. Nenhum acesso à internet é necessário para rodar.

## Como rodar

Cada script é independente, deve ser rodado a partir de qualquer pasta (os caminhos são relativos ao próprio arquivo), usa sementes fixas, imprime as tabelas no console e reescreve `figs/NN_*.png` e `figs/NN_saida.txt`. Contadores (nós, avaliações, backtracks) são reproduzíveis; os tempos variam com a carga da máquina.

```
python 01_busca_informada.py
python 02_busca_nao_informada.py
python 03_busca_complexa.py
python 04_algoritmo_genetico.py            # opcional: python 04_algoritmo_genetico.py minha_imagem.png
python 05_csp.py
python 06_base_conhecimento.py
```

| Script | Tempo aproximado (máquina de desenvolvimento) |
|---|---|
| `01_busca_informada.py` | cerca de 5 s |
| `02_busca_nao_informada.py` | 1 a 4 min |
| `03_busca_complexa.py` | 2 a 5 min |
| `04_algoritmo_genetico.py` | 3 a 6 min |
| `05_csp.py` | 4 min |
| `06_base_conhecimento.py` | 2 a 2,5 min |

Todos terminam com código 0 e passam os `assert` de autoverificação embutidos.
