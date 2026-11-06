# Módulo 9.3 - Tratando incerteza

Três projetos em Python que implementam, à mão, algoritmos de raciocínio sob incerteza (só numpy e matplotlib; nenhuma biblioteca de redes bayesianas, HMM ou filtragem). O texto completo, com resultados e discussão, está em [`relatorio.md`](relatorio.md).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `relatorio.md` | Relatório: introdução, 3 capítulos (um por projeto) e conclusão geral |
| `01_redes_bayesianas.py` | Diagnóstico respiratório com 15 nós noisy-OR: enumeração, eliminação de variáveis, amostragem por rejeição/ponderação/Gibbs e aprendizado de parâmetros (verdade: conjunta por força bruta) |
| `02_hmm_digitacao.py` | HMM que corrige erros de digitação num teclado QWERTY: forward-backward, Viterbi, Baum-Welch e HMM de 2ª ordem (verdade: força bruta em HMM pequeno) |
| `03_filtro_kalman.py` | Rastreio de um drone que atravessa um túnel sem sinal: filtro de Kalman, suavizador RTS, filtro de partículas, NEES/NIS (verdade: solução MAP em lote e fórmulas do filtro alfa-beta) |
| `corpus_pt.txt` | Corpus em português (Machado de Assis, domínio público), lido pelo projeto 2 (cópia do arquivo do módulo 9.2) |
| `figs/` | Figuras (`.png`) e a saída textual completa de cada script (`NN_saida.txt`) |
| `requirements.txt` | Dependências: numpy, matplotlib |

## Instalação

```
pip install -r requirements.txt
```

Testado com Python 3.12 no Linux. Nenhum acesso à internet é necessário para rodar.

## Como rodar

Cada script é independente, roda a partir de qualquer pasta (caminhos relativos ao próprio arquivo), usa sementes fixas, imprime as tabelas no console e reescreve `figs/NN_*.png` e `figs/NN_saida.txt`. As métricas e contagens são reproduzíveis; só os tempos de relógio variam.

```
python 01_redes_bayesianas.py
python 02_hmm_digitacao.py
python 03_filtro_kalman.py
```

| Script | Tempo aproximado (máquina de desenvolvimento) |
|---|---|
| `01_redes_bayesianas.py` | cerca de 1 min |
| `02_hmm_digitacao.py` | cerca de 30 s |
| `03_filtro_kalman.py` | cerca de 25 s |

Todos terminam com código 0 e passam os `assert` de autoverificação embutidos.
