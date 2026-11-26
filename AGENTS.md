# AGENTS.md

Guia para agentes (e pessoas) que vão construir os próximos módulos do portfólio de **FGA0221 - Inteligência Artificial** (UnB, prof. Fabiano Araujo Soares, 2º/2026) replicando o que foi feito em [`metodos-classicos-ia/`](metodos-classicos-ia/), que é o módulo de referência. Em caso de dúvida, abra o equivalente lá e copie o padrão.

## 1. Objetivo e regras de avaliação

Cada módulo corresponde a uma parte do portfólio (nota 0-10; a nota final é a média das quatro). Regras do plano de ensino que **mandam** sobre qualquer preferência de estilo:

- Tudo em **Python**. Cada projeto é entregue como **`.py` comentado** **e** descrito no **relatório**. Algoritmo sem relatório é desconsiderado; projeto não implementado custa **50% da nota da parte**. Entregue **todos** os projetos.
- O relatório tem **no máximo 4 páginas por projeto** e, para cada projeto, deve conter: (1) explicação do problema, (2) descrição detalhada dos algoritmos desenvolvidos, (3) resultados com **imagens e valores**, (4) considerações finais com **potenciais e limitações**.
- Critérios de correção: originalidade (cópia = zero; evitar o problema de livro-texto), nível de dificuldade, clareza (imagens em boa resolução, código explicado), completude, contribuições (algoritmos/conceitos que não foram vistos em aula valem mais).

### Módulos pendentes

| Parte | Pasta sugerida | Entrega | Projetos obrigatórios |
|---|---|---|---|
| 9.1 Introdução à IA | (texto, sem código) | 03/09/2026 | Resenha crítica ampliada (máx. 12 páginas, ABNT). Fora do escopo deste guia |
| 9.2 Métodos clássicos | `metodos-classicos-ia/` | 09/10/2026 | **Feito** (6 projetos). Use como modelo |
| 9.3 Tratando incerteza | `tratando-incerteza-ia/` | 05/11/2026 | 1. Redes Bayesianas · 2. Modelos de Markov ocultos (HMM) · 3. Filtro de Kalman |
| 9.4 Aprendizado de máquina | `aprendizado-maquina-ia/` | 08/12/2026 | 1. Supervisionado **não neural** · 2. Não supervisionado · 3. Aprendizado por reforço · 4. Deep Learning (CNN, RNN, LSTM, Transformer, LeNet etc.; dados públicos como MNIST ou Cats and Dogs/Kaggle) |

Entrega pelo Microsoft Teams (pasta "Avaliações" > subpasta da atividade > pasta com a matrícula). Isso é feito pelo aluno, não pelo agente.

## 2. Estrutura de um módulo

Idêntica à do módulo de referência (ver também a seção "Convenções dos módulos" do [README.md](README.md)):

```
<modulo>/
├─ README.md          # arquivos, instalação, comando e tempo de cada projeto
├─ requirements.txt   # dependências só deste módulo
├─ relatorio.md       # relatório do módulo (todos os projetos)
├─ NN_<projeto>.py    # um script independente por projeto (NN = 01, 02, ...)
├─ <dados auxiliares> # corpora, datasets pequenos etc., ao lado dos scripts
└─ figs/              # NN_*.png e NN_saida.txt gerados pelos scripts
```

Nomes de arquivo e pasta em `snake_case`/`kebab-case` sem acento, como no módulo 9.2 (`01_busca_informada.py`, `metodos-classicos-ia/`). Ao terminar um módulo, **acrescente uma linha na tabela "Módulos" do [README.md](README.md) raiz**.

## 3. Como escrever cada script `NN_*.py`

Padrão observado em `01`-`06`; siga todos os itens.

1. **Docstring inicial** longa e honesta, em seções em maiúsculas: `PROBLEMA`, modelagem/algoritmos (o que é estado, variável, observação etc.), `AUTOVERIFICACAO` e a linha `Execucao: python NN_x.py  (gera figs/NN_*.png e figs/NN_saida.txt)`. Declare simplificações e limites conhecidos aqui.
2. **Sem acentos** em comentários, strings e saídas do código (compatibilidade de console/Windows). Acentos ficam só nos `.md`.
3. **Implemente o algoritmo central à mão.** No módulo 9.2 não se usou biblioteca de IA/CSP/GA/SAT; só `numpy`, `matplotlib`, `pillow` e a biblioteca padrão. Para os módulos novos a mesma regra vale para o **algoritmo que dá nome ao projeto**: rede Bayesiana (inferência por enumeração/eliminação de variáveis), HMM (forward, Viterbi, Baum-Welch), Kalman (predict/update em numpy), regressão/árvore/SVM/k-means/Q-learning etc. escritos por você. Bibliotecas servem para dados, métricas e figuras e como **referência independente** nos `assert` (ex.: comparar com `scikit-learn`/`filterpy`, se estiver instalada de forma opcional). Exceção natural: **Deep Learning** (9.4 item 4) pode usar PyTorch ou TensorFlow/Keras; documente isso no `requirements.txt` e no relatório e implemente à mão pelo menos o laço de treino/avaliação ou um componente (ex.: convolução, atenção) para mostrar compreensão.
4. **Reprodutível:** sementes fixas (`random.Random(n)`, `np.random.default_rng(n)`, `torch.manual_seed`). Contadores e métricas devem repetir entre execuções; só tempos de relógio variam (e o relatório deve privilegiar contagens/métricas).
5. **Caminhos relativos ao arquivo:** `PASTA = Path(__file__).resolve().parent; FIGS = PASTA / "figs"`. O script roda de qualquer diretório. `matplotlib.use("Agg")` (sem janela). `FIGS.mkdir(exist_ok=True)`.
6. **Saída:** uma função de impressão que imprime e acumula as linhas (padrão `_saida`/`_LOG`); no fim grava `figs/NN_saida.txt`. Imprima as tabelas que o relatório vai citar. Sem acesso à internet em tempo de execução, **exceto** download de dataset público documentado (nesse caso faça cache em `dados/` e fallback claro; prefira datasets pequenos).
7. **Autoverificação com `assert`** dentro do `main()`: compare o seu algoritmo com uma verdade independente (solução exata, força bruta em instância pequena, fórmula fechada, implementação de biblioteca, propriedade matemática como "probabilidades somam 1", "covariância permanece simétrica positiva semidefinida"). Ao menos um assert estatístico robusto sobre o resultado principal. O script deve terminar com código 0.
8. **Experimentos com comparação justa:** mesmas instâncias/sementes para todos os algoritmos (comparação pareada), várias sementes quando o método é estocástico, intervalos de confiança ou desvio quando fizer sentido, e **ablação** (remover um componente e medir). Varie o tamanho/dificuldade para mostrar como o custo ou o erro cresce.
9. **Figuras:** `.png` em `figs/NN_<nome>.png`, resolução boa (dpi >= 150), eixos, legenda e título legíveis. Pelo menos 2-3 figuras por projeto: uma de **ilustração do problema/solução** e uma ou duas de **resultado quantitativo**. GIFs são opcionais.
10. **Tempo:** idealmente cada script termina em poucos minutos (referência: 5 s a 6 min). Registre o tempo aproximado no README do módulo.
11. **Originalidade e dificuldade pesam:** não repita o exemplo de livro (nada de "rede do alarme/terremoto" como único exemplo, "MNIST com 2 camadas" sem análise, "rastrear uma bolinha 1D" só). Escolha um problema com (i) verdade verificável, (ii) tamanho ajustável, (iii) resultado visual ou interpretável, e inclua pelo menos um conceito além do visto em aula.

### Ideias de problemas (apenas pontos de partida)

- **Bayes:** diagnóstico com nós ruidosos (noisy-OR), comparação enumeração x eliminação de variáveis x amostragem (rejeição, ponderação por verossimilhança, Gibbs), aprendizado de parâmetros por contagem, verificação por força bruta.
- **HMM:** decodificação de texto/ortografia (estados ocultos = letras/classes gramaticais), reconhecimento de atividade a partir de sensores sintéticos, forward-backward, Viterbi e Baum-Welch com verificação por log-verossimilhança monótona.
- **Kalman:** rastreamento 2D com observações ruidosas e perdidas, comparação KF x média móvel x filtro de partículas, consistência por NEES/NIS, sensibilidade a Q e R.
- **ML supervisionado (não neural):** árvore de decisão/random forest/SVM/k-NN/regressão logística escritos à mão em dados públicos tabulares, validação cruzada e matriz de confusão.
- **Não supervisionado:** k-means (k-means++), GMM por EM, PCA por SVD, DBSCAN; métricas de silhueta/ARI.
- **RL:** Q-learning/SARSA/Dyna ou política por gradiente em um ambiente próprio (gridworld não trivial, corrida, estoque), curvas de aprendizado por várias sementes.
- **Deep Learning:** CNN/LSTM/Transformer pequeno em dado público, com curvas de treino/validação, matriz de confusão, exemplos de erros e ablação (regularização, profundidade).

## 4. Como escrever `relatorio.md`

Siga o `relatorio.md` do módulo 9.2 (estrutura e tom). Um único arquivo por módulo, em português com acentos:

1. **Título** do módulo e **Introdução**: objetivo, tabela dos projetos (`#`, família, problema, link do script), "Como reproduzir" (Python, `pip install -r requirements.txt`, scripts independentes, sementes fixas, saídas em `figs/`, tempos variam), "Critério de escolha dos problemas" e **Índice**.
2. **Um capítulo por projeto** (`## Projeto N - <Família>: <problema>`), com a linha de código/execução logo após o título e **exatamente quatro seções**:
   - `### 1. Problema` - o que é, como foi modelado (estado/variáveis/observações), métricas e instâncias.
   - `### 2. Algoritmos` - descrição detalhada, pseudocódigo curto em bloco de código, parâmetros e otimizações, e o que é "verdade de referência".
   - `### 3. Resultados` - **tabelas com valores reais** copiadas de `figs/NN_saida.txt` e as figuras com `![legenda](figs/NN_x.png)`, mais a leitura de cada resultado (não só números).
   - `### 4. Considerações finais` - potenciais, **limitações honestas** (instâncias pequenas, poucas sementes, simplificações), o que não deu certo e o que mudaria.
3. **Conclusão geral** do módulo: tabela comparativa (abordagem, garantia, o que se aprendeu, quando usar), padrões que se repetem e limitações do estudo.
4. **Limite de 4 páginas por projeto** (aprox. 1.800-2.200 palavras de texto + figuras/tabelas por capítulo; o capítulo do projeto 6 de referência cabe nisso). Se estourar, corte prosa, não resultados.
5. **Links** para os scripts no formato absoluto do GitHub, como no módulo de referência: `https://github.com/fsousac/portifolio-IA/blob/main/<modulo>/NN_<projeto>.py`.
6. **Números do relatório vêm das execuções reais.** Rode o script, leia `figs/NN_saida.txt` e cite o que está lá. Nunca invente, arredonde para favorecer ou copie valores de execuções antigas. Se rodar de novo e os contadores mudarem, atualize o texto.
7. Se for preciso entregar em PDF (o Teams pede um arquivo), gere a partir do `.md`, conferindo a contagem de páginas por projeto. A conversão é uma etapa final, fora do código dos projetos.

## 5. README do módulo

Copie o formato de [`metodos-classicos-ia/README.md`](metodos-classicos-ia/README.md): título "Módulo 9.x - ...", tabela de arquivos, instalação (`pip install -r requirements.txt`), "Como rodar" (comandos), tabela de tempo aproximado por script e a nota "Todos terminam com código 0 e passam os `assert` de autoverificação". Mantenha `requirements.txt` mínimo (pacotes sem versão fixa como no módulo de referência, a menos que algo quebre).

## 6. Fluxo de trabalho para cada módulo

1. Leia o plano de ensino (seção 9.x) e o módulo de referência. Escolha os problemas (um por projeto obrigatório) e liste **verdade de referência**, **métricas**, **ablação** e **figuras** de cada um antes de codar.
2. Crie a pasta do módulo, `requirements.txt` e instale em um `.venv` do módulo (já no `.gitignore`).
3. Para cada projeto: escreva `NN_*.py` -> **rode** -> confira asserts, saída e figuras (abra as imagens) -> ajuste. Só então escreva o capítulo do relatório com os números gerados. Não declare um algoritmo pronto sem tê-lo executado até o fim.
4. Escreva `README.md` do módulo e `relatorio.md` (introdução, 3 ou 4 capítulos, conclusão). Atualize o README raiz.
5. Verificação final: todos os scripts rodam do zero (apague `figs/` e rode), terminam com código 0, os números do relatório batem com `figs/NN_saida.txt`, todas as figuras citadas existem, links funcionam e nenhum projeto obrigatório falta.
6. **Entregáveis mínimos por projeto: algoritmo (`.py`) + figuras + saída + capítulo do relatório.** Os quatro vêm juntos; nunca entregue o código sem o relatório.

## 7. Commits e escopo

- Um commit por etapa coerente (ex.: "adicionando projeto de filtro de Kalman", "adicionando relatório do módulo de incerteza"), em português, no estilo do histórico atual. Não faça commit nem push sem pedido explícito.
- Não altere o módulo `metodos-classicos-ia/` ao construir os próximos, a menos que seja um erro real.
- Não versione `.venv/`. Versione `figs/` e as saídas, pois o relatório depende delas. Datasets grandes não entram no repositório: baixe em tempo de execução para um cache ignorado pelo git e documente.
