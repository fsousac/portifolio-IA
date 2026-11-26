# Módulo 9.4 - Aprendizado de máquina

Quatro projetos em Python, um para cada família de aprendizado pedida no plano de ensino: supervisionado não neural, não supervisionado, por reforço e deep learning. Os algoritmos que dão nome a cada projeto foram escritos à mão (árvores, Random Forest, Rotation Forest, boosting com CV aninhada, stacking, calibração de Platt, k-means, EM, DBSCAN, PCA, agrupamento espectral, propagação de rótulos, Q-learning, SARSA, Expected SARSA, Dyna-Q, R-MAX, iteração de valor; convolução por im2col, entropia cruzada, SGD de Nesterov e laço de treino). O scikit-learn entra só para carregar dados, fazer as partições e servir de **referência independente** nos `assert`. No projeto 4 o PyTorch fornece tensores, autograd e camadas prontas. O texto completo, com resultados e discussão, está em [`relatorio.md`](relatorio.md).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `relatorio.md` | Relatório: introdução, 4 capítulos (um por projeto) e conclusão geral |
| `relatorio.pdf` | Mesmo relatório em PDF (A4, cada projeto em até 4 páginas), para a entrega no Teams |
| `01_supervisionado_arvores.py` | Diagnóstico de tumor de mama (Breast Cancer Wisconsin) e dados sintéticos com ruído de rótulo: CART, Bagging, Random Forest com OOB e calibração de Platt, Rotation Forest, Gradient Boosting (Newton, estocástico com CV aninhada), stacking, regressão logística, limiar de Bayes por custo e importância por permutação (verdade: scikit-learn) |
| `02_nao_supervisionado_digitos.py` | Descobrir dígitos manuscritos (UCI Digits) sem rótulos: PCA por SVD, k-means/k-means++, GMM por EM com BIC, DBSCAN, agrupamento espectral em grafo kNN e propagação de rótulos semissupervisionada (verdade: scikit-learn e propriedades do SVD/EM/Laplaciano) |
| `03_reforco_estoque.py` | Padaria que aprende a repor estoque com demanda semanal: iteração de valor/política, Q-learning, SARSA, Expected SARSA, Dyna-Q, R-MAX e política (s,S) (verdade: MDP resolvido exatamente) |
| `04_deep_learning_cnn.py` | Fashion-MNIST: softmax, MLP, LeNet-5 com convolução própria, CNN-16 e CNN-32 com BatchNorm/dropout/aumento de dados/suavização de rótulos/TTA, mais ablação com 6 variantes × 2 sementes (verdade: `F.conv2d`, `torch.optim.SGD`, diferenças finitas) |
| `figs/` | Figuras (`.png`) e a saída textual completa de cada script (`NN_saida.txt`) |
| `dados/` | Cache do Fashion-MNIST e dos pesos treinados pelo projeto 4 (`dados/cache04/`), **não versionado** |
| `requirements.txt` | Dependências: numpy, matplotlib, scikit-learn, torch, torchvision |

## Instalação

```
pip install -r requirements.txt
```

Para instalar o PyTorch só para CPU (bem menor), use antes `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`. Testado com Python 3.12 no Linux, sem GPU.

Os projetos 1 a 3 rodam sem internet (os dados 1 e 2 acompanham o scikit-learn; o 3 é um simulador próprio). O projeto 4 baixa o Fashion-MNIST na primeira execução para `dados/` e guarda os modelos treinados em `dados/cache04/`; execuções seguintes reaproveitam os pesos (apague a pasta para treinar do zero). Os projetos 1, 3 e 4 usam 4 processos em paralelo.

## Como rodar

Cada script é independente, roda a partir de qualquer pasta (caminhos relativos ao próprio arquivo), usa sementes fixas, imprime as tabelas no console e reescreve `figs/NN_*.png` e `figs/NN_saida.txt`. As métricas e contagens são reproduzíveis; só os tempos de relógio variam.

```
python 01_supervisionado_arvores.py
python 02_nao_supervisionado_digitos.py
python 03_reforco_estoque.py
python 04_deep_learning_cnn.py
```

| Script | Tempo aproximado (máquina de desenvolvimento, 4 núcleos, sem GPU) |
|---|---|
| `01_supervisionado_arvores.py` | cerca de 11 min (árvores em Python puro; Rotation Forest e stacking dominam) |
| `02_nao_supervisionado_digitos.py` | cerca de 3 min |
| `03_reforco_estoque.py` | cerca de 5 min (10 sementes × 1 milhão de dias por método) |
| `04_deep_learning_cnn.py` | cerca de 1 min com os pesos em `dados/cache04/`; **algumas horas** do zero (treina 5 redes por 20 épocas e 12 variantes da ablação em CPU; cerca de 2,8 h na execução de desenvolvimento) |

Todos terminam com código 0 e passam os `assert` de autoverificação embutidos.
