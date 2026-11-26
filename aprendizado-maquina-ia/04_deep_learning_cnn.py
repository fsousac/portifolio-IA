"""
Projeto 4 - Deep Learning: redes convolucionais para reconhecer pecas de roupa
(Fashion-MNIST), com a convolucao, a perda, o otimizador e o laco de treino
escritos a mao e uma ablacao do que realmente faz a CNN funcionar.

PROBLEMA
    Fashion-MNIST (Zalando Research, publico): 70 mil imagens 28x28 em tons de cinza
    de 10 classes de roupa (camiseta, calca, pulover, vestido, casaco, sandalia,
    camisa, tenis, bolsa, bota). E o "MNIST dificil": camisa x camiseta x pulover x
    casaco se confundem ate para pessoas. Divisao: 55 mil treino, 5 mil validacao
    (escolha de epoca/curvas) e 10 mil teste (so no fim).

MODELAGEM
    Classificador f_theta: R^{28x28} -> 10 logits, treinado por minimizacao da
    entropia cruzada media. Comparam-se: regressao softmax (linear), perceptron
    multicamadas (MLP), LeNet-5 (LeCun et al., 1998, adaptada a 28x28) e uma CNN
    "moderna" com blocos conv3x3-conv3x3-pool, normalizacao em lote (BatchNorm),
    dropout e aumento de dados (deslocamento aleatorio e espelhamento horizontal),
    em duas larguras (16 e 32 canais); a de 32 usa suavizacao de rotulos (0.1) e e
    avaliada tambem com TTA (media dos logits da imagem e da espelhada).
    Comparacao principal: 20 epocas no treino completo. Ablacao: 6 variantes x 2
    sementes, 20 mil imagens x 15 epocas, em 4 processos paralelos.

O QUE FOI ESCRITO A MAO (o PyTorch fornece tensores, autograd e camadas prontas)
    - Convolucao 2D por im2col (desdobra as janelas e faz um produto de matrizes),
      usada na camada Conv2dIm2col; verificada contra torch.nn.functional.conv2d.
    - Entropia cruzada estavel (log-softmax com o truque do maximo), com suavizacao
      de rotulos opcional.
    - Otimizador SGD com momento de Nesterov e decaimento de pesos.
    - Escalonamento cosseno da taxa de aprendizado, laco de treino/validacao,
      aumento de dados vetorizado, matriz de confusao, metricas.
    - Verificacao de gradientes por diferencas finitas centrais.

VERDADE DE REFERENCIA / AUTOVERIFICACAO
    - Conv2dIm2col == F.conv2d (saida e gradientes, varias formas, stride e padding).
    - Entropia cruzada propria == F.cross_entropy; SGD proprio == torch.optim.SGD
      (mesmos pesos apos 20 passos com nesterov e weight decay).
    - Gradiente do autograd == diferencas finitas (float64) numa CNN minima com a
      convolucao propria.
    - Matriz de confusao soma ao tamanho do teste; acuracia == traco / total.
    - Ordenacao esperada no teste: CNN-32 > LeNet > MLP > linear, e CNN-32 > 93%.
    - McNemar exato (pareado no teste): LeNet x MLP e CNN-32 x LeNet com p < 0.05;
      IC 95% da acuracia por bootstrap pareado.
    - Ablacao (2 sementes pareadas): sem nenhuma regularizacao a diferenca treino -
      validacao e mais que o dobro da rede completa, e a rede completa e melhor que
      a "nada" no teste nas duas sementes.

DADOS: baixados uma vez pelo torchvision para dados/ (ignorado pelo git, ~80 MB).
Sem internet e sem cache o script para com mensagem explicativa. Os pesos treinados
ficam em dados/cache04/ (apague a pasta para treinar tudo de novo, ~2-3 h em CPU).

Execucao: python 04_deep_learning_cnn.py  (gera figs/04_*.png e figs/04_saida.txt)
"""

import math
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PASTA = Path(__file__).resolve().parent
FIGS = PASTA / "figs"
DADOS = PASTA / "dados"
FIGS.mkdir(exist_ok=True)
# nucleos fisicos: com hyperthreading, mais threads deixam a convolucao em CPU mais LENTA (medido)
N_THREADS = max(1, (os.cpu_count() or 2) // 2)
torch.set_num_threads(N_THREADS)
CACHE = DADOS / "cache04"  # resultados de treino ja calculados (apague a pasta para treinar tudo de novo)

CLASSES = ["camiseta", "calca", "pulover", "vestido", "casaco", "sandalia", "camisa", "tenis", "bolsa", "bota"]

_LOG = []


def _saida(*args):
    """Imprime a linha e a acumula em _LOG para gravar em figs/04_saida.txt."""
    linha = " ".join(str(a) for a in args)
    print(linha)
    _LOG.append(linha)


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def carregar():
    """Fashion-MNIST como tensores float normalizados; cache em dados/."""
    try:
        import torchvision
        tr = torchvision.datasets.FashionMNIST(DADOS, train=True, download=True)
        te = torchvision.datasets.FashionMNIST(DADOS, train=False, download=True)
    except Exception as erro:  # sem internet e sem cache
        raise SystemExit(f"Nao foi possivel obter o Fashion-MNIST em {DADOS} ({erro}). Rode uma vez com internet.")
    Xtr = tr.data.float().div(255).unsqueeze(1)
    Xte = te.data.float().div(255).unsqueeze(1)
    mu, sd = Xtr.mean().item(), Xtr.std().item()
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(0))
    va, tr_i = perm[:5000], perm[5000:]
    return (Xtr[tr_i], tr.targets[tr_i]), (Xtr[va], tr.targets[va]), (Xte, te.targets)


def aumentar(x, g):
    """Aumento de dados por lote: deslocamento aleatorio de ate 2 px e espelhamento horizontal (50%)."""
    n = len(x)
    xp = F.pad(x, (2, 2, 2, 2), value=x.min().item())
    dy = torch.randint(0, 5, (n,), generator=g)
    dx = torch.randint(0, 5, (n,), generator=g)
    lin = (dy[:, None] + torch.arange(28)[None, :])[:, :, None]          # n x 28 x 1
    col = (dx[:, None] + torch.arange(28)[None, :])[:, None, :]          # n x 1 x 28
    out = xp[torch.arange(n)[:, None, None], 0, lin, col].unsqueeze(1)
    flip = torch.rand(n, generator=g) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


# ----------------------------------------------------------------------------
# Componentes escritos a mao
# ----------------------------------------------------------------------------
def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    """Convolucao (correlacao cruzada, como no PyTorch) por im2col: janelas viram colunas, conv vira matmul."""
    n, c, h, wd = x.shape
    co, _, kh, kw = w.shape
    if padding:
        x = F.pad(x, (padding,) * 4)
    ho = (x.shape[2] - kh) // stride + 1
    wo = (x.shape[3] - kw) // stride + 1
    # janelas: n x c x ho x wo x kh x kw (visao sem copia) -> n x (ho*wo) x (c*kh*kw)
    jan = x.unfold(2, kh, stride).unfold(3, kw, stride)
    col = jan.permute(0, 2, 3, 1, 4, 5).reshape(n, ho * wo, c * kh * kw)
    y = col @ w.reshape(co, -1).T                       # n x (ho*wo) x co
    if b is not None:
        y = y + b
    return y.permute(0, 2, 1).reshape(n, co, ho, wo)


class Conv2dIm2col(nn.Module):
    """Camada convolucional usando a convolucao propria (pesos inicializados como no nn.Conv2d)."""

    def __init__(self, c_in, c_out, k, padding=0):
        super().__init__()
        ref = nn.Conv2d(c_in, c_out, k, padding=padding)
        self.weight, self.bias, self.padding = nn.Parameter(ref.weight.data.clone()), nn.Parameter(ref.bias.data.clone()), padding

    def forward(self, x):
        return conv2d_im2col(x, self.weight, self.bias, padding=self.padding)


def entropia_cruzada(logits, y, suav=0.0):
    """-log softmax(z)_y medio, com log-sum-exp estavel; suav > 0 = suavizacao de rotulos
    (alvo = (1 - suav) no rotulo certo + suav/K espalhado em todas as classes)."""
    m = logits.max(1, keepdim=True).values
    logp = logits - (m + torch.log(torch.exp(logits - m).sum(1, keepdim=True)))
    nll = -logp.gather(1, y[:, None]).squeeze(1)
    return ((1 - suav) * nll - suav * logp.mean(1)).mean()


class SGDNesterov:
    """v <- mu v + (g + wd w);  w <- w - lr (g + wd w + mu v)  (mesma convencao do torch.optim.SGD)."""

    def __init__(self, params, lr, momento=0.9, wd=5e-4):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.mu, self.wd = lr, momento, wd
        self.v = [None] * len(self.params)

    @torch.no_grad()
    def passo(self):
        for i, p in enumerate(self.params):
            g = p.grad + self.wd * p
            self.v[i] = g.clone() if self.v[i] is None else self.mu * self.v[i] + g
            p -= self.lr * (g + self.mu * self.v[i])

    def zerar(self):
        for p in self.params:
            p.grad = None


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------
def linear():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))


def lenet():
    """LeNet-5 adaptada: entrada 28x28 com padding 2 (o original usava 32x32), tanh e pooling medio."""
    return nn.Sequential(
        Conv2dIm2col(1, 6, 5, padding=2), nn.Tanh(), nn.AvgPool2d(2),
        Conv2dIm2col(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
        nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10))


def cnn(bn=True, drop=True, blocos=2, largura=16):
    """CNN moderna: blocos [conv3x3 - (BN) - ReLU] x 2 + maxpool; cabeca densa com dropout."""
    camadas, c = [], 1
    for i in range(blocos):
        co = largura * 2 ** i
        for _ in range(2):
            camadas += [nn.Conv2d(c, co, 3, padding=1, bias=not bn)] + ([nn.BatchNorm2d(co)] if bn else []) + [nn.ReLU()]
            c = co
        camadas += [nn.MaxPool2d(2)] + ([nn.Dropout(0.1)] if drop else [])
    lado = 28 // 2 ** blocos
    camadas += [nn.Flatten(), nn.Linear(c * lado * lado, 128), nn.ReLU()] + ([nn.Dropout(0.4)] if drop else []) + [nn.Linear(128, 10)]
    return nn.Sequential(*camadas)


# ----------------------------------------------------------------------------
# Treino e avaliacao
# ----------------------------------------------------------------------------
@torch.no_grad()
def avaliar(modelo, X, y, lote=2000, tta=False):
    """Perda, acuracia e logits; tta = media dos logits da imagem e da imagem espelhada."""
    modelo.eval()
    logits = torch.cat([modelo(X[i:i + lote]) for i in range(0, len(X), lote)])
    if tta:
        logits = (logits + torch.cat([modelo(X[i:i + lote].flip(-1)) for i in range(0, len(X), lote)])) / 2
    return entropia_cruzada(logits, y).item(), (logits.argmax(1) == y).float().mean().item(), logits


def treinar(fabrica, treino, val, epocas, semente, lr=0.05, aug=False, lote=128, n_treino=None, suav=0.0):
    """Laco de treino escrito a mao: SGD Nesterov proprio + cosseno; devolve modelo e historico por epoca."""
    torch.manual_seed(semente)
    g = torch.Generator().manual_seed(semente)
    modelo = fabrica()
    X, y = treino
    if n_treino:
        X, y = X[:n_treino], y[:n_treino]
    opt = SGDNesterov(modelo.parameters(), lr)
    passos_tot = epocas * ((len(X) + lote - 1) // lote)
    k = 0
    hist = {"loss_tr": [], "acc_tr": [], "loss_va": [], "acc_va": []}
    for _ in range(epocas):
        modelo.train()
        perm = torch.randperm(len(X), generator=g)
        soma, acertos = 0.0, 0
        for i in range(0, len(X), lote):
            b = perm[i:i + lote]
            xb, yb = X[b], y[b]
            if aug:
                xb = aumentar(xb, g)
            opt.lr = lr * 0.5 * (1 + np.cos(np.pi * k / passos_tot))   # escalonamento cosseno
            out = modelo(xb)
            perda = entropia_cruzada(out, yb, suav)
            opt.zerar()
            perda.backward()
            opt.passo()
            k += 1
            soma += perda.item() * len(b)
            acertos += (out.argmax(1) == yb).sum().item()
        lv, av, _ = avaliar(modelo, *val)
        for chave, v in zip(hist, (soma / len(X), acertos / len(X), lv, av)):
            hist[chave].append(v)
    # acuracia de treino "limpa" (modo de avaliacao, sem dropout nem aumento) em ate 10 mil imagens
    hist["acc_tr_limpa"] = avaliar(modelo, X[:10000], y[:10000])[1]
    return modelo, hist


def treinar_cache(chave, fabrica, *args, **kw):
    """Treina ou recupera de dados/cache04 (o treino e deterministico por semente)."""
    arq = CACHE / f"{chave}.pt"
    if arq.exists():
        d = torch.load(arq, weights_only=False)
        modelo = fabrica()
        modelo.load_state_dict(d["estado"])
        return modelo, d["hist"], d["tempo"]
    t0 = time.perf_counter()
    modelo, hist = treinar(fabrica, *args, **kw)
    dt = time.perf_counter() - t0
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"estado": modelo.state_dict(), "hist": hist, "tempo": dt}, arq)
    return modelo, hist, dt


def n_param(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------
# Autoverificacao dos componentes
# ----------------------------------------------------------------------------
def verificar():
    torch.manual_seed(0)
    _saida("Verificacao dos componentes escritos a mao:")
    pior = 0.0
    for (n, c, h, co, k, s, p) in ((2, 1, 28, 6, 5, 1, 2), (3, 6, 14, 16, 5, 1, 0), (2, 4, 9, 3, 3, 2, 1), (1, 2, 7, 5, 1, 1, 0)):
        x = torch.randn(n, c, h, h, requires_grad=True)
        w = torch.randn(co, c, k, k, requires_grad=True)
        b = torch.randn(co, requires_grad=True)
        y1 = conv2d_im2col(x, w, b, s, p)
        y2 = F.conv2d(x, w, b, s, p)
        gy = torch.randn_like(y1)
        g1 = torch.autograd.grad(y1, (x, w, b), gy)
        g2 = torch.autograd.grad(y2, (x, w, b), gy)
        pior = max(pior, (y1 - y2).abs().max().item(), *[(a - c_).abs().max().item() for a, c_ in zip(g1, g2)])
    _saida(f"  conv im2col x F.conv2d (4 formas, saida e 3 gradientes): max |dif| = {pior:.2e}")
    assert pior < 1e-4

    z = torch.randn(64, 10) * 30
    yy = torch.randint(0, 10, (64,))
    d = max(abs(entropia_cruzada(z, yy, sv).item() - F.cross_entropy(z, yy, label_smoothing=sv).item()) for sv in (0.0, 0.1))
    _saida(f"  entropia cruzada propria x F.cross_entropy (logits ate +-100, suavizacao 0 e 0.1): max |dif| = {d:.2e}")
    assert d < 1e-5

    torch.manual_seed(1)
    m1 = mlp()
    m2 = mlp()
    m2.load_state_dict(m1.state_dict())
    m1.eval(), m2.eval()
    o1 = SGDNesterov(m1.parameters(), 0.05)
    o2 = torch.optim.SGD(m2.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)
    xb, yb = torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))
    for _ in range(20):
        o1.zerar()
        entropia_cruzada(m1(xb), yb).backward()
        o1.passo()
        o2.zero_grad()
        F.cross_entropy(m2(xb), yb).backward()
        o2.step()
    d = max((a - b_).abs().max().item() for a, b_ in zip(m1.parameters(), m2.parameters()))
    _saida(f"  SGD Nesterov proprio x torch.optim.SGD (20 passos): max |dif pesos| = {d:.2e}")
    assert d < 1e-5

    torch.manual_seed(2)
    rede = nn.Sequential(Conv2dIm2col(1, 2, 3, padding=1), nn.Tanh(), nn.Flatten(), nn.Linear(2 * 6 * 6, 3)).double()
    xb, yb = torch.randn(4, 1, 6, 6, dtype=torch.float64), torch.tensor([0, 2, 1, 2])
    rede.zero_grad()
    entropia_cruzada(rede(xb), yb).backward()
    pior, eps = 0.0, 1e-6
    for p in rede.parameters():
        plano = p.data.view(-1)
        for i in range(0, plano.numel(), max(1, plano.numel() // 7)):
            orig = plano[i].item()
            plano[i] = orig + eps
            fp = entropia_cruzada(rede(xb), yb).item()
            plano[i] = orig - eps
            fm = entropia_cruzada(rede(xb), yb).item()
            plano[i] = orig
            pior = max(pior, abs((fp - fm) / (2 * eps) - p.grad.view(-1)[i].item()))
    _saida(f"  gradiente autograd x diferencas finitas centrais (CNN minima, float64): max |dif| = {pior:.2e}")
    assert pior < 1e-7


# ----------------------------------------------------------------------------
# Ablacao (em processos paralelos, 1 thread cada)
# ----------------------------------------------------------------------------
EP_A, N_A = 15, 20000
VARIANTES = {
    "completa (BN+drop+aug)": (dict(), True),
    "sem BatchNorm": (dict(bn=False), True),
    "sem dropout": (dict(drop=False), True),
    "sem aumento de dados": (dict(), False),
    "1 bloco (rasa)": (dict(blocos=1), True),
    "nada (sem BN/drop/aug)": (dict(bn=False, drop=False), False),
}


def _job_ablacao(args):
    """Executado num processo filho: treina uma variante com uma semente e devolve as metricas."""
    nome, semente = args
    torch.set_num_threads(1)
    treino, val, teste = carregar()
    kw_cnn, aug = VARIANTES[nome]
    chave = "abl_" + "".join(c for c in nome if c.isalnum()) + f"_s{semente}"
    m, h, dt = treinar_cache(chave, lambda: cnn(largura=16, **kw_cnn), treino, val, EP_A, semente, aug=aug, n_treino=N_A)
    return nome, semente, avaliar(m, *teste)[1], h, dt


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    t_ini = time.perf_counter()
    verificar()
    treino, val, teste = carregar()
    _saida(f"\nFashion-MNIST: treino {len(treino[0])}, validacao {len(val[0])}, teste {len(teste[0])} (threads: {torch.get_num_threads()})")

    # --- 1. Comparacao principal (treino completo, semente 0) ---
    EP = 20
    modelos = {
        "Softmax linear": (linear, dict()),
        "MLP 256-128": (mlp, dict()),
        "LeNet-5 (conv propria)": (lenet, dict()),
        "CNN-16 BN+drop+aug": (lambda: cnn(largura=16), dict(aug=True)),
        "CNN-32 BN+drop+aug+LS": (lambda: cnn(largura=32), dict(aug=True, suav=0.1)),
    }
    _saida(f"\nComparacao principal: {EP} epocas, lote 128, SGD Nesterov (lr 0.05, wd 5e-4) + cosseno, semente 0")
    _saida("(acc treino = modo de avaliacao, 10 mil imagens de treino; LS = suavizacao de rotulos 0.1; TTA = media com a imagem espelhada)")
    _saida(f"{'modelo':<24}{'parametros':>11}{'acc treino':>12}{'acc val':>10}{'acc teste':>11}{'teste+TTA':>11}{'perda teste':>13}{'tempo (s)':>11}")
    hist, res, treinados = {}, {}, {}
    for nome, (fab, kw) in modelos.items():
        chave = "main_" + "".join(c for c in nome if c.isalnum())
        m, h, dt = treinar_cache(chave, fab, treino, val, EP, 0, **kw)
        lt, at, logits = avaliar(m, *teste)
        _, at2, logits2 = avaliar(m, *teste, tta=True) if kw.get("aug") else (lt, at, logits)  # TTA so para quem treinou com espelhamento
        hist[nome], treinados[nome] = h, m
        res[nome] = (lt, at, logits, at2, logits2)
        _saida(f"{nome:<24}{n_param(m):>11d}{h['acc_tr_limpa']:>12.4f}{h['acc_va'][-1]:>10.4f}{at:>11.4f}{(f"{at2:.4f}" if kw.get("aug") else "-"):>11}{lt:>13.4f}{dt:>11.0f}")
    acc = {n: r[1] for n, r in res.items()}
    assert acc["CNN-32 BN+drop+aug+LS"] > acc["LeNet-5 (conv propria)"] > acc["MLP 256-128"] > acc["Softmax linear"]
    assert acc["CNN-32 BN+drop+aug+LS"] > 0.93, "a CNN principal deveria passar de 93% no teste"

    # --- 1b. Significancia: IC por bootstrap e McNemar exato, ambos pareados no mesmo teste ---
    yte = teste[1]
    certo = {n: (r[2].argmax(1) == yte).numpy() for n, r in res.items()}
    certo["CNN-32 + TTA"] = (res["CNN-32 BN+drop+aug+LS"][4].argmax(1) == yte).numpy()
    idx = np.random.default_rng(0).integers(0, len(yte), (2000, len(yte)))  # mesmas reamostragens para todos
    _saida("\nIC 95% da acuracia de teste (bootstrap pareado, 2000 reamostragens):")
    for n, c in certo.items():
        lo, hi = np.percentile(c[idx].mean(1), [2.5, 97.5])
        _saida(f"  {n:<24} {c.mean():.4f}  [{lo:.4f}, {hi:.4f}]")

    def mcnemar(a, b):
        """Teste exato: so contam as imagens em que os modelos discordam (binomial com p = 1/2)."""
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        n, k = n01 + n10, min(n01, n10)
        return n01, n10, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

    _saida("\nMcNemar exato (A acerta e B erra / A erra e B acerta / p bilateral):")
    pares = [("CNN-32 + TTA", "CNN-32 BN+drop+aug+LS"), ("CNN-32 BN+drop+aug+LS", "CNN-16 BN+drop+aug"),
             ("CNN-32 BN+drop+aug+LS", "LeNet-5 (conv propria)"), ("LeNet-5 (conv propria)", "MLP 256-128"),
             ("MLP 256-128", "Softmax linear")]
    pval = {}
    for a, b in pares:
        n01, n10, p = mcnemar(certo[a], certo[b])
        pval[(a, b)] = p
        _saida(f"  {a:<22} x {b:<22} {n01:>5} {n10:>5}   p = {p:.2e}")
    assert pval[("LeNet-5 (conv propria)", "MLP 256-128")] < 0.05
    assert pval[("CNN-32 BN+drop+aug+LS", "LeNet-5 (conv propria)")] < 0.05

    # --- 2. Matriz de confusao e erros da melhor rede (com TTA) ---
    melhor = "CNN-32 BN+drop+aug+LS"
    logits = res[melhor][4]
    pred = logits.argmax(1)
    yte = teste[1]
    M = torch.zeros(10, 10, dtype=torch.long)
    M.index_put_((yte, pred), torch.ones_like(yte), accumulate=True)
    assert M.sum().item() == len(yte) and abs(M.trace().item() / len(yte) - res[melhor][3]) < 1e-6  # acc em float32
    _saida(f"\nAcuracia por classe ({melhor}, com TTA) e confusao mais frequente:")
    for c in range(10):
        lin = M[c].clone()
        lin[c] = -1
        _saida(f"  {CLASSES[c]:<10} {M[c, c].item() / M[c].sum().item():.3f}   mais confundida com {CLASSES[lin.argmax().item()]:<9} ({lin.max().item()} casos)")
    sup = [0, 2, 4, 6]  # camiseta, pulover, casaco, camisa
    dentro = M[sup][:, sup].sum().item() - M[sup, sup].sum().item()
    _saida(f"Erros dentro do grupo 'parte de cima' (camiseta/pulover/casaco/camisa): {dentro} de {len(yte) - M.trace().item()} erros ({100 * dentro / (len(yte) - M.trace().item()):.1f}%)")

    # --- 3. Ablacao: 6 variantes x 2 sementes, 20 mil imagens, 15 epocas ---
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(n, s) for n in VARIANTES for s in (0, 1)]
    with ProcessPoolExecutor(N_THREADS, mp_context=mp.get_context("spawn")) as ex:  # spawn: fork + threads do torch pode travar
        saidas = list(ex.map(_job_ablacao, jobs))
    tab_abl = {n: [None, None] for n in VARIANTES}
    hist_abl, gap_abl = {}, {n: [] for n in VARIANTES}
    for nome, sem, a, h, _ in saidas:
        tab_abl[nome][sem] = a
        gap_abl[nome].append(h["acc_tr_limpa"] - h["acc_va"][-1])
        if sem == 0:
            hist_abl[nome] = h
    _saida(f"\nAblacao da CNN-16: {N_A} imagens de treino, {EP_A} epocas, sementes 0 e 1 (pareadas), acuracia de teste:")
    _saida(f"{'variante':<26}{'semente 0':>11}{'semente 1':>11}{'media':>9}{'dif. vs completa (p.p.)':>26}{'acc treino - val':>18}")
    base = np.mean(tab_abl["completa (BN+drop+aug)"])
    for nome, accs in tab_abl.items():
        _saida(f"{nome:<26}{accs[0]:>11.4f}{accs[1]:>11.4f}{np.mean(accs):>9.4f}{100 * (np.mean(accs) - base):>26.2f}{np.mean(gap_abl[nome]):>18.4f}")
    # sem nenhuma regularizacao a rede decora o treino (gap grande) e generaliza pior, nas duas sementes
    assert np.mean(gap_abl["nada (sem BN/drop/aug)"]) > 2 * np.mean(gap_abl["completa (BN+drop+aug)"])
    assert all(a > b for a, b in zip(tab_abl["completa (BN+drop+aug)"], tab_abl["nada (sem BN/drop/aug)"]))

    # ------------------------------------------------------------------------
    # Figuras
    # ------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    cores = ["#555", "#e90", "#a4c", "#6bd", "#26a"]
    ep = np.arange(1, EP + 1)
    for (nome, h), c in zip(hist.items(), cores):
        axs[0].plot(ep, h["loss_va"], "-o", color=c, ms=3, label=nome)
        axs[1].plot(ep, 100 * (1 - np.array(h["acc_va"])), "-o", color=c, ms=3, label=nome)
    axs[0].set_ylabel("entropia cruzada (validacao)")
    axs[0].set_yscale("log")
    axs[1].set_ylabel("erro de validacao (%)")
    axs[1].set_yscale("log")
    for ax, t in zip(axs[:2], ("Perda de validacao por epoca", "Erro de validacao por epoca")):
        ax.set_xlabel("epoca")
        ax.set_title(t)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    ax = axs[2]
    epa = np.arange(1, EP_A + 1)
    for nome, c in zip(VARIANTES, ["#26a", "#d62", "#2a6", "#e90", "#a4c", "#555"]):
        h = hist_abl[nome]
        ax.plot(epa, 100 * (1 - np.array(h["acc_va"])), "-", color=c, label=nome)
        ax.plot(epa, 100 * (1 - np.array(h["acc_tr"])), ":", color=c, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("epoca")
    ax.set_ylabel("erro (%)  (pontilhado = treino em modo de treino)")
    ax.set_title(f"Ablacao: validacao x treino ({N_A // 1000} mil imagens, semente 0)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "04_curvas.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    nomes_a = list(tab_abl)[::-1]
    medias = [100 * np.mean(tab_abl[n]) for n in nomes_a]
    ax.barh(nomes_a, medias, color=["#d62" if n.startswith("nada") else "#26a" if n.startswith("completa") else "#8ab" for n in nomes_a])
    for i, n in enumerate(nomes_a):
        ax.plot([100 * v for v in tab_abl[n]], [i, i], "k|", ms=12)
        ax.text(min(medias) - 0.3, i, f"{100 * np.mean(tab_abl[n]):.2f}%   (treino-val {100 * np.mean(gap_abl[n]):+.1f} p.p.)", va="center", ha="right", fontsize=8, color="w")
    ax.set_xlim(min(medias) - 6, max(medias) + 0.5)
    ax.set_xlabel("acuracia de teste (%)  | = cada semente")
    ax.set_title(f"Ablacao da CNN ({N_A // 1000} mil imagens, {EP_A} epocas, 2 sementes)")
    fig.tight_layout()
    fig.savefig(FIGS / "04_ablacao.png", dpi=170)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    Mn = (M.float() / M.sum(1, keepdim=True)).numpy()
    ax = axs[0]
    ax.imshow(Mn, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if M[i, j] > 0:
                ax.text(j, i, str(M[i, j].item()), ha="center", va="center", fontsize=7.5, color="w" if Mn[i, j] > 0.5 else "k")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("previsto")
    ax.set_ylabel("verdadeiro")
    ax.set_title(f"Matriz de confusao no teste ({melhor}, TTA, acc = {100 * res[melhor][3]:.2f}%)", fontsize=9.5)
    ax = axs[1]
    prob = F.softmax(logits, 1)
    errados = torch.nonzero(pred != yte).squeeze(1)
    conf = prob[errados, pred[errados]]
    piores = errados[conf.argsort(descending=True)[:16]]
    ax.axis("off")
    ax.set_title("Os 16 erros mais confiantes (verdadeiro -> previsto, prob.)")
    for k, i in enumerate(piores.tolist()):
        sub = fig.add_axes([0.53 + (k % 4) * 0.115, 0.66 - (k // 4) * 0.205, 0.09, 0.15])
        sub.imshow(teste[0][i, 0], cmap="gray_r")
        sub.set_title(f"{CLASSES[yte[i]]}->{CLASSES[pred[i]]}\n{prob[i, pred[i]]:.2f}", fontsize=7)
        sub.axis("off")
    fig.savefig(FIGS / "04_confusao_erros.png", dpi=170)
    plt.close(fig)

    # filtros da 1a camada da LeNet (conv propria) e da CNN, mapas de ativacao de um exemplo
    w = treinados["LeNet-5 (conv propria)"][0].weight.detach()
    cw = treinados[melhor][0].weight.detach()
    x0 = teste[0][0:1]
    with torch.no_grad():
        mapas = treinados[melhor][:3].eval()(x0)[0]
    fig, axs = plt.subplots(3, 12, figsize=(14, 3.9))
    for ax in axs.ravel():
        ax.axis("off")
    for i in range(6):
        axs[0, i].imshow(w[i, 0], cmap="RdBu_r", vmin=-w.abs().max(), vmax=w.abs().max())
        axs[0, i].set_title(f"LeNet f{i}", fontsize=8)
    axs[0, 7].imshow(x0[0, 0], cmap="gray_r")
    axs[0, 7].set_title(f"entrada: {CLASSES[teste[1][0]]}", fontsize=8)
    for i in range(12):
        axs[1, i].imshow(cw[i, 0], cmap="RdBu_r", vmin=-cw.abs().max(), vmax=cw.abs().max())
        axs[1, i].set_title(f"CNN f{i}", fontsize=8)
        axs[2, i].imshow(mapas[i], cmap="magma")
        axs[2, i].set_title(f"mapa f{i}", fontsize=8)
    fig.suptitle("Filtros da 1a camada (LeNet 5x5 e CNN-32 3x3) e mapas de ativacao da CNN para uma imagem de teste")
    fig.tight_layout()
    fig.savefig(FIGS / "04_filtros_mapas.png", dpi=170)
    plt.close(fig)

    _saida(f"\nTodos os assert passaram. (tempo total ~ {time.perf_counter() - t_ini:.0f} s)")
    (FIGS / "04_saida.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
