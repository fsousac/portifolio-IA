"""
Projeto 1 - Aprendizado supervisionado nao neural: arvores, florestas e boosting
para diagnosticar tumores de mama (e o que acontece quando os rotulos tem ruido).

PROBLEMA
    Breast Cancer Wisconsin (Diagnostic), dado publico da UCI que acompanha o
    scikit-learn (569 exames, 30 atributos numericos extraidos de imagens de puncao,
    212 malignos / 357 benignos). Classe positiva = MALIGNO. Pede-se um classificador
    que (i) acerte muito, (ii) de probabilidades uteis e (iii) permita escolher o
    limiar de decisao quando um falso negativo (cancer nao detectado) custa muito mais
    que um falso positivo (exame extra). Um segundo conjunto, SINTETICO
    (make_classification), tem RUIDO DE ROTULO ajustavel (0% a 30% de rotulos
    trocados), para medir como cada metodo se degrada com a dificuldade.

MODELAGEM
    x em R^30 (atributos), y em {0,1} (1 = maligno). Avaliacao por validacao cruzada
    estratificada 5 dobras repetida com 5 sementes (25 dobras PAREADAS: todos os
    metodos veem exatamente as mesmas particoes). Metricas: acuracia, AUC (area sob a
    curva ROC), log-loss, custo esperado com FN = 10 e FP = 1.

ALGORITMOS IMPLEMENTADOS A MAO (numpy)
    - Arvore CART: divisao binaria x_j <= t escolhida por busca exaustiva vetorizada
      (ordenacao + somas acumuladas). Criterio = soma dos quadrados dos residuos (SSE).
      Para y em {0,1}, SSE de um no = n p (1 - p) = (n/2) Gini, logo a MESMA rotina e a
      arvore de classificacao por Gini e a arvore de regressao usada no boosting.
    - Random Forest: bootstrap + subespaco aleatorio de atributos (sqrt(p)) por no,
      erro OUT-OF-BAG (OOB) como estimativa de generalizacao sem conjunto de validacao.
    - Gradient Boosting com perda logistica (conceito alem da aula): cada arvore
      ajusta o pseudo-residuo y - p e cada folha recebe o passo de Newton
      sum(r) / sum(p(1-p)); taxa de aprendizado (shrinkage) nu.
    - Boosting ESTOCASTICO (subamostra de 70% por arvore) com profundidade {1,2,3} e
      numero de arvores escolhidos por validacao cruzada ANINHADA (3 dobras internas,
      so com o treino da dobra externa, curva de perda por estagio).
    - Rotation Forest (Rodriguez, Kuncheva e Alonso, 2006): cada arvore ve os atributos
      rodados por PCA em grupos aleatorios de 3 (cortes obliquos no espaco original).
    - Calibracao de Platt da Random Forest usando as probabilidades OUT-OF-BAG.
    - Stacking (Wolpert, 1992): logistica sobre os logits fora-da-dobra de 4 modelos
      (logistica, RF, Rotation Forest, boosting), com CV interna de 5 dobras.
    - Regressao logistica com L2 por Newton (IRLS), linha de base linear.
    - AUC por estatistica de Mann-Whitney, erro de calibracao (ECE), importancia por
      permutacao, limiar de Bayes por custo (analise de decisao).

VERDADE DE REFERENCIA / AUTOVERIFICACAO
    - Arvore propria == sklearn DecisionTreeClassifier (Gini) e DecisionTreeRegressor
      (SSE) nas mesmas profundidades: mesma acuracia de treino e >= 97% de concordancia
      nas predicoes de teste (empates entre divisoes equivalentes podem divergir).
    - Logistica propria == sklearn LogisticRegression (mesmo C): coeficientes iguais
      ate 1e-4. AUC propria == sklearn roc_auc_score.
    - Boosting com shrinkage (nu <= 0.3): log-loss de TREINO nao cresce a cada
      iteracao (descida). Com nu = 1 o passo de Newton inteiro pode ultrapassar o minimo
      e a perda sobe em algumas iteracoes (contado e reportado, nao exigido).
    - Erro OOB da floresta proximo do erro de validacao cruzada (diferenca < 3 p.p.).
    - Cada matriz de rotacao do Rotation Forest e ortogonal (R^T R = I); Platt == sklearn.
    - Estatisticos (pareados nas 25 dobras): ver os assert no main() (RF x arvore,
      shrinkage, calibracao, stacking, ruido de rotulo, custo com limiar de Bayes).

Execucao: python 01_supervisionado_arvores.py  (gera figs/01_*.png e figs/01_saida.txt)
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # processos paralelos: 1 thread de BLAS cada (evita sobreinscricao)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import StratifiedKFold

PASTA = Path(__file__).resolve().parent
FIGS = PASTA / "figs"
FIGS.mkdir(exist_ok=True)

_LOG = []


def _saida(*args):
    """Imprime a linha e a acumula em _LOG para gravar em figs/01_saida.txt."""
    linha = " ".join(str(a) for a in args)
    print(linha)
    _LOG.append(linha)


# ----------------------------------------------------------------------------
# Arvore CART (criterio SSE == Gini para rotulos 0/1)
# ----------------------------------------------------------------------------
class Arvore:
    """Arvore binaria guardada em vetores (feat, thr, esq, dir, valor); folha tem feat = -1."""

    def __init__(self, prof_max=None, min_folha=1, max_feat=None, rng=None):
        self.prof_max = prof_max if prof_max is not None else 10 ** 9
        self.min_folha = min_folha
        self.max_feat = max_feat  # None = todos os atributos; inteiro = subespaco aleatorio por no
        self.rng = rng

    def _melhor_divisao(self, X, y, idx):
        """Busca exaustiva: para cada atributo ordena o no e avalia todos os cortes com somas acumuladas."""
        n = len(idx)
        p = X.shape[1]
        feats = np.arange(p) if self.max_feat is None else self.rng.choice(p, self.max_feat, replace=False)
        sse_pai = ((y[idx] - y[idx].mean()) ** 2).sum()
        melhor = (sse_pai - 1e-12, -1, 0.0)
        k = self.min_folha
        for j in feats:
            xs = X[idx, j]
            o = np.argsort(xs, kind="stable")
            xs, ys = xs[o], y[idx][o]
            cs, cs2 = np.cumsum(ys), np.cumsum(ys * ys)
            nl = np.arange(1, n)                       # tamanho do lado esquerdo para corte apos a posicao i
            sl, sl2 = cs[:-1], cs2[:-1]
            sr, sr2 = cs[-1] - sl, cs2[-1] - sl2
            sse = (sl2 - sl * sl / nl) + (sr2 - sr * sr / (n - nl))
            ok = (xs[1:] > xs[:-1]) & (nl >= k) & (n - nl >= k)   # so corta entre valores distintos
            if not ok.any():
                continue
            sse = np.where(ok, sse, np.inf)
            i = int(np.argmin(sse))
            if sse[i] < melhor[0]:
                melhor = (sse[i], j, (xs[i] + xs[i + 1]) / 2)  # limiar no ponto medio, como no sklearn
        return melhor[1], melhor[2]

    def fit(self, X, y):
        self.feat, self.thr, self.esq, self.dir, self.valor = [], [], [], [], []
        self.folhas = []  # (no, indices) para o boosting recalcular o valor das folhas
        pilha = [(np.arange(len(y)), 0, None, None)]
        while pilha:
            idx, prof, pai, lado = pilha.pop()
            no = len(self.feat)
            if pai is not None:
                (self.esq if lado == 0 else self.dir)[pai] = no
            self.feat.append(-1)
            self.thr.append(0.0)
            self.esq.append(-1)
            self.dir.append(-1)
            self.valor.append(y[idx].mean())
            j = -1
            if prof < self.prof_max and len(idx) >= 2 * self.min_folha and np.ptp(y[idx]) > 0:
                j, t = self._melhor_divisao(X, y, idx)
            if j < 0:
                self.folhas.append((no, idx))
                continue
            self.feat[no], self.thr[no] = j, t
            m = X[idx, j] <= t
            pilha.append((idx[~m], prof + 1, no, 1))
            pilha.append((idx[m], prof + 1, no, 0))
        for nome in ("feat", "thr", "esq", "dir", "valor"):
            setattr(self, nome, np.array(getattr(self, nome)))
        return self

    def folha_de(self, X):
        """Desce todos os exemplos ao mesmo tempo (vetorizado) e devolve o indice da folha."""
        no = np.zeros(len(X), dtype=int)
        while True:
            interno = self.feat[no] >= 0
            if not interno.any():
                return no
            ni = no[interno]
            vai_esq = X[np.flatnonzero(interno), self.feat[ni]] <= self.thr[ni]
            no[interno] = np.where(vai_esq, self.esq[ni], self.dir[ni])

    def predict(self, X):
        return self.valor[self.folha_de(X)]


# ----------------------------------------------------------------------------
# Random Forest, Gradient Boosting e regressao logistica
# ----------------------------------------------------------------------------
class Floresta:
    def __init__(self, n_arv=150, max_feat="sqrt", min_folha=1, semente=0):
        self.n_arv, self.max_feat, self.min_folha, self.semente = n_arv, max_feat, min_folha, semente

    def fit(self, X, y):
        n, p = X.shape
        rng = np.random.default_rng(self.semente)
        mf = int(np.sqrt(p)) if self.max_feat == "sqrt" else None
        self.arvs = []
        soma, cont = np.zeros(n), np.zeros(n)
        self.oob_erro = []
        for b in range(self.n_arv):
            boot = rng.integers(0, n, n)
            arv = Arvore(min_folha=self.min_folha, max_feat=mf, rng=rng).fit(X[boot], y[boot])
            self.arvs.append(arv)
            fora = np.setdiff1d(np.arange(n), boot)
            soma[fora] += arv.predict(X[fora])
            cont[fora] += 1
            vis = cont > 0
            self.oob_erro.append(np.mean((soma[vis] / cont[vis] > 0.5) != y[vis]))
        self.oob_p = np.where(cont > 0, soma / np.maximum(cont, 1), y.mean())
        return self

    def predict_proba(self, X, n=None):
        return np.mean([a.predict(X) for a in self.arvs[:n]], axis=0)


def sigm(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class Boosting:
    """Gradient boosting com perda logistica: F_m = F_{m-1} + nu * h_m, folhas com passo de Newton."""

    def __init__(self, n_iter=150, nu=0.1, prof=2, min_folha=5, sub=1.0, semente=0):
        self.n_iter, self.nu, self.prof, self.min_folha = n_iter, nu, prof, min_folha
        self.sub, self.semente = sub, semente  # sub < 1: boosting ESTOCASTICO (Friedman, 2002)

    def fit(self, X, y):
        rng = np.random.default_rng(self.semente)
        n = len(y)
        pm = y.mean()
        self.f0 = np.log(pm / (1 - pm))
        F = np.full(n, self.f0)
        self.arvs, self.loss_treino = [], []
        for _ in range(self.n_iter):
            p = sigm(F)
            r = y - p                                  # pseudo-residuo = -gradiente da log-loss
            lin = np.arange(n) if self.sub >= 1 else np.sort(rng.choice(n, int(self.sub * n), replace=False))
            arv = Arvore(prof_max=self.prof, min_folha=self.min_folha).fit(X[lin], r[lin])
            for no, idx in arv.folhas:                 # passo de Newton em cada folha
                g = lin[idx]
                arv.valor[no] = r[g].sum() / max((p[g] * (1 - p[g])).sum(), 1e-12)
            F = F + self.nu * arv.predict(X)
            self.arvs.append(arv)
            self.loss_treino.append(logloss(y, sigm(F)))
        return self

    def decision(self, X, n=None):
        return self.f0 + self.nu * sum(a.predict(X) for a in self.arvs[:n])

    def predict_proba(self, X, n=None):
        return sigm(self.decision(X, n))

    def perda_por_estagio(self, X, y):
        """log-loss depois de 1, 2, ..., n_iter arvores (uma passada so)."""
        F = np.full(len(X), self.f0)
        out = np.empty(len(self.arvs))
        for m, a in enumerate(self.arvs):
            F = F + self.nu * a.predict(X)
            out[m] = logloss(y, sigm(F))
        return out


class BoostingCV:
    """Boosting estocastico com profundidade e numero de arvores escolhidos por CV INTERNA
    (3 dobras, so com os dados de treino da dobra externa: validacao cruzada aninhada)."""

    def __init__(self, profs=(1, 2, 3), n_max=400, nu=0.05, sub=0.7, semente=0):
        self.profs, self.n_max, self.nu, self.sub, self.semente = profs, n_max, nu, sub, semente

    def fit(self, X, y):
        internas = list(StratifiedKFold(3, shuffle=True, random_state=self.semente).split(X, y))
        self.perdas = {}
        for d in self.profs:
            ll = np.zeros(self.n_max)
            for tr, va in internas:
                g = Boosting(self.n_max, self.nu, d, sub=self.sub, semente=self.semente).fit(X[tr], y[tr])
                ll += g.perda_por_estagio(X[va], y[va]) * len(va)
            self.perdas[d] = ll / len(y)
        self.prof = min(self.perdas, key=lambda d: self.perdas[d].min())
        self.n = int(np.argmin(self.perdas[self.prof])) + 1
        self.modelo = Boosting(self.n, self.nu, self.prof, sub=self.sub, semente=self.semente).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.modelo.predict_proba(X)


class RotationForest:
    """Rotation Forest (Rodriguez, Kuncheva e Alonso, 2006). Para cada arvore: divide os
    atributos (padronizados) em grupos aleatorios de m, roda PCA em cada grupo com uma
    amostra de 75% (de um subconjunto aleatorio de classes) e monta uma matriz de rotacao
    bloco-diagonal R (ortogonal). A arvore CART completa e treinada em X R. Os cortes,
    paralelos aos eixos rodados, ficam OBLIQUOS no espaco original."""

    def __init__(self, n_arv=100, m=3, frac=0.75, semente=0):
        self.n_arv, self.m, self.frac, self.semente = n_arv, m, frac, semente

    def fit(self, X, y):
        rng = np.random.default_rng(self.semente)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        p = Z.shape[1]
        self.rots, self.arvs = [], []
        for _ in range(self.n_arv):
            Rm = np.zeros((p, p))
            for g in np.array_split(rng.permutation(p), int(np.ceil(p / self.m))):
                classes = rng.choice(2, rng.integers(1, 3), replace=False)
                pool = np.flatnonzero(np.isin(y, classes))
                am = rng.choice(pool, max(len(g) + 1, int(self.frac * len(pool))), replace=True)
                Zg = Z[np.ix_(am, g)]
                _, _, Vt = np.linalg.svd(Zg - Zg.mean(0), full_matrices=True)
                Rm[np.ix_(g, g)] = Vt.T
            self.rots.append(Rm)
            self.arvs.append(Arvore().fit(Z @ Rm, y))
        return self

    def predict_proba(self, X):
        Z = (X - self.mu) / self.sd
        return np.mean([a.predict(Z @ Rm) for a, Rm in zip(self.arvs, self.rots)], axis=0)


def logistica_newton(X, y, C=1.0, iters=50):
    """Minimiza 0.5||w||^2 + C * sum log-loss (intercepto sem penalidade) por Newton (IRLS)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    R = np.eye(len(w)) / C
    R[-1, -1] = 0
    for _ in range(iters):
        p = sigm(Xb @ w)
        g = Xb.T @ (p - y) + R @ w
        Hs = Xb.T @ (Xb * (p * (1 - p))[:, None]) + R
        passo = np.linalg.solve(Hs, g)
        w -= passo
        if np.abs(passo).max() < 1e-10:
            break
    return w


class Logistica:
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12   # padronizacao calculada so no treino
        self.w = logistica_newton((X - self.mu) / self.sd, y)
        return self

    def predict_proba(self, X):
        return sigm(((X - self.mu) / self.sd) @ self.w[:-1] + self.w[-1])


class ArvoreCls:
    def __init__(self, prof=None, min_folha=1):
        self.a = Arvore(prof_max=prof, min_folha=min_folha)

    def fit(self, X, y):
        self.a.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.a.predict(X)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class FlorestaPlatt:
    """Random Forest + calibracao de Platt: P(y=1|s) = sigm(a s + b), ajustada nas
    probabilidades OUT-OF-BAG s (nao precisa separar um conjunto de calibracao)."""

    def __init__(self, n_arv=100, semente=0):
        self.n_arv, self.semente = n_arv, semente

    def fit(self, X, y):
        self.rf = Floresta(n_arv=self.n_arv, semente=self.semente).fit(X, y)
        self.w = logistica_newton(self.rf.oob_p[:, None], y, C=1e6)
        return self

    def predict_proba(self, X):
        return sigm(self.w[0] * self.rf.predict_proba(X) + self.w[1])


class Empilhamento:
    """Stacking (Wolpert, 1992): os modelos de base geram probabilidades FORA-DA-DOBRA por
    CV interna de 5 dobras; um meta-modelo logistico (L2) aprende a combina-las a partir dos
    logits. Depois as bases sao reajustadas em todo o treino."""

    def __init__(self, semente=0):
        self.semente = semente

    def bases(self):
        return [Logistica(), Floresta(n_arv=100, semente=self.semente), RotationForest(n_arv=50, semente=self.semente), Boosting(150, 0.1)]

    def fit(self, X, y):
        Z = np.zeros((len(y), len(self.bases())))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + self.semente).split(X, y):
            for j, b in enumerate(self.bases()):
                Z[va, j] = b.fit(X[tr], y[tr]).predict_proba(X[va])
        self.w = logistica_newton(logit(Z), y, C=1.0)
        self.ms = [b.fit(X, y) for b in self.bases()]
        return self

    def predict_proba(self, X):
        Z = np.column_stack([m.predict_proba(X) for m in self.ms])
        return sigm(logit(Z) @ self.w[:-1] + self.w[-1])


# ----------------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------------
def auc(y, s):
    """AUC = P(score de um positivo > score de um negativo), via postos (Mann-Whitney) com empates."""
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    postos = np.empty(len(s))
    i = 0
    while i < len(s):  # posto medio para empates
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        postos[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return (postos[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(y, p, bins=10):
    """Erro de calibracao esperado: media ponderada de |freq. observada - prob. media| em 10 faixas."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[b == k].mean() - p[b == k].mean()) * np.mean(b == k) for k in range(bins) if np.any(b == k))


def ic95(v):
    """Media e meia-largura do IC 95% (t de Student) de uma amostra."""
    v = np.asarray(v, float)
    t = {4: 2.776, 9: 2.262, 24: 2.064}.get(len(v) - 1, 1.96)
    return v.mean(), t * v.std(ddof=1) / np.sqrt(len(v))


CUSTO_FN, CUSTO_FP = 10.0, 1.0


def custo(y, p, lim):
    pred = p >= lim
    return (CUSTO_FN * np.sum(~pred & (y == 1)) + CUSTO_FP * np.sum(pred & (y == 0))) / len(y)


def dobras(y, sementes):
    """Particoes estratificadas 5x, repetidas; as mesmas para todos os metodos (comparacao pareada)."""
    return [(tr, te) for s in sementes for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(len(y)), y)]


# ----------------------------------------------------------------------------
# Autoverificacao contra o scikit-learn
# ----------------------------------------------------------------------------
def verificar(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=7).split(X, y)))
    _saida("Verificacao contra o scikit-learn (referencia independente):")
    _saida(f"{'modelo':<26}{'acc treino (nosso/sk)':>24}{'concordancia teste':>21}")
    for prof in (1, 2, 3, 5, None):
        nos = Arvore(prof_max=prof).fit(X[tr], y[tr])
        sk = DecisionTreeClassifier(max_depth=prof, random_state=0).fit(X[tr], y[tr])
        a_n = np.mean((nos.predict(X[tr]) > 0.5) == y[tr])
        a_s = sk.score(X[tr], y[tr])
        conc = np.mean((nos.predict(X[te]) > 0.5) == sk.predict(X[te]))
        _saida(f"{'CART Gini prof=' + str(prof):<26}{a_n:>12.4f} / {a_s:<9.4f}{conc:>21.4f}")
        assert abs(a_n - a_s) < 1e-12 and conc >= 0.97
    r = np.random.default_rng(1).normal(size=len(tr))
    for prof in (2, 4):
        nos = Arvore(prof_max=prof).fit(X[tr], r)
        sk = DecisionTreeRegressor(max_depth=prof, random_state=0).fit(X[tr], r)
        dif = np.abs(nos.predict(X[te]) - sk.predict(X[te])).max()
        _saida(f"{'CART SSE prof=' + str(prof):<26}{'max |dif| teste =':>24}{dif:>21.2e}")
        assert dif < 1e-9
    m = Logistica().fit(X[tr], y[tr])
    sk = LogisticRegression(C=1.0, tol=1e-12, max_iter=10000).fit((X[tr] - m.mu) / m.sd, y[tr])
    dif = max(np.abs(sk.coef_[0] - m.w[:-1]).max(), abs(sk.intercept_[0] - m.w[-1]))
    _saida(f"{'Logistica L2 (C=1)':<26}{'max |dif| coef =':>24}{dif:>21.2e}")
    assert dif < 1e-4
    p = m.predict_proba(X[te])
    assert abs(auc(y[te], p) - roc_auc_score(y[te], p)) < 1e-12
    pq = np.round(p, 1)  # forca empates
    assert abs(auc(y[te], pq) - roc_auc_score(y[te], pq)) < 1e-12
    _saida(f"{'AUC propria vs sklearn':<26}{'iguais (com empates)':>24}")
    rot = RotationForest(n_arv=5, semente=0).fit(X[tr], y[tr])
    orto = max(np.abs(Rm.T @ Rm - np.eye(X.shape[1])).max() for Rm in rot.rots)
    _saida(f"{'Rotation Forest':<26}{'max |R^T R - I| =':>24}{orto:>21.2e}")
    assert orto < 1e-10, "cada rotacao deve ser ortogonal"
    s_ = np.random.default_rng(3).random(len(tr))
    w = logistica_newton(s_[:, None], y[tr], C=1e6)
    sk = LogisticRegression(C=1e6, tol=1e-12, max_iter=10000).fit(s_[:, None], y[tr])
    dif = max(abs(sk.coef_[0, 0] - w[0]), abs(sk.intercept_[0] - w[1]))
    _saida(f"{'Platt (logistica 1-D)':<26}{'max |dif| (a, b) =':>24}{dif:>21.2e}")
    assert dif < 1e-3


# ----------------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------------
def metodos():
    """Metodos comparados; a ablacao esta embutida (arvore rasa/profunda, bagging x RF, RF com/sem
    calibracao, nu=1 x nu=0.1 x boosting aninhado)."""
    return {
        "Logistica L2": lambda s: Logistica(),
        "CART prof=3": lambda s: ArvoreCls(prof=3),
        "CART profunda": lambda s: ArvoreCls(),
        "Bagging (todos atrib.)": lambda s: Floresta(n_arv=100, max_feat=None, semente=s),
        "Random Forest": lambda s: Floresta(n_arv=100, semente=s),
        "RF + Platt (OOB)": lambda s: FlorestaPlatt(n_arv=100, semente=s),
        "Rotation Forest": lambda s: RotationForest(n_arv=100, semente=s),
        "Boosting nu=1.0": lambda s: Boosting(n_iter=150, nu=1.0),
        "Boosting nu=0.1": lambda s: Boosting(n_iter=150, nu=0.1),
        "Boosting estoc. CV aninh.": lambda s: BoostingCV(semente=s),
        "Stacking (4 bases)": lambda s: Empilhamento(semente=s),
    }


def _ajusta(args):
    """Executado em processo filho: ajusta um metodo numa dobra e devolve as probabilidades de teste."""
    nome, k, X, y, tr, te = args
    m = metodos()[nome](k).fit(X[tr], y[tr])
    extra = (m.prof, m.n) if isinstance(m, BoostingCV) else None
    return nome, k, m.predict_proba(X[te]), extra


def rodar_paralelo(jobs):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        return list(ex.map(_ajusta, jobs, chunksize=1))


def validacao_cruzada(X, y, sementes, nomes=None):
    """Roda todos os metodos nas mesmas dobras; devolve metricas por dobra e probabilidades fora-da-dobra."""
    nomes = nomes or list(metodos())
    ds = dobras(y, sementes)
    saidas = rodar_paralelo([(n, k, X, y, tr, te) for k, (tr, te) in enumerate(ds) for n in nomes])
    res = {n: {"acc": [], "auc": [], "ll": []} for n in nomes}
    oof = {n: np.zeros((len(sementes), len(y))) for n in nomes}
    escolhas = []
    for n, k, p, extra in sorted(saidas, key=lambda t: t[1]):
        te = ds[k][1]
        res[n]["acc"].append(np.mean((p > 0.5) == y[te]))
        res[n]["auc"].append(auc(y[te], p))
        res[n]["ll"].append(logloss(y[te], p))
        oof[n][k // 5, te] = p
        if extra:
            escolhas.append(extra)
    return res, oof, escolhas


def importancia_permutacao(modelo, X, y, rng, rep=5):
    """Queda de AUC ao embaralhar cada atributo (no conjunto de teste)."""
    base = auc(y, modelo.predict_proba(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        for _ in range(rep):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            imp[j] += (base - auc(y, modelo.predict_proba(Xp))) / rep
    return imp


def main():
    t_ini = time.perf_counter()
    dados = load_breast_cancer()
    X = dados.data
    y = (dados.target == 0).astype(float)  # 1 = maligno
    nomes_at = list(dados.feature_names)
    _saida(f"Breast Cancer Wisconsin: {X.shape[0]} exames, {X.shape[1]} atributos, {int(y.sum())} malignos\n")

    verificar(X, y)

    # --- 1. Comparacao pareada por validacao cruzada 5x5 ---
    sementes = [0, 1, 2, 3, 4]
    res, oof, escolhas = validacao_cruzada(X, y, sementes)
    _saida("\nValidacao cruzada estratificada 5 dobras x 5 sementes (25 dobras pareadas), media +- IC95%:")
    _saida("(ECE = erro de calibracao das probabilidades fora-da-dobra, media das 5 sementes)")
    _saida(f"{'metodo':<27}{'acuracia':>17}{'AUC':>17}{'log-loss':>17}{'ECE':>8}")
    for n, r in res.items():
        cols = "".join(f"{ic95(r[m])[0]:>10.4f} +-{ic95(r[m])[1]:.4f}" for m in ("acc", "auc", "ll"))
        e_ = np.mean([ece(y, oof[n][s_]) for s_ in range(len(sementes))])
        _saida(f"{n:<27}{cols}{e_:>8.4f}")
    from collections import Counter
    _saida("Boosting aninhado, (profundidade, n de arvores) escolhidos pela CV interna nas 25 dobras: " + ", ".join(f"prof {d}: {c}x" for d, c in sorted(Counter(d for d, _ in escolhas).items())) + f"; n medio = {np.mean([n_ for _, n_ in escolhas]):.0f} (min {min(n_ for _, n_ in escolhas)}, max {max(n_ for _, n_ in escolhas)})")
    for a_, b_ in (("Rotation Forest", "Random Forest"), ("Stacking (4 bases)", "Random Forest"), ("Stacking (4 bases)", "Logistica L2"), ("Boosting estoc. CV aninh.", "Boosting nu=0.1"), ("RF + Platt (OOB)", "Random Forest")):
        for m_ in ("acc", "ll"):
            dd = np.array(res[a_][m_]) - np.array(res[b_][m_])
            _saida(f"Pareado {m_:<3} {a_} - {b_}: {ic95(dd)[0]:+.4f} +- {ic95(dd)[1]:.4f} (IC95%), {a_} melhor em {np.sum(dd > 0) if m_ == 'acc' else np.sum(dd < 0)}/25 dobras")
    for a_ in ("Rotation Forest", "Stacking (4 bases)"):
        dd = np.array(res[a_]["acc"]) - np.array(res["Random Forest"]["acc"])
        assert ic95(dd)[0] - ic95(dd)[1] > 0, f"{a_} deveria ter acuracia maior que a RF (IC95% pareado acima de 0)"
    ece_m = lambda n: np.mean([ece(y, oof[n][s_]) for s_ in range(len(sementes))])
    assert ece_m("RF + Platt (OOB)") < ece_m("Random Forest") and np.mean(res["RF + Platt (OOB)"]["ll"]) < np.mean(res["Random Forest"]["ll"]), "Platt deveria calibrar a RF"
    d = np.array(res["Random Forest"]["auc"]) - np.array(res["CART profunda"]["auc"])
    md, hd = ic95(d)
    _saida(f"\nDiferenca pareada AUC(RF) - AUC(CART profunda) = {md:.4f} +- {hd:.4f} (IC95%); RF melhor em {np.sum(d > 0)}/25 dobras")
    assert md - hd > 0, "RF deveria ter AUC maior que a arvore profunda"
    d2 = np.array(res["Boosting nu=0.1"]["ll"]) - np.array(res["Boosting nu=1.0"]["ll"])
    _saida(f"Diferenca pareada log-loss(nu=0.1) - log-loss(nu=1.0) = {ic95(d2)[0]:.4f} +- {ic95(d2)[1]:.4f}")
    assert ic95(d2)[0] < 0, "shrinkage deveria reduzir a log-loss"

    # --- 2. OOB x validacao cruzada, curvas por numero de arvores ---
    tr, te = next(iter(StratifiedKFold(4, shuffle=True, random_state=11).split(X, y)))
    rf = Floresta(n_arv=300, semente=0).fit(X[tr], y[tr])
    err_te_rf = [np.mean((rf.predict_proba(X[te], n) > 0.5) != y[te]) for n in range(1, 301)]
    acc_cv_rf = ic95(res["Random Forest"]["acc"])[0]
    _saida(f"\nRandom Forest (300 arvores, 75% treino): erro OOB = {rf.oob_erro[-1]:.4f}, erro de teste = {err_te_rf[-1]:.4f}, erro CV 5x5 (100 arv.) = {1 - acc_cv_rf:.4f}")
    assert abs(rf.oob_erro[-1] - (1 - acc_cv_rf)) < 0.03, "OOB deveria estimar o erro de generalizacao"
    gbs = {nu: Boosting(n_iter=300, nu=nu).fit(X[tr], y[tr]) for nu in (1.0, 0.3, 0.1)}
    ll_te = {nu: [logloss(y[te], g.predict_proba(X[te], n)) for n in range(1, 301)] for nu, g in gbs.items()}
    for nu, g in gbs.items():
        sobe = int(np.sum(np.diff(g.loss_treino) > 1e-12))
        if nu <= 0.3:  # passo de Newton completo (nu=1) pode ultrapassar; com shrinkage deve descer sempre
            assert sobe == 0, "log-loss de treino do boosting com shrinkage deveria ser monotona"
        _saida(f"Boosting nu={nu}: log-loss treino final = {g.loss_treino[-1]:.4f}, iteracoes em que a perda de treino SUBIU = {sobe}, teste min = {min(ll_te[nu]):.4f} (iter {int(np.argmin(ll_te[nu])) + 1}), teste final = {ll_te[nu][-1]:.4f}")

    # --- 3. Ruido de rotulo (dificuldade ajustavel) em dados sinteticos ---
    _saida("\nDados sinteticos (1000 x 20, 6 informativos), acuracia no rotulo LIMPO de teste, CV 5x2:")
    ruidos = [0.0, 0.1, 0.2, 0.3]
    nomes_r = ["Logistica L2", "CART profunda", "Random Forest", "Rotation Forest", "Boosting nu=0.1", "Boosting estoc. CV aninh."]
    _saida(f"{'ruido':<8}" + "".join(f"{n[:16]:>17}" for n in nomes_r))
    tab_ruido = {n: [] for n in nomes_r}
    Xs, ys = make_classification(1000, 20, n_informative=6, n_redundant=4, flip_y=0, class_sep=1.0, random_state=3)
    ys = ys.astype(float)
    ds2 = dobras(ys, [0, 1])
    for rz in ruidos:
        rng = np.random.default_rng(42)
        yr = ys.copy()
        troca = rng.random(len(ys)) < rz
        yr[troca] = 1 - yr[troca]
        acc = {n: [None] * len(ds2) for n in nomes_r}
        # treina com o rotulo ruidoso (yr) e avalia no rotulo verdadeiro (ys)
        for n, k, p, _ in rodar_paralelo([(n, k, Xs, yr, tr2, te2) for k, (tr2, te2) in enumerate(ds2) for n in nomes_r]):
            acc[n][k] = np.mean((p > 0.5) == ys[ds2[k][1]])
        for n in nomes_r:
            tab_ruido[n].append(acc[n])
        _saida(f"{rz:<8.1f}" + "".join(f"{np.mean(acc[n]):>11.4f}+-{ic95(acc[n])[1]:.3f}" for n in nomes_r))
    queda = {n: np.mean(tab_ruido[n][0]) - np.mean(tab_ruido[n][-1]) for n in nomes_r}
    _saida("Queda de acuracia de 0% para 30% de ruido: " + ", ".join(f"{n} {v:.3f}" for n, v in queda.items()))
    assert np.mean(tab_ruido["Random Forest"][-1]) > np.mean(tab_ruido["CART profunda"][-1]) + 0.05
    assert queda["CART profunda"] > queda["Random Forest"]
    assert all(np.mean(tab_ruido["Boosting estoc. CV aninh."][i]) > np.mean(tab_ruido["Boosting nu=0.1"][i]) for i in range(1, len(ruidos))), "CV aninhada deveria proteger o boosting do ruido"

    # --- 4. Limiar por custo (FN = 10, FP = 1) com probabilidades fora-da-dobra ---
    _saida(f"\nDecisao com custo assimetrico (FN = {CUSTO_FN:g}, FP = {CUSTO_FP:g}), probabilidades fora-da-dobra (5 sementes):")
    _saida("(limiar de Bayes 1/11 e fixado ANTES de ver os dados: estimativa honesta; 'melhor lim' e escolhido a posteriori: otimista)")
    limiares = np.linspace(0.01, 0.99, 99)
    lt = CUSTO_FP / (CUSTO_FP + CUSTO_FN)  # limiar de Bayes se p for calibrada
    _saida(f"{'metodo':<27}{'custo lim=0.5':>14}{'custo lim=1/11':>15}{'FN em 1/11':>12}{'melhor lim':>12}{'custo':>9}")
    curvas_custo, custo_bayes = {}, {}
    for n in ["Logistica L2", "Random Forest", "RF + Platt (OOB)", "Rotation Forest", "Boosting estoc. CV aninh.", "Stacking (4 bases)"]:
        cs = np.array([[custo(y, oof[n][s], l) for l in limiares] for s in range(len(sementes))]).mean(0)
        curvas_custo[n] = cs
        custo_bayes[n] = c_teor = np.mean([custo(y, oof[n][s], lt) for s in range(len(sementes))])
        c05 = np.mean([custo(y, oof[n][s], 0.5) for s in range(len(sementes))])
        fn = np.mean([np.sum((oof[n][s] < lt) & (y == 1)) for s in range(len(sementes))])
        _saida(f"{n:<27}{c05:>14.4f}{c_teor:>15.4f}{fn:>12.1f}{limiares[np.argmin(cs)]:>12.2f}{cs.min():>9.4f}")
        assert c_teor < c05, "limiar de Bayes deveria reduzir o custo"
    assert custo_bayes["RF + Platt (OOB)"] < custo_bayes["Random Forest"], "com probabilidades calibradas o limiar 1/11 deveria funcionar melhor"
    p_rf = oof["RF + Platt (OOB)"][0]
    conf = {}
    for lim in (0.5, CUSTO_FP / (CUSTO_FP + CUSTO_FN)):
        pr = p_rf >= lim
        conf[lim] = np.array([[np.sum(~pr & (y == 0)), np.sum(pr & (y == 0))], [np.sum(~pr & (y == 1)), np.sum(pr & (y == 1))]])
        _saida(f"Matriz de confusao RF+Platt (semente 0), limiar {lim:.3f}: [[VN FP] [FN VP]] = {conf[lim].tolist()}")

    # --- 5. Importancia por permutacao (RF, conjunto de teste) ---
    imp = importancia_permutacao(rf, X[te], y[te], np.random.default_rng(5))
    ordem = np.argsort(imp)[::-1]
    _saida("\nImportancia por permutacao (queda de AUC no teste, RF de 300 arvores), top 8:")
    for j in ordem[:8]:
        _saida(f"  {nomes_at[j]:<26}{imp[j]:.4f}")

    # ------------------------------------------------------------------------
    # Figuras
    # ------------------------------------------------------------------------
    # 2o atributo: o mais importante que NAO seja quase colinear com o 1o (raio, area e perimetro sao)
    j1 = ordem[0]
    j2 = next(j for j in ordem[1:] if abs(np.corrcoef(X[:, j1], X[:, j])[0, 1]) < 0.75)
    X2 = X[:, [j1, j2]]
    g1 = np.linspace(X2[:, 0].min(), X2[:, 0].max(), 250)
    g2 = np.linspace(X2[:, 1].min(), X2[:, 1].max(), 250)
    G1, G2 = np.meshgrid(g1, g2)
    grade = np.c_[G1.ravel(), G2.ravel()]
    mods2 = [("Logistica L2", Logistica()), ("CART profunda", ArvoreCls()), ("Random Forest", Floresta(n_arv=150, semente=0)), ("Rotation Forest", RotationForest(n_arv=150, semente=0)), ("Boosting nu=0.1", Boosting())]
    fig, axs = plt.subplots(1, 5, figsize=(20, 4.3))
    for ax, (n, m) in zip(axs, mods2):
        m.fit(X2, y)
        Z = m.predict_proba(grade).reshape(G1.shape)
        ax.contourf(G1, G2, Z, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.75)
        ax.contour(G1, G2, Z, levels=[0.5], colors="k", linewidths=1)
        ax.scatter(X2[y == 0, 0], X2[y == 0, 1], s=7, c="#1f4e9c", label="benigno", edgecolors="none")
        ax.scatter(X2[y == 1, 0], X2[y == 1, 1], s=7, c="#b2182b", label="maligno", edgecolors="none")
        ax.set_title(f"{n} (2 atributos)")
        ax.set_xlabel(nomes_at[j1])
    axs[0].set_ylabel(nomes_at[j2])
    axs[0].legend(loc="upper left")
    fig.suptitle("Fronteiras de decisao com o atributo mais importante e o melhor nao colinear (cor = P(maligno), linha = 0.5)")
    fig.tight_layout()
    fig.savefig(FIGS / "01_fronteiras.png", dpi=170)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axs[0]
    ax.plot(range(1, 301), rf.oob_erro, label="erro OOB (treino)", color="#d62")
    ax.plot(range(1, 301), err_te_rf, label="erro de teste", color="#26a")
    ax.axhline(1 - acc_cv_rf, color="k", ls=":", lw=1, label="erro CV 5x5 (100 arv.)")
    ax.set_xscale("log")
    ax.set_xlabel("numero de arvores")
    ax.set_ylabel("taxa de erro")
    ax.set_title("Random Forest: o erro OOB acompanha o de teste")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axs[1]
    cores = {1.0: "#d62", 0.3: "#a4c", 0.1: "#26a"}
    for nu, g in gbs.items():
        ax.plot(range(1, 301), g.loss_treino, color=cores[nu], ls="--", lw=1)
        ax.plot(range(1, 301), ll_te[nu], color=cores[nu], label=f"nu = {nu}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iteracoes de boosting")
    ax.set_ylabel("log-loss (tracejado = treino)")
    ax.set_title("Boosting: shrinkage atrasa o sobreajuste")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "01_curvas_ensembles.png", dpi=170)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axs[0]
    for n, c in zip(nomes_r, ["#555", "#d62", "#2a6", "#a4c", "#6bd", "#26a"]):
        m = [np.mean(v) for v in tab_ruido[n]]
        e = [ic95(v)[1] for v in tab_ruido[n]]
        ax.errorbar(np.array(ruidos) * 100, m, yerr=e, label=n, color=c, marker="o", capsize=3)
    ax.set_xlabel("% de rotulos trocados no treino")
    ax.set_ylabel("acuracia no rotulo verdadeiro")
    ax.set_title("Sintetico: robustez ao ruido de rotulo")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axs[1]
    for (n, cs), c in zip(curvas_custo.items(), ["#555", "#2a6", "#9c6", "#a4c", "#26a", "#e90"]):
        ax.plot(limiares, cs, label=n, color=c)
    ax.axvline(CUSTO_FP / (CUSTO_FP + CUSTO_FN), color="k", ls=":", lw=1, label="limiar de Bayes 1/11")
    ax.axvline(0.5, color="k", ls="--", lw=0.8, label="limiar 0.5")
    ax.set_xlabel("limiar de decisao")
    ax.set_ylabel("custo medio por exame")
    ax.set_title("Custo com FN = 10 x FP (fora-da-dobra)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axs[2]
    top = ordem[:12][::-1]
    ax.barh([nomes_at[j] for j in top], imp[top], color="#2a6")
    ax.set_xlabel("queda de AUC ao embaralhar")
    ax.set_title("Importancia por permutacao (RF)")
    fig.tight_layout()
    fig.savefig(FIGS / "01_ruido_custo_importancia.png", dpi=170)
    plt.close(fig)

    # diagrama de confiabilidade (calibracao) com as probabilidades fora-da-dobra
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axs[0]
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="calibracao perfeita")
    bins = np.linspace(0, 1, 11)
    for n, c in zip(["Logistica L2", "Random Forest", "RF + Platt (OOB)", "Boosting nu=1.0", "Stacking (4 bases)"], ["#555", "#2a6", "#9c6", "#d62", "#e90"]):
        pp, yy = oof[n].ravel(), np.tile(y, len(sementes))
        b = np.minimum((pp * 10).astype(int), 9)
        xs_ = [pp[b == k].mean() for k in range(10) if np.sum(b == k) >= 10]
        ys_ = [yy[b == k].mean() for k in range(10) if np.sum(b == k) >= 10]
        e_ = np.mean([ece(y, oof[n][s_]) for s_ in range(len(sementes))])
        ax.plot(xs_, ys_, "-o", color=c, ms=4, label=f"{n} (ECE {e_:.3f})")
    ax.set_xlabel("probabilidade prevista de maligno (media da faixa)")
    ax.set_ylabel("fracao realmente maligna")
    ax.set_title("Diagrama de confiabilidade (fora-da-dobra, 5 sementes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axs[1]
    nm = list(res)
    acc_m = [100 * (1 - np.mean(res[n]["acc"])) for n in nm]
    ll_m = [np.mean(res[n]["ll"]) for n in nm]
    ax.scatter(acc_m, ll_m, c=["#26a" if n in ("Stacking (4 bases)", "Logistica L2", "Rotation Forest", "Boosting estoc. CV aninh.") else "#999" for n in nm], s=40)
    for n, a_, l_ in zip(nm, acc_m, ll_m):
        ax.annotate(n, (a_, l_), fontsize=7.5, xytext=(4, -10) if n in ("Boosting nu=0.1", "Logistica L2") else (4, 3), textcoords="offset points")
    ax.set_xlabel("erro de classificacao CV 5x5 (%)")
    ax.set_ylabel("log-loss CV 5x5")
    ax.set_yscale("log")
    ax.set_title("Erro x qualidade das probabilidades (11 metodos)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "01_calibracao.png", dpi=170)
    plt.close(fig)

    _saida(f"\nTodos os assert passaram. (tempo total ~ {time.perf_counter() - t_ini:.0f} s)")
    (FIGS / "01_saida.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
