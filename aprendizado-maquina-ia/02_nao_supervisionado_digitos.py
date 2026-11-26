"""
Projeto 2 - Aprendizado nao supervisionado: descobrir os digitos manuscritos sem
olhar os rotulos (e quanto rotulo e preciso depois para nomear os grupos).

PROBLEMA
    Optical Recognition of Handwritten Digits (UCI, acompanha o scikit-learn): 1797
    imagens 8x8 em tons de cinza (64 atributos de 0 a 16) de digitos 0-9 escritos por
    43 pessoas. O algoritmo recebe SO as imagens. Perguntas: (i) os agrupamentos
    encontrados coincidem com os digitos? (ii) quantos grupos os dados "pedem"?
    (iii) a reducao de dimensao ajuda ou atrapalha? (iv) se um humano rotular UMA
    imagem por grupo (a mais central), qual acuracia de classificacao se obtem?
    Os rotulos verdadeiros so sao usados para AVALIAR (ARI, acuracia apos casamento).

MODELAGEM
    x em R^64 (pixels / 16). PCA projeta em R^d (d = 2 ... 64). Um agrupamento e uma
    funcao c: imagens -> {0..k-1}. Metricas: inercia (soma das distancias^2 ao
    centroide), silhueta (coesao x separacao, sem rotulos), BIC (verossimilhanca
    penalizada do GMM), ARI (indice de Rand ajustado ao acaso, com rotulos) e
    acuracia apos casamento otimo grupo->digito (Hungaro).

ALGORITMOS IMPLEMENTADOS A MAO (numpy)
    - PCA por SVD da matriz centrada (variancia explicada, reconstrucao).
    - k-means de Lloyd com inicializacao aleatoria x k-means++ (Arthur e Vassilvitskii).
    - Mistura de gaussianas (GMM) por EM, covariancia completa e diagonal, em
      log-espaco (logsumexp) com regularizacao da covariancia; selecao de k por BIC.
    - DBSCAN (Ester et al.) por busca em largura sobre o grafo de vizinhanca-eps.
    - Agrupamento ESPECTRAL (Ng, Jordan e Weiss, 2002): grafo de k vizinhos, Laplaciano
      normalizado, k menores autovetores (eigh), linhas normalizadas e k-means++;
      eigengap para sugerir k.
    - Propagacao de rotulos no grafo (label spreading, Zhou et al., 2004), forma fechada
      F = (1 - a)(I - a S)^-1 Y: aprendizado SEMI-SUPERVISIONADO com 10 a 50 rotulos.
    - Metricas ARI e silhueta escritas a mao; "rotulagem com k exemplos": cada grupo
      recebe o rotulo da imagem mais proxima do seu centro (aprendizado ativo minimo,
      conceito alem da aula).

VERDADE DE REFERENCIA / AUTOVERIFICACAO
    - PCA: razao de variancia explicada == sklearn.PCA; erro de reconstrucao ==
      soma dos autovalores descartados (propriedade do SVD).
    - k-means: inercia nunca cresce entre iteracoes (ambos os passos a minimizam);
      melhor de 10 inicios ate 1% da inercia do sklearn.KMeans(n_init=10).
    - EM (reg = 1e-6, passo M praticamente exato): log-verossimilhanca monotona nao
      decrescente (com reg = 1e-3 aparecem quedas de ~1e-6, pois somar reg*I deixa de
      ser o maximizador exato); a log-verossimilhanca propria
      == sklearn.GaussianMixture.score com os MESMOS parametros carregados.
    - DBSCAN: mesmos pontos-nucleo e mesmo particionamento (ARI = 1) que sklearn.DBSCAN.
    - ARI e silhueta proprios == sklearn.metrics.
    - Laplaciano: simetrico, espectro em [0, 2], multiplicidade do autovalor 0 == numero
      de componentes conexas (BFS propria == scipy); espectral proprio comparado ao
      sklearn.SpectralClustering; propagacao: forma fechada == iteracao de ponto fixo.
    - Estatistico: (i) nos digitos (20 sementes pareadas, k = 10 e 50) a inercia INICIAL
      media do k-means++ e menor que a da inicializacao aleatoria (a FINAL nao e exigida:
      o Lloyd corrige quase todo inicio ruim neste dado, ver relatorio); (ii) num
      controle com 25 gaussianas bem separadas, cujo otimo global e conhecido, o
      k-means++ acha o otimo em pelo menos 20 p.p. mais sementes que o aleatorio.

Execucao: python 02_nao_supervisionado_digitos.py  (gera figs/02_*.png e figs/02_saida.txt)
"""

import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_digits

PASTA = Path(__file__).resolve().parent
FIGS = PASTA / "figs"
FIGS.mkdir(exist_ok=True)

_LOG = []


def _saida(*args):
    """Imprime a linha e a acumula em _LOG para gravar em figs/02_saida.txt."""
    linha = " ".join(str(a) for a in args)
    print(linha)
    _LOG.append(linha)


# ----------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------
def pca(X):
    """Devolve media, componentes (linhas) e variancias (autovalores da covariancia) via SVD."""
    mu = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt, s ** 2 / (len(X) - 1)


# ----------------------------------------------------------------------------
# k-means
# ----------------------------------------------------------------------------
def d2(X, C):
    """Distancias euclidianas ao quadrado entre linhas de X e de C (n x k)."""
    return np.maximum((X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :], 0)


def init_kmeanspp(X, k, rng):
    """k-means++: cada novo centro e sorteado com probabilidade proporcional a D(x)^2."""
    C = [X[rng.integers(len(X))]]
    dmin = d2(X, np.array(C))[:, 0]
    for _ in range(1, k):
        C.append(X[rng.choice(len(X), p=dmin / dmin.sum())])
        dmin = np.minimum(dmin, d2(X, C[-1][None])[:, 0])
    return np.array(C)


def kmeans(X, k, rng, init="++", max_iter=300):
    """Lloyd: atribui ao centro mais proximo, recalcula medias; devolve rotulos, centros e historico de inercia."""
    C = init_kmeanspp(X, k, rng) if init == "++" else X[rng.choice(len(X), k, replace=False)].copy()
    hist = []
    for _ in range(max_iter):
        D = d2(X, C)
        lab = D.argmin(1)
        hist.append(D[np.arange(len(X)), lab].sum())
        novo = C.copy()
        for j in range(k):
            m = lab == j
            if m.any():
                novo[j] = X[m].mean(0)
            else:  # grupo vazio: recoloca no ponto mais distante do seu centro
                novo[j] = X[D[np.arange(len(X)), lab].argmax()]
        if np.allclose(novo, C):
            break
        C = novo
    D = d2(X, C)
    lab = D.argmin(1)
    hist.append(D[np.arange(len(X)), lab].sum())
    return lab, C, np.array(hist)


def kmeans_melhor(X, k, rng, n_init=10, init="++"):
    melhor = None
    for _ in range(n_init):
        r = kmeans(X, k, rng, init)
        if melhor is None or r[2][-1] < melhor[2][-1]:
            melhor = r
    return melhor


# ----------------------------------------------------------------------------
# GMM por EM
# ----------------------------------------------------------------------------
def logsumexp(A, axis=1):
    m = A.max(axis, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(axis, keepdims=True))).squeeze(axis)


def log_gauss(X, mu, S):
    """log N(x | mu, S) para todas as linhas de X, via Cholesky."""
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (z * z).sum(0) - np.log(np.diag(L)).sum() - 0.5 * X.shape[1] * np.log(2 * np.pi)


def gmm_em(X, k, rng, diag=False, reg=1e-3, max_iter=300, tol=1e-6):
    """EM para mistura de gaussianas, inicializado por k-means++; devolve parametros e historico da log-verossimilhanca media."""
    n, d = X.shape
    lab, mu, _ = kmeans(X, k, rng)
    w = np.bincount(lab, minlength=k) / n + 1e-12
    S = np.array([np.cov(X[lab == j].T) + reg * np.eye(d) if np.sum(lab == j) > 1 else np.eye(d) for j in range(k)])
    if diag:
        S = np.array([np.diag(np.diag(s)) for s in S])
    hist = []
    for _ in range(max_iter):
        # passo E: responsabilidades em log-espaco
        L = np.stack([np.log(w[j]) + log_gauss(X, mu[j], S[j]) for j in range(k)], 1)
        ll = logsumexp(L)
        hist.append(ll.mean())
        R = np.exp(L - ll[:, None])
        # passo M: pesos, medias e covariancias ponderadas
        nk = R.sum(0) + 1e-12
        w = nk / n
        mu = (R.T @ X) / nk[:, None]
        for j in range(k):
            Xc = X - mu[j]
            if diag:
                S[j] = np.diag((R[:, j] @ (Xc * Xc)) / nk[j] + reg)
            else:
                S[j] = (Xc * R[:, j][:, None]).T @ Xc / nk[j] + reg * np.eye(d)
        if len(hist) > 1 and hist[-1] - hist[-2] < tol:
            break
    L = np.stack([np.log(w[j]) + log_gauss(X, mu[j], S[j]) for j in range(k)], 1)
    ll = logsumexp(L)
    hist.append(ll.mean())
    return {"w": w, "mu": mu, "S": S, "lab": L.argmax(1), "hist": np.array(hist), "ll": ll.sum()}


def bic(g, n, diag):
    k, d = g["mu"].shape
    npar = (k - 1) + k * d + k * (d if diag else d * (d + 1) // 2)
    return -2 * g["ll"] + npar * np.log(n)


# ----------------------------------------------------------------------------
# DBSCAN
# ----------------------------------------------------------------------------
def dbscan(X, eps, min_pts):
    """Rotulos (-1 = ruido) e mascara de pontos-nucleo; expansao por busca em largura."""
    viz = [np.flatnonzero(lin) for lin in (d2(X, X) <= eps * eps)]
    nucleo = np.array([len(v) >= min_pts for v in viz])
    lab = np.full(len(X), -1)
    c = 0
    for i in range(len(X)):
        if not nucleo[i] or lab[i] >= 0:
            continue
        lab[i] = c
        fila = deque([i])
        while fila:
            p = fila.popleft()
            for q in viz[p]:
                if lab[q] < 0:
                    lab[q] = c
                    if nucleo[q]:
                        fila.append(q)
        c += 1
    return lab, nucleo


# ----------------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Agrupamento espectral e propagacao de rotulos (grafo de vizinhos)
# ----------------------------------------------------------------------------
def grafo_knn(X, k=10):
    """Afinidade simetrica do grafo de k vizinhos: A = (C + C^T)/2, C_ij = 1 se j esta entre os k vizinhos de i."""
    n = len(X)
    D = d2(X, X)
    np.fill_diagonal(D, np.inf)
    viz = np.argsort(D, 1)[:, :k]
    C = np.zeros((n, n))
    C[np.arange(n)[:, None], viz] = 1.0
    return 0.5 * (C + C.T)


def laplaciano_norm(A):
    """S = D^-1/2 A D^-1/2 e L = I - S (Laplaciano normalizado simetrico)."""
    dm = 1.0 / np.sqrt(A.sum(1))
    S = dm[:, None] * A * dm[None, :]
    return np.eye(len(A)) - S, S


def espectral(X, k, rng, kviz=10, n_init=10):
    """Ng, Jordan e Weiss (2002): k menores autovetores de L, linhas normalizadas, k-means++.
    Devolve rotulos, autovalores de L e a imersao espectral (linhas normalizadas)."""
    L, _ = laplaciano_norm(grafo_knn(X, kviz))
    w, U = np.linalg.eigh(L)
    E = U[:, :k] / np.linalg.norm(U[:, :k], axis=1, keepdims=True)
    return kmeans_melhor(E, k, rng, n_init)[0], w, E


def propagar_rotulos(S, idx, rot, alfa=0.99, iterativo=False, n_cls=10):
    """Label spreading (Zhou et al., 2004): F = (1 - alfa) (I - alfa S)^-1 Y. Cada imagem recebe
    a classe com maior 'calor' difundido a partir das poucas imagens rotuladas pelo grafo."""
    Y = np.zeros((len(S), n_cls))
    Y[idx, rot] = 1.0
    if not iterativo:
        return (1 - alfa) * np.linalg.solve(np.eye(len(S)) - alfa * S, Y)
    F = Y.copy()
    for _ in range(5000):  # F <- alfa S F + (1 - alfa) Y converge pois o raio espectral de alfa S e < 1
        Fn = alfa * S @ F + (1 - alfa) * Y
        if np.abs(Fn - F).max() < 1e-13:
            return Fn
        F = Fn
    return F


def componentes(A):
    """Componentes conexas do grafo por busca em largura (referencia para o espectro de L)."""
    n = len(A)
    comp = -np.ones(n, int)
    c = 0
    for i in range(n):
        if comp[i] >= 0:
            continue
        fila = deque([i])
        comp[i] = c
        while fila:
            u = fila.popleft()
            for v in np.flatnonzero(A[u] > 0):
                if comp[v] < 0:
                    comp[v] = c
                    fila.append(v)
        c += 1
    return c


def centrais(Z, lab, C):
    """Indice da imagem mais proxima do centro de cada grupo (as que um humano rotularia)."""
    return np.array([np.flatnonzero(lab == j)[d2(Z[lab == j], C[j][None])[:, 0].argmin()] for j in range(len(C)) if np.any(lab == j)])


def ari(a, b):
    """Indice de Rand ajustado a partir da tabela de contingencia."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    T = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(T, (a, b), 1)
    c2 = lambda x: x * (x - 1) / 2
    s_ij, s_a, s_b = c2(T).sum(), c2(T.sum(1)).sum(), c2(T.sum(0)).sum()
    esp = s_a * s_b / c2(len(a))
    if 0.5 * (s_a + s_b) == esp:  # caso degenerado (ex.: um unico grupo nas duas particoes): concordancia perfeita
        return 1.0
    return (s_ij - esp) / (0.5 * (s_a + s_b) - esp)


def silhueta(X, lab):
    """Media de (b - a) / max(a, b): a = distancia media ao proprio grupo, b = ao grupo vizinho mais proximo."""
    D = np.sqrt(d2(X, X))
    ks = np.unique(lab)
    M = np.stack([D[:, lab == k].sum(1) for k in ks], 1)
    cont = np.array([np.sum(lab == k) for k in ks])
    idx = np.searchsorted(ks, lab)
    a = M[np.arange(len(X)), idx] / np.maximum(cont[idx] - 1, 1)
    Mb = M / cont
    Mb[np.arange(len(X)), idx] = np.inf
    b = Mb.min(1)
    s = (b - a) / np.maximum(a, b)
    s[cont[idx] == 1] = 0
    return s.mean()


def acc_casada(y, lab):
    """Acuracia apos o melhor casamento 1-para-1 grupo->digito (algoritmo Hungaro)."""
    T = np.zeros((lab.max() + 1, 10))
    np.add.at(T, (lab, y), 1)
    r, c = linear_sum_assignment(-T)
    return T[r, c].sum() / len(y)


def acc_k_rotulos(X, y, lab, C):
    """Rotula cada grupo com o digito da imagem mais proxima do centro (k consultas a um humano)."""
    rot = np.array([y[np.flatnonzero(lab == j)[d2(X[lab == j], C[j][None])[:, 0].argmin()]] if np.any(lab == j) else -1 for j in range(len(C))])
    return np.mean(rot[lab] == y)


def media_dp(v):
    return f"{np.mean(v):.4f} +- {np.std(v, ddof=1):.4f}"


# ----------------------------------------------------------------------------
# Autoverificacao
# ----------------------------------------------------------------------------
def verificar(X, y):
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.mixture import GaussianMixture

    _saida("Verificacao contra o scikit-learn e propriedades matematicas:")
    mu, V, lam = pca(X)
    sk = PCA().fit(X)
    dif = np.abs(lam / lam.sum() - sk.explained_variance_ratio_).max()
    _saida(f"  PCA: max |dif| variancia explicada = {dif:.2e}")
    assert dif < 1e-10
    for dd in (5, 20):
        Z = (X - mu) @ V[:dd].T
        rec = ((X - mu - Z @ V[:dd]) ** 2).sum() / (len(X) - 1)
        assert abs(rec - lam[dd:].sum()) < 1e-8 * lam.sum()
    _saida("  PCA: erro de reconstrucao == soma dos autovalores descartados (d = 5, 20): OK")

    rng = np.random.default_rng(0)
    lab, C, hist = kmeans_melhor(X, 10, rng)
    skm = KMeans(10, n_init=10, random_state=0).fit(X)
    _saida(f"  k-means++ melhor de 10: inercia = {hist[-1]:.2f}; sklearn KMeans(n_init=10) = {skm.inertia_:.2f}")
    assert hist[-1] <= 1.01 * skm.inertia_
    for s in range(20):
        for ini in ("++", "rand"):
            h = kmeans(X, 10, np.random.default_rng(s), ini)[2]
            assert np.all(np.diff(h) <= 1e-9 * h[0]), "inercia deveria ser monotona"
    _saida("  k-means: inercia monotona nao crescente em 40 execucoes: OK")

    a1, a2 = ari(y, lab), adjusted_rand_score(y, lab)
    s1, s2 = silhueta(X, lab), silhouette_score(X, lab)
    _saida(f"  ARI proprio = {a1:.6f} / sklearn = {a2:.6f};  silhueta propria = {s1:.6f} / sklearn = {s2:.6f}")
    assert abs(a1 - a2) < 1e-10 and abs(s1 - s2) < 1e-10

    Z = (X - mu) @ V[:10].T
    for diag in (False, True):
        g = gmm_em(Z, 8, np.random.default_rng(1), diag=diag, reg=1e-6)  # reg ~ 0: passo M exato
        assert np.all(np.diff(g["hist"]) >= -1e-9), "log-verossimilhanca do EM deveria ser monotona"
        gm = GaussianMixture(8, covariance_type="full")
        gm.weights_, gm.means_, gm.covariances_ = g["w"], g["mu"], g["S"]
        gm.precisions_cholesky_ = np.array([np.linalg.inv(np.linalg.cholesky(s)).T for s in g["S"]])
        dif = abs(gm.score(Z) - g["ll"] / len(Z))
        _saida(f"  GMM {'diag' if diag else 'full'} (d=10, k=8): {len(g['hist']) - 1} iter. EM, logL monotona, |logL proprio - sklearn.score| = {dif:.2e}")
        assert dif < 1e-9

    for eps, mp in ((0.7, 8), (0.8, 5)):
        lab_d, nuc = dbscan(Z, eps, mp)
        sk = DBSCAN(eps=eps, min_samples=mp).fit(Z)
        nuc_sk = np.zeros(len(Z), bool)
        nuc_sk[sk.core_sample_indices_] = True
        assert np.array_equal(nuc, nuc_sk)
        cm = nuc  # nucleos tem atribuicao unica; pontos de borda podem depender da ordem de visita
        assert ari(lab_d[cm], sk.labels_[cm]) == 1.0 and np.array_equal(lab_d < 0, sk.labels_ < 0)
        _saida(f"  DBSCAN (PCA-10, eps={eps}, min={mp}): {lab_d.max() + 1} grupos, {np.sum(lab_d < 0)} ruido; nucleos e particao == sklearn")

    from scipy.sparse.csgraph import connected_components
    from sklearn.cluster import SpectralClustering
    for kv in (2, 3, 10):
        A = grafo_knn(X, kv)
        L, S = laplaciano_norm(A)
        w, U = np.linalg.eigh(L)
        ncomp = componentes(A)
        assert ncomp == connected_components(A, directed=False)[0]
        zeros = int(np.sum(w < 1e-9))
        assert np.allclose(L, L.T) and w.min() > -1e-10 and w.max() < 2 + 1e-10, "L normalizado: simetrico, espectro em [0, 2]"
        assert zeros == ncomp, "multiplicidade do autovalor 0 == numero de componentes conexas"
        assert np.abs(L @ U[:, :5] - U[:, :5] * w[:5]).max() < 1e-9
        _saida(f"  Grafo {kv}-NN: L simetrico, espectro em [{w.min():.1e}, {w.max():.3f}], autovalores ~0: {zeros} == componentes conexas: {ncomp}")
    lab_e, _, _ = espectral(X, 10, np.random.default_rng(0))
    sk = SpectralClustering(10, affinity="nearest_neighbors", n_neighbors=10, assign_labels="kmeans", random_state=0).fit(X)
    _saida(f"  Espectral proprio (10-NN): ARI = {ari(y, lab_e):.4f}; sklearn SpectralClustering = {adjusted_rand_score(y, sk.labels_):.4f}; concordancia ARI entre os dois = {ari(lab_e, sk.labels_):.4f}")
    assert ari(lab_e, sk.labels_) > 0.8
    idx = np.random.default_rng(0).choice(len(X), 30, replace=False)
    F1 = propagar_rotulos(S, idx, y[idx])
    F2 = propagar_rotulos(S, idx, y[idx], iterativo=True)
    _saida(f"  Propagacao de rotulos: forma fechada x iteracao de ponto fixo, max |dif| = {np.abs(F1 - F2).max():.1e}")
    assert np.abs(F1 - F2).max() < 1e-9


# ----------------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------------
def main():
    t_ini = time.perf_counter()
    dig = load_digits()
    X, y, imgs = dig.data / 16.0, dig.target, dig.images
    _saida(f"Digits (UCI): {X.shape[0]} imagens 8x8, {X.shape[1]} atributos; rotulos usados SO na avaliacao\n")
    verificar(X, y)
    mu, V, lam = pca(X)
    proj = lambda d: (X - mu) @ V[:d].T
    vexp = np.cumsum(lam) / lam.sum()
    _saida("\nVariancia acumulada do PCA: " + ", ".join(f"d={d}: {vexp[d - 1]:.3f}" for d in (2, 5, 10, 20, 30, 40)))

    # --- 1. Ablacao da inicializacao (20 sementes pareadas, espaco original) ---
    _saida("\nAblacao: inicializacao do k-means (64 dims, 20 sementes pareadas, 1 inicio cada):")
    _saida(f"{'k':<4}{'init':<11}{'inercia inicial':>22}{'inercia final':>24}{'iteracoes':>18}{'ARI':>20}")
    abl, abl_k = {}, {}
    for k in (10, 50):
        for ini in ("rand", "++"):
            r = [kmeans(X, k, np.random.default_rng(s), ini) for s in range(20)]
            a = {"i0": [h[0] for _, _, h in r], "in": [h[-1] for _, _, h in r], "it": [len(h) - 1 for _, _, h in r], "ari": [ari(y, l) for l, _, _ in r], "h": [h for _, _, h in r]}
            abl_k[(k, ini)] = a
            if k == 10:
                abl[ini] = a
            _saida(f"{k:<4}{'aleatoria' if ini == 'rand' else 'k-means++':<11}{media_dp(a['i0']):>22}{media_dp(a['in']):>24}{media_dp(a['it']):>18}{media_dp(a['ari']):>20}")
        d0 = np.array(abl_k[(k, "++")]["i0"]) - np.array(abl_k[(k, "rand")]["i0"])
        d_in = np.array(abl_k[(k, "++")]["in"]) - np.array(abl_k[(k, "rand")]["in"])
        _saida(f"    pareado k={k}: inercia inicial ++ menor em {np.sum(d0 < 0)}/20 (dif. media {d0.mean():.1f}); inercia final ++ menor em {np.sum(d_in < 0)}/20 (dif. media {d_in.mean():.2f})")
        assert d0.mean() < 0, "k-means++ deveria comecar, em media, mais perto do otimo"

    # controle: grupos BEM separados (grade 5x5 de gaussianas), onde o otimo global e conhecido
    rng_b = np.random.default_rng(9)
    cent = np.array([(i, j) for i in range(5) for j in range(5)], float)
    B = np.repeat(cent, 40, 0) + rng_b.normal(0, 0.08, (1000, 2))
    yb = np.repeat(np.arange(25), 40)
    otimo = sum(((B[yb == j] - B[yb == j].mean(0)) ** 2).sum() for j in range(25))  # particao verdadeira
    _saida(f"Controle: 25 gaussianas bem separadas (grade 5x5, 1000 pontos), inercia da particao verdadeira = {otimo:.3f}")
    taxa = {}
    for ini in ("rand", "++"):
        fin = np.array([kmeans(B, 25, np.random.default_rng(s), ini)[2][-1] for s in range(50)])
        taxa[ini] = np.mean(fin <= otimo * 1.001)
        _saida(f"    {'aleatoria' if ini == 'rand' else 'k-means++':<11}: acha o otimo global em {100 * taxa[ini]:.0f}% de 50 sementes; inercia final media = {fin.mean():.3f}")
    assert taxa["++"] > taxa["rand"] + 0.2, "em grupos separados o k-means++ deveria achar o otimo bem mais vezes"

    # --- 2. Efeito da dimensao do PCA (ARI) ---
    dims = [2, 5, 10, 20, 30, 64]
    _saida("\nARI x dimensao do PCA (k=10; k-means: 10 sementes x melhor de 5; GMM full: 5 sementes):")
    _saida(f"{'d':<6}{'k-means++':>22}{'GMM full':>22}{'GMM diag':>22}")
    ari_d = {"km": [], "gf": [], "gd": []}
    for dd in dims:
        Z = proj(dd) if dd < 64 else X
        km = [ari(y, kmeans_melhor(Z, 10, np.random.default_rng(s), 5)[0]) for s in range(10)]
        gf = [ari(y, gmm_em(Z, 10, np.random.default_rng(s), reg=1e-2)["lab"]) for s in range(5)] if dd <= 30 else [np.nan]
        gd = [ari(y, gmm_em(Z, 10, np.random.default_rng(s), diag=True, reg=1e-2)["lab"]) for s in range(5)]
        for key, v in zip(("km", "gf", "gd"), (km, gf, gd)):
            ari_d[key].append(v)
        fmt = lambda v: f"{np.mean(v):.4f} +- {np.std(v, ddof=1):.4f}" if len(v) > 1 else "  (instavel em d=64)"
        _saida(f"{dd:<6}{fmt(km):>22}{fmt(gf):>22}{fmt(gd):>22}")
    assert np.mean(ari_d["km"][0]) < np.mean(ari_d["km"][3]), "2 dimensoes deveriam perder informacao"

    # --- 3. Selecao de k sem rotulos ---
    ks = list(range(4, 21))
    Z20 = proj(20)
    _saida("\nSelecao de k sem rotulos (PCA-20): inercia, silhueta, BIC do GMM full e diag (menor = melhor):")
    _saida(f"{'k':<5}{'inercia':>10}{'silhueta':>10}{'BIC full':>12}{'BIC diag':>12}{'ARI k-means':>13}")
    sel = {"in": [], "sil": [], "bf": [], "bd": [], "ari": []}
    for k in ks:
        lab, C, h = kmeans_melhor(Z20, k, np.random.default_rng(k), 5)
        bf = min(bic(gmm_em(Z20, k, np.random.default_rng(s), reg=1e-2), len(X), False) for s in range(3))
        bd = min(bic(gmm_em(Z20, k, np.random.default_rng(s), diag=True, reg=1e-2), len(X), True) for s in range(3))
        for key, v in zip(sel, (h[-1], silhueta(Z20, lab), bf, bd, ari(y, lab))):
            sel[key].append(v)
        _saida(f"{k:<5}{h[-1]:>10.1f}{sel['sil'][-1]:>10.4f}{bf:>12.1f}{bd:>12.1f}{sel['ari'][-1]:>13.4f}")
    _saida(f"k escolhido: silhueta -> {ks[int(np.argmax(sel['sil']))]}, BIC full -> {ks[int(np.argmin(sel['bf']))]}, BIC diag -> {ks[int(np.argmin(sel['bd']))]}, maior ARI -> {ks[int(np.argmax(sel['ari']))]}")

    # --- 4. Do agrupamento a classificacao com k rotulos ---
    _saida("\nAcuracia de classificacao rotulando SO a imagem mais central de cada grupo (k-means++ melhor de 5, 10 sementes):")
    _saida(f"{'k (= rotulos pedidos)':<24}{'espaco':<10}{'acuracia':>20}{'acc. Hungaro (k=10)':>22}")
    acc_k = {}
    for espaco, Z in (("64 px", X), ("PCA-20", Z20)):
        for k in (10, 20, 30, 50):
            acc, hung = [], []
            for s in range(10):
                lab, C, _ = kmeans_melhor(Z, k, np.random.default_rng(100 + s), 5)
                acc.append(acc_k_rotulos(Z, y, lab, C))
                if k == 10:
                    hung.append(acc_casada(y, lab))
            acc_k[(espaco, k)] = acc
            _saida(f"{k:<24}{espaco:<10}{media_dp(acc):>20}{media_dp(hung) if hung else '':>22}")
    assert np.mean(acc_k[("64 px", 50)]) > np.mean(acc_k[("64 px", 10)]), "mais grupos (mais rotulos) deveriam ajudar"

    # --- 5. DBSCAN: densidade descobre grupos sem fixar k ---
    _saida("\nDBSCAN em PCA-10 (min_pts = 8), varrendo eps:")
    _saida(f"{'eps':<6}{'grupos':>8}{'% ruido':>10}{'ARI (sem ruido)':>17}")
    Z10 = proj(10)
    melhor_db = None
    for eps in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        lab_d, _ = dbscan(Z10, eps, 8)
        m = lab_d >= 0
        a = ari(y[m], lab_d[m]) if lab_d.max() > 0 else 0.0
        _saida(f"{eps:<6}{lab_d.max() + 1:>8}{100 * (1 - m.mean()):>10.1f}{a:>17.4f}")
        if melhor_db is None or a > melhor_db[1]:
            melhor_db = (eps, a, lab_d)

    # --- 6. Agrupamento espectral (grafo de vizinhos): forma, nao distancia ao centro ---
    _saida("\nAgrupamento espectral (Ng-Jordan-Weiss, grafo k-NN, 64 px, k = 10 grupos, 5 sementes do k-means final):")
    _saida(f"{'vizinhos':<10}{'ARI':>20}{'acc Hungaro':>20}{'eigengap -> k':>15}")
    esp = {}
    for kv in (5, 10, 15, 20):
        r = [espectral(X, 10, np.random.default_rng(s), kv) for s in range(5)]
        w = r[0][1]
        esp[kv] = {"ari": [ari(y, l) for l, _, _ in r], "acc": [acc_casada(y, l) for l, _, _ in r], "w": w, "gap": int(np.argmax(np.diff(w[1:25]))) + 2}
        _saida(f"{kv:<10}{media_dp(esp[kv]['ari']):>20}{media_dp(esp[kv]['acc']):>20}{esp[kv]['gap']:>15}")
    lab_esp, _, _ = espectral(X, 10, np.random.default_rng(0), 10)
    gm_best = [max((gmm_em(Z20, 10, np.random.default_rng(100 * s + r), reg=1e-2) for r in range(10)), key=lambda g: g["ll"]) for s in range(5)]
    ari_gm_best = [ari(y, g["lab"]) for g in gm_best]
    _saida(f"GMM full em PCA-20 escolhendo o melhor de 10 inicios pela log-verossimilhanca (sem rotulos), 5 repeticoes: ARI = {media_dp(ari_gm_best)}")
    resumo = {
        "k-means++ (64 px)": ari_d["km"][-1],
        "k-means++ (PCA-20)": ari_d["km"][3],
        "GMM full (PCA-20)": ari_d["gf"][3],
        "GMM full melhor logL": ari_gm_best,
        f"DBSCAN eps={melhor_db[0]} (sem ruido)": [melhor_db[1]],
        "Espectral 10-NN": esp[10]["ari"],
    }
    _saida("Resumo (ARI com os digitos, k = 10):  " + "; ".join(f"{n}: {np.mean(v):.3f}" for n, v in resumo.items()))
    dd = np.array(esp[10]["ari"]) - np.array(ari_d["km"][-1][:5])
    _saida(f"Pareado (mesmas 5 sementes): ARI(espectral) - ARI(k-means++ 64 px) = {dd.mean():.3f}, espectral melhor em {np.sum(dd > 0)}/5")
    assert np.all(dd > 0), "o espectral deveria superar o k-means em todas as sementes"
    assert np.mean(esp[10]["ari"]) > np.mean(ari_gm_best), "e tambem o melhor GMM"

    # --- 7. Semi-supervisionado: poucos rotulos + grafo ---
    _saida("\nClassificacao com m imagens rotuladas por um humano (acuracia nas 1797 imagens, 10 sementes):")
    _saida("  A: k-means++ (64 px, k = m) e cada grupo herda o rotulo da imagem central (secao 4)")
    _saida("  B: 1-vizinho mais proximo usando m imagens sorteadas ao acaso")
    _saida("  C: propagacao de rotulos no grafo 10-NN a partir de m imagens sorteadas")
    _saida("  D: propagacao de rotulos a partir das MESMAS m imagens centrais de A (ablacao pareada: so muda como o rotulo se espalha)")
    _saida("  E: propagacao de rotulos a partir das m imagens centrais dos grupos ESPECTRAIS (k-means em m autovetores)")
    _, S10 = laplaciano_norm(grafo_knn(X, 10))
    L10 = np.eye(len(X)) - S10
    w10, U10 = np.linalg.eigh(L10)
    ms_ = (10, 20, 30, 50)
    _saida(f"{'m':<5}{'A: centrais k-means':>22}{'B: 1-NN aleatorio':>22}{'C: propag. aleat.':>22}{'D: propag. centr. A':>22}{'E: propag. centr. esp.':>24}")
    semi = {c: {m_: [] for m_ in ms_} for c in "ABCDE"}
    for m_ in ms_:
        Em = U10[:, :m_] / np.linalg.norm(U10[:, :m_], axis=1, keepdims=True)
        for s in range(10):
            rng_s = np.random.default_rng(500 + s)
            lab_a, C_a, _ = kmeans_melhor(X, m_, np.random.default_rng(100 + s), 5)
            semi["A"][m_].append(acc_k_rotulos(X, y, lab_a, C_a))
            ia = centrais(X, lab_a, C_a)
            semi["D"][m_].append(np.mean(propagar_rotulos(S10, ia, y[ia]).argmax(1) == y))
            idx = rng_s.choice(len(X), m_, replace=False)
            semi["B"][m_].append(np.mean(y[idx][d2(X, X[idx]).argmin(1)] == y))
            semi["C"][m_].append(np.mean(propagar_rotulos(S10, idx, y[idx]).argmax(1) == y))
            lab_m, C_m, _ = kmeans_melhor(Em, m_, rng_s, 5)
            ic = centrais(Em, lab_m, C_m)
            semi["E"][m_].append(np.mean(propagar_rotulos(S10, ic, y[ic]).argmax(1) == y))
        _saida(f"{m_:<5}" + "".join(f"{media_dp(semi[c][m_]):>22}" for c in "ABCD") + f"{media_dp(semi['E'][m_]):>24}")
    for m_ in ms_:
        dd = np.array(semi["D"][m_]) - np.array(semi["A"][m_])
        _saida(f"  m={m_}: pareado D - A = {100 * dd.mean():+.2f} p.p. (D melhor em {np.sum(dd > 0)}/10)")
    for m_ in ms_:
        assert np.mean(semi["D"][m_]) > np.mean(semi["A"][m_]) and np.sum(np.array(semi["D"][m_]) > np.array(semi["A"][m_])) >= 8, "propagar pelo grafo deveria ajudar"
        assert np.mean(semi["C"][m_]) > np.mean(semi["B"][m_]), "com rotulos ao acaso, o grafo deveria superar o 1-NN"
        assert np.mean(semi["D"][m_]) > np.mean(semi["C"][m_]), "escolher as imagens centrais deveria superar o sorteio"
    assert np.mean(semi["E"][10]) > np.mean(semi["A"][10]) + 0.05

    # ------------------------------------------------------------------------
    # Figuras
    # ------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axs[0]
    for kv, c in zip((5, 10, 15, 20), ["#e90", "#26a", "#2a6", "#a4c"]):
        ax.plot(range(1, 26), esp[kv]["w"][:25], "o-", ms=3.5, color=c, label=f"{kv}-NN (eigengap -> k = {esp[kv]['gap']})")
    ax.axvline(10.5, color="k", ls=":", lw=1)
    ax.set_xlabel("indice do autovalor")
    ax.set_ylabel("autovalor do Laplaciano normalizado")
    ax.set_title("Espectro do Laplaciano (o maior salto sugere k)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axs[1]
    T = np.zeros((10, 10), int)
    np.add.at(T, (lab_esp, y), 1)
    lin, col = linear_sum_assignment(-T)
    T = T[lin[np.argsort(col)]]  # grupo casado com o digito d fica na linha d
    ax.imshow(T / T.sum(0, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if T[i, j]:
                ax.text(j, i, str(T[i, j]), ha="center", va="center", fontsize=7, color="w" if T[i, j] > 0.5 * T[:, j].sum() else "k")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("digito verdadeiro")
    ax.set_ylabel("grupo espectral (casado ao digito)")
    ax.set_title(f"Espectral 10-NN x digitos: ARI = {ari(y, lab_esp):.3f}, acc casada = {acc_casada(y, lab_esp):.3f}")
    ax = axs[2]
    nomes_semi = {"A": "A: centrais do k-means", "B": "B: 1-NN, rotulos ao acaso", "C": "C: propagacao, ao acaso", "D": "D: propagacao, centrais de A", "E": "E: propagacao, centrais espectrais"}
    for c, cor in zip("ABCDE", ["#555", "#d62", "#e90", "#26a", "#2a6"]):
        ax.errorbar(ms_, [100 * np.mean(semi[c][m_]) for m_ in ms_], yerr=[100 * np.std(semi[c][m_], ddof=1) for m_ in ms_], marker="o", capsize=3, color=cor, label=nomes_semi[c])
    ax.set_xlabel("imagens rotuladas por um humano (m)")
    ax.set_ylabel("acuracia nas 1797 imagens (%)")
    ax.set_title("Poucos rotulos: escolher bem + propagar pelo grafo")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "02_espectral.png", dpi=170)
    plt.close(fig)

    rng = np.random.default_rng(0)
    lab10, C10, _ = kmeans_melhor(X, 10, rng)
    Z2 = proj(2)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.9))
    cm = plt.get_cmap("tab10")
    axs[0].scatter(Z2[:, 0], Z2[:, 1], c=y, cmap="tab10", s=6)
    for dgt in range(10):
        c = Z2[y == dgt].mean(0)
        axs[0].text(c[0], c[1], str(dgt), fontsize=13, weight="bold", ha="center", va="center", bbox=dict(fc="w", alpha=0.7, lw=0))
    axs[0].set_title("Rotulos verdadeiros (so para avaliar), PCA-2")
    axs[1].scatter(Z2[:, 0], Z2[:, 1], c=lab10, cmap="tab10", s=6)
    axs[1].set_title(f"k-means++ em 64 px (k=10), ARI = {ari(y, lab10):.3f}")
    eps_b, a_b, lab_b = melhor_db
    m = lab_b >= 0
    axs[2].scatter(Z2[~m, 0], Z2[~m, 1], c="#bbb", s=4, label="ruido")
    axs[2].scatter(Z2[m, 0], Z2[m, 1], c=lab_b[m] % 10, cmap="tab10", s=6)
    axs[2].set_title(f"DBSCAN em PCA-10 (eps={eps_b}), {lab_b.max() + 1} grupos, ARI = {a_b:.3f}")
    axs[2].legend(loc="upper right")
    for ax in axs:
        ax.set_xlabel("componente principal 1")
        ax.set_ylabel("componente principal 2")
    fig.tight_layout()
    fig.savefig(FIGS / "02_projecoes.png", dpi=170)
    plt.close(fig)

    # centroides do k-means como imagens + composicao de cada grupo
    T = np.zeros((10, 10), int)
    np.add.at(T, (lab10, y), 1)
    ordem = np.argsort(T.argmax(1))
    fig, axs = plt.subplots(2, 10, figsize=(15, 3.6), gridspec_kw={"height_ratios": [1, 0.75]})
    for i, j in enumerate(ordem):
        axs[0, i].imshow(C10[j].reshape(8, 8), cmap="gray_r")
        axs[0, i].set_title(f"grupo {j}: '{T[j].argmax()}'\n{T[j].max()}/{T[j].sum()} ({100 * T[j].max() / T[j].sum():.0f}%)", fontsize=9)
        axs[0, i].axis("off")
        axs[1, i].bar(range(10), T[j] / T[j].sum(), color=[cm(t) for t in range(10)])
        axs[1, i].set_xticks(range(10), [str(t) for t in range(10)], fontsize=6)
        axs[1, i].set_ylim(0, 1)
        axs[1, i].set_yticks([] if i else [0, 0.5, 1])
    axs[1, 0].set_ylabel("fracao")
    fig.suptitle("Centroides do k-means++ (k=10, sem rotulos) e composicao de cada grupo pelos digitos verdadeiros")
    fig.tight_layout()
    fig.savefig(FIGS / "02_centroides.png", dpi=170)
    plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.3))
    ax = axs[0]
    for ini, c in (("rand", "#d62"), ("++", "#26a")):
        for i, h in enumerate(abl[ini]["h"]):
            ax.plot(h, color=c, alpha=0.5, lw=0.8, label=("aleatoria" if ini == "rand" else "k-means++") if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("iteracao de Lloyd")
    ax.set_ylabel("inercia")
    ax.set_title("Inicializacao (20 sementes)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax = axs[1]
    for key, n, c in (("km", "k-means++", "#26a"), ("gf", "GMM full", "#2a6"), ("gd", "GMM diag", "#d62")):
        mm = [np.mean(v) for v in ari_d[key]]
        ee = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in ari_d[key]]
        ax.errorbar(dims, mm, yerr=ee, label=n, color=c, marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("dimensao apos PCA")
    ax.set_ylabel("ARI com os digitos")
    ax.set_title("Quanto reduzir a dimensao?")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axs[2]
    ax.plot(ks, (np.array(sel["bf"]) - min(sel["bf"])) / 1000, "o-", color="#2a6", label="BIC full - min (x1000)")
    ax.plot(ks, (np.array(sel["bd"]) - min(sel["bd"])) / 1000, "s-", color="#d62", label="BIC diag - min (x1000)")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC relativo (menor = melhor)")
    ax2 = ax.twinx()
    ax2.plot(ks, sel["sil"], "^-", color="#26a", label="silhueta (k-means)")
    ax2.plot(ks, sel["ari"], "v--", color="#555", label="ARI (usa rotulos)")
    ax2.set_ylabel("silhueta / ARI")
    ax.axvline(10, color="k", ls=":", lw=1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right", framealpha=0.9)
    ax.set_title("Quantos grupos? (PCA-20)")
    ax.grid(alpha=0.3)
    ax = axs[3]
    kk = [10, 20, 30, 50]
    for espaco, c in (("64 px", "#26a"), ("PCA-20", "#2a6")):
        ax.errorbar(kk, [np.mean(acc_k[(espaco, k)]) for k in kk], yerr=[np.std(acc_k[(espaco, k)], ddof=1) for k in kk], marker="o", capsize=3, color=c, label=espaco)
    ax.set_xlabel("k grupos = imagens rotuladas por um humano")
    ax.set_ylabel("acuracia em todas as 1797 imagens")
    ax.set_title("De grupos a classificador com k rotulos")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "02_selecao_ablacao.png", dpi=170)
    plt.close(fig)

    _saida(f"\nTodos os assert passaram. (tempo total ~ {time.perf_counter() - t_ini:.0f} s)")
    (FIGS / "02_saida.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
