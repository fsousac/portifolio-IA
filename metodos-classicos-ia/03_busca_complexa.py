"""
Projeto 3 - Busca complexa (busca local e otimizacao): layout de teclado para portugues.

PROBLEMA
    Dadas 30 teclas (26 letras + 'c cedilha' + ',' '.' ';') e uma grade fisica 3x10 de
    posicoes, encontrar a atribuicao tecla -> posicao (uma permutacao) que minimiza o
    "esforco de digitacao" medio por bigrama de um corpus em portugues (Dom Casmurro,
    Machado de Assis, dominio publico, Project Gutenberg, salvo em corpus_pt.txt).

    Espaco de busca: 30! ~ 2,65e32 permutacoes -> inviavel enumerar; sem estrutura
    que permita garantir o otimo, logo usamos busca local (meta-heuristicas).

FUNCAO DE CUSTO (media por bigrama; F = frequencias normalizadas de bigramas a->b)
    custo(layout) = soma_{a,b} F[a,b] * C[pos(a), pos(b)]
    C[p,q] = e(p) + e(q)                                  (esforco individual de cada tecla)
           + [p!=q e mesmo dedo]      * (LAM_MESMO_DEDO + MU_DIST * dist(p,q))
           + [mesma mao, dedos dif.]  * LAM_MESMA_MAO     (alternar maos e "gratis")
           + [mesma mao, dedos dif. e salto de 2 linhas] * LAM_SALTO_LINHA
    e(p) = PESO_DEDO[f] * dist(p, home(f)) + BASE_DEDO[f]
    Dedo por coluna: [mindinho, anelar, medio, indicador, indicador, indicador,
                      indicador, medio, anelar, mindinho] (colunas 0..9).
    Casa (home) de cada dedo: linha do meio (1); indicadores em col 3 e col 6.
    Distancia euclidiana com passo de tecla = 1 (sem escalonamento de linhas).
    Bigramas envolvendo espaco, digitos ou outros simbolos sao descartados (o
    espaco e batido pelo polegar e "quebra" a sequencia).

ALGORITMOS (implementados a mao)
    Hill climbing (steepest e first-improvement), HC com reinicios aleatorios,
    Simulated Annealing (resfriamento geometrico, linear e logaritmico) e
    Local Beam Search. Vizinhanca: troca de duas teclas (435 vizinhos).

COMPARACAO JUSTA
    Todos usam o mesmo orcamento maximo de avaliacoes de custo (ORCAMENTO), o mesmo
    layout inicial por semente e N_SEEDS sementes. Baselines: QWERTY (ABNT2, que
    ocupa exatamente as 30 posicoes) e a media de layouts aleatorios.

Uso: python 03_busca_complexa.py   (gera figs/03_*.png e figs/03_saida.txt)
"""
import math
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # backend sem janela (so salva arquivos)
import matplotlib.pyplot as plt  # noqa: E402  (precisa vir depois do use("Agg"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

# ----------------------------------------------------------------------------
# Configuracao geral
# ----------------------------------------------------------------------------
PASTA = Path(__file__).resolve().parent
FIGS = PASTA / "figs"
FIGS.mkdir(exist_ok=True)
CORPUS = PASTA / "corpus_pt.txt"

# numero de sementes (execucoes independentes) por algoritmo
N_SEEDS = 12
ORCAMENTO = 30_000      # maximo de avaliacoes de custo por execucao
N_OTIMOS_LOCAIS = 300   # partidas de HC para estudar otimos locais
N_ALEATORIOS = 2000     # layouts aleatorios para o baseline

# Pesos da funcao de custo (documentados no docstring)
LAM_MESMO_DEDO = 2.0
MU_DIST = 1.0
LAM_MESMA_MAO = 0.2
LAM_SALTO_LINHA = 0.4
PESO_DEDO = [1.5, 1.2, 1.0, 0.8, 0.8, 0.8, 0.8, 1.0, 1.2, 1.5]  # por coluna
BASE_DEDO = [0.3, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.3]

# coluna -> dedo (0..7); 0-3 esq., 4-7 dir.
DEDO_COL = [0, 1, 2, 3, 3, 4, 4, 5, 6, 7]
# coluna da casa de cada dedo (linha 1)
HOME_COL = [0, 1, 2, 3, 6, 7, 8, 9]

SIMBOLOS = list("abcdefghijklmnopqrstuvwxyz") + ["ç", ",", ".", ";"]
IDX = {s: i for i, s in enumerate(SIMBOLOS)}
N = len(SIMBOLOS)  # 30
# QWERTY (ABNT2) na grade 3x10; o valor e o simbolo, a posicao e linha*10+coluna
QWERTY = list("qwertyuiop" "asdfghjklç" "zxcvbnm,.;")

SAIDA = []  # linhas impressas (para salvar em figs/03_saida.txt)


def say(txt=""):
    """Imprime e guarda a linha para o arquivo de saida."""
    print(txt)
    SAIDA.append(str(txt))


# ----------------------------------------------------------------------------
# Corpus e matriz de bigramas
# ----------------------------------------------------------------------------
def normalizar(texto):
    """Minusculas, remove acentos (mantem 'ç') e troca o resto por '\\n' (quebra)."""
    texto = texto.lower().replace("ç", "\x01")          # protege o cedilha
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = texto.replace("\x01", "ç")
    return "".join(c if c in IDX else "\n" for c in texto)


def matriz_bigramas(texto):
    """F[a,b] = frequencia relativa do bigrama a->b (so simbolos adjacentes validos)."""
    ids = np.array([IDX.get(c, -1) for c in texto])
    a, b = ids[:-1], ids[1:]
    ok = (a >= 0) & (b >= 0)
    F = np.zeros((N, N))
    np.add.at(F, (a[ok], b[ok]), 1.0)
    return F / F.sum(), ids[ids >= 0]


# ----------------------------------------------------------------------------
# Geometria do teclado e matriz de custo por par de posicoes
# ----------------------------------------------------------------------------
def pos_rc(p):
    """Posicao linear 0..29 -> (linha, coluna)."""
    return divmod(p, 10)


def dedo_de(p):
    """Dedo (0..7) que digita a posicao p, dado pela coluna."""
    return DEDO_COL[pos_rc(p)[1]]


def mao_de(p):
    """Mao da posicao p: 0 esquerda, 1 direita."""
    return 0 if dedo_de(p) < 4 else 1  # 0 esquerda, 1 direita


def dist(p, q):
    """Distancia euclidiana entre as posicoes p e q (passo de tecla = 1)."""
    (r1, c1), (r2, c2) = pos_rc(p), pos_rc(q)
    return math.hypot(r1 - r2, c1 - c2)


def esforco_tecla(p):
    """e(p): distancia da tecla a casa do seu dedo, ponderada pela forca do dedo."""
    r, c = pos_rc(p)
    d = math.hypot(r - 1, c - HOME_COL[dedo_de(p)])
    return PESO_DEDO[c] * d + BASE_DEDO[c]


def matriz_custo_posicoes():
    """C[p,q]: custo de digitar uma tecla na posicao p seguida de outra em q."""
    C = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            c = esforco_tecla(p) + esforco_tecla(q)
            if p != q:
                fp, fq = dedo_de(p), dedo_de(q)
                if fp == fq:
                    c += LAM_MESMO_DEDO + MU_DIST * dist(p, q)
                elif mao_de(p) == mao_de(q):
                    c += LAM_MESMA_MAO
                    if abs(pos_rc(p)[0] - pos_rc(q)[0]) == 2:
                        c += LAM_SALTO_LINHA
            C[p, q] = c
    return C


# ----------------------------------------------------------------------------
# Avaliador com contagem de avaliacoes (orcamento) e curva de convergencia
# ----------------------------------------------------------------------------
class OrcamentoEsgotado(Exception):
    """Sinaliza que o numero maximo de avaliacoes foi atingido."""


class Avaliador:
    """Calcula o custo de um layout (perm[simbolo] = posicao 0..29 da tecla).

    Conta cada avaliacao, guarda o melhor ja visto e a curva 'melhor ate agora'.
    """

    def __init__(self, F, C, orcamento):
        self.F, self.C, self.orcamento = F, C, orcamento
        self.n = 0
        self.melhor = math.inf
        self.melhor_perm = None
        self.curva = np.full(orcamento, np.nan)

    def custo_puro(self, perm):
        """Custo sem contabilizar (usado para baselines e verificacoes)."""
        return float((self.F * self.C[perm[:, None], perm[None, :]]).sum())

    def __call__(self, perm):
        if self.n >= self.orcamento:
            raise OrcamentoEsgotado
        c = self.custo_puro(perm)
        if c < self.melhor:
            self.melhor, self.melhor_perm = c, perm.copy()
        self.curva[self.n] = self.melhor
        self.n += 1
        return c

    def curva_completa(self):
        """Curva com a cauda preenchida (algoritmos que param cedo ficam planos)."""
        c = self.curva.copy()
        if self.n < self.orcamento:
            c[self.n:] = self.melhor
        return c


PARES = [(i, j) for i in range(N)
         for j in range(i + 1, N)]  # 435 trocas possiveis


def trocar(perm, i, j):
    """Troca in-place os valores das entradas i e j de perm (vizinho = 1 troca)."""
    perm[i], perm[j] = perm[j], perm[i]


# ----------------------------------------------------------------------------
# Algoritmos de busca local
# ----------------------------------------------------------------------------
def hc_steepest(ev, rng, inicial):
    """Hill climbing de maior subida: avalia toda a vizinhanca e faz a melhor troca."""
    perm = inicial.copy()
    atual = ev(perm)
    while True:
        melhor_c, melhor_par = atual, None
        for i, j in PARES:
            trocar(perm, i, j)
            c = ev(perm)
            trocar(perm, i, j)
            if c < melhor_c - 1e-12:
                melhor_c, melhor_par = c, (i, j)
        if melhor_par is None:          # otimo local (nenhum vizinho melhora)
            return perm, atual
        trocar(perm, *melhor_par)
        atual = melhor_c


def hc_first(ev, rng, inicial):
    """Hill climbing 'primeira melhoria': aceita o primeiro vizinho melhor (ordem aleatoria)."""
    perm = inicial.copy()
    atual = ev(perm)
    melhorou = True
    while melhorou:
        melhorou = False
        ordem = rng.permutation(len(PARES))
        for k in ordem:
            i, j = PARES[k]
            trocar(perm, i, j)
            c = ev(perm)
            if c < atual - 1e-12:
                atual, melhorou = c, True   # mantem a troca e reembaralha
                break
            trocar(perm, i, j)
    return perm, atual                  # varredura completa sem melhora = otimo local


def hc_reinicios(ev, rng, inicial):
    """HC first-improvement com reinicios aleatorios ate esgotar o orcamento."""
    perm = inicial.copy()
    melhor_c, melhor_p = math.inf, perm
    try:
        while True:
            p, c = hc_first(ev, rng, perm)
            if c < melhor_c:
                melhor_c, melhor_p = c, p.copy()
            perm = rng.permutation(N)   # novo ponto de partida aleatorio
    except OrcamentoEsgotado:
        return melhor_p, melhor_c


def esquema_temperatura(nome, T0, Tf, total):
    """Retorna T(k) para k=0..total-1, indo de T0 ate Tf conforme o esquema."""
    if nome == "geometrico":            # T_k = T0 * alfa^k
        alfa = (Tf / T0) ** (1.0 / (total - 1))
        return lambda k: T0 * alfa ** k
    if nome == "linear":                # T_k = T0 - (T0-Tf) k/(total-1)
        return lambda k: T0 - (T0 - Tf) * k / (total - 1)
    if nome == "logaritmico":           # T_k = T0 / (1 + c ln(1+k))
        c = (T0 / Tf - 1.0) / math.log(total)
        return lambda k: T0 / (1.0 + c * math.log(1 + k))
    raise ValueError(nome)


def simulated_annealing(ev, rng, inicial, esquema, T0, Tf, passos=None):
    """SA: troca aleatoria de duas teclas; piora aceita com prob. exp(-delta/T).

    Devolve o MELHOR layout visitado (nunca pior que o inicial).
    """
    passos = passos or ev.orcamento
    T = esquema_temperatura(esquema, T0, Tf, passos)
    perm = inicial.copy()
    atual = ev(perm)
    custo_inicial = atual
    melhor_c, melhor_p = atual, perm.copy()
    try:
        for k in range(passos - 1):
            i, j = PARES[rng.integers(len(PARES))]
            trocar(perm, i, j)
            c = ev(perm)
            d = c - atual
            if d <= 0 or rng.random() < math.exp(-d / T(k)):
                atual = c
                if c < melhor_c:
                    melhor_c, melhor_p = c, perm.copy()
            else:
                trocar(perm, i, j)      # rejeita: desfaz a troca
    except OrcamentoEsgotado:
        pass
    assert melhor_c <= custo_inicial + 1e-12, "SA devolveu pior que o inicial"
    return melhor_p, melhor_c


def local_beam(ev, rng, inicial, k=8, m=30, paciencia=10):
    """Local Beam Search: k estados; cada um gera m vizinhos aleatorios; mantem os k melhores.

    O conjunto candidato inclui os pais (elitismo) e duplicatas sao removidas;
    para quando o melhor nao melhora por 'paciencia' iteracoes (feixe convergiu).
    """
    estados = [inicial.copy()] + [rng.permutation(N) for _ in range(k - 1)]
    pool = [(ev(p), p) for p in estados]
    pool.sort(key=lambda t: t[0])
    melhor, sem_melhora = pool[0][0], 0
    try:
        while sem_melhora < paciencia:
            filhos = []
            for _, p in pool:
                for kk in rng.choice(len(PARES), size=m, replace=False):
                    i, j = PARES[kk]
                    q = p.copy()
                    trocar(q, i, j)
                    filhos.append((ev(q), q))
            todos, vistos = [], set()
            for c, p in sorted(pool + filhos, key=lambda t: t[0]):
                chave = p.tobytes()
                if chave not in vistos:
                    vistos.add(chave)
                    todos.append((c, p))
            pool = todos[:k]
            if pool[0][0] < melhor - 1e-12:
                melhor, sem_melhora = pool[0][0], 0
            else:
                sem_melhora += 1
    except OrcamentoEsgotado:
        pass
    return pool[0][1], pool[0][0]


# ----------------------------------------------------------------------------
# Utilitarios de experimento
# ----------------------------------------------------------------------------
def layout_qwerty():
    """Vetor perm[simbolo] = posicao para o QWERTY."""
    perm = np.empty(N, dtype=int)
    for pos, s in enumerate(QWERTY):
        perm[IDX[s]] = pos
    return perm


def calibrar_temperaturas(F, C, rng):
    """Devolve (piora media, T0): T0 aceita 50% da piora media de uma troca aleatoria.

    Amostra deltas de trocas aleatorias em layouts aleatorios (media das pioras).
    Tf = T0 / 1000 (fator fixo, definido em main) para a sensibilidade variar T0
    mantendo a razao.
    """
    ev = Avaliador(F, C, 1)
    pioras = []
    for _ in range(100):
        p = rng.permutation(N)
        c0 = ev.custo_puro(p)
        for _ in range(10):
            i, j = PARES[rng.integers(len(PARES))]
            trocar(p, i, j)
            d = ev.custo_puro(p) - c0
            trocar(p, i, j)
            if d > 0:
                pioras.append(d)
    media = float(np.mean(pioras))
    return media, -media / math.log(0.5)


def executar(nome_alg, funcao, F, C, seed, inicial, rng_offset):
    """Roda um algoritmo com orcamento fixo; devolve (custo, perm, evals, curva)."""
    ev = Avaliador(F, C, ORCAMENTO)
    rng = np.random.default_rng(seed * 100 + rng_offset)
    try:
        perm, c = funcao(ev, rng, inicial)
    except OrcamentoEsgotado:
        perm, c = ev.melhor_perm, ev.melhor
    # o custo final e sempre o melhor visto pelo avaliador (consistente entre algoritmos)
    assert abs(ev.melhor - ev.custo_puro(ev.melhor_perm)) < 1e-9
    return ev.melhor, ev.melhor_perm, ev.n, ev.curva_completa()


def estatisticas_layout(F, perm):
    """Metricas interpretaveis: % de bigramas em mesmo dedo, alternando maos e linha base."""
    pos_de, inv = perm, np.argsort(perm)  # inv[posicao] = simbolo
    mesmo_dedo = alterna = 0.0
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            pa, pb = pos_de[a], pos_de[b]
            if dedo_de(pa) == dedo_de(pb):
                mesmo_dedo += F[a, b]
            elif mao_de(pa) != mao_de(pb):
                alterna += F[a, b]
    uni = F.sum(axis=1) + F.sum(axis=0)
    uni = uni / uni.sum()
    casa = sum(uni[inv[p]] for p in range(N) if pos_rc(p)[0] == 1)
    dist_media = sum(uni[inv[p]] * (esforco_tecla(p)) for p in range(N))
    return dict(mesmo_dedo=100 * mesmo_dedo, alterna=100 * alterna,
                linha_casa=100 * casa, esforco_uni=dist_media)


def transformar(perm, f):
    """Aplica f(linha, coluna) -> (linha, coluna) a todas as teclas de um layout."""
    q = np.empty(N, dtype=int)
    for s in range(N):
        r, c = f(*pos_rc(perm[s]))
        q[s] = r * 10 + c
    return q


def layout_texto(perm):
    """Desenha o layout como 3 linhas de texto (uma tecla por posicao)."""
    inv = np.argsort(perm)  # inv[posicao] = simbolo
    linhas = []
    for r in range(3):
        linhas.append(" ".join(SIMBOLOS[inv[r * 10 + c]] for c in range(10)))
    return "\n".join("    " + l for l in linhas)


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
CORES = {"HC steepest": "#1f77b4", "HC first": "#17becf", "HC reinicios": "#2ca02c",
         "SA geometrico": "#d62728", "SA linear": "#ff7f0e", "SA logaritmico": "#9467bd",
         "Beam (k=8)": "#8c564b"}


def fig_convergencia(curvas, c_qwerty, c_alea, c_melhor):
    """(a) Curvas de convergencia: melhor custo medio (+-1 desvio) vs avaliacoes.

    Esquerda: visao completa (escala log em x). Direita: zoom nas ultimas
    avaliacoes, onde os algoritmos se separam.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6),
                            gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(1, ORCAMENTO + 1)
    for k, ax in enumerate(axs):
        for nome, M in curvas.items():
            m, s = M.mean(axis=0), M.std(axis=0)
            ax.plot(x, m, label=nome if k == 0 else None, color=CORES[nome], lw=1.6)
            ax.fill_between(x, m - s, m + s, color=CORES[nome], alpha=0.12)
        ax.axhline(c_melhor, color="green", ls="-.", lw=1,
                   label="melhor encontrado" if k == 0 else None)
        ax.set_xscale("log")
        ax.set_xlabel("avaliacoes de custo (escala log)")
        ax.grid(alpha=0.3)
    axs[0].axhline(c_qwerty, color="k", ls="--", lw=1, label="QWERTY")
    axs[0].axhline(c_alea, color="gray", ls=":", lw=1, label="aleatorio (media)")
    axs[0].set_ylim(c_melhor * 0.97, c_alea * 1.02)
    axs[0].set_ylabel("melhor custo ate agora (por bigrama)")
    axs[0].set_title(
        f"Convergencia (media de {N_SEEDS} sementes, faixa = 1 desvio)", fontsize=10)
    axs[0].legend(fontsize=7.5, ncol=2, loc="lower left")  # canto vazio: nao cobre as curvas
    axs[1].set_xlim(1000, ORCAMENTO)
    axs[1].set_ylim(c_melhor * 0.99, 1.25)
    axs[1].set_title("Zoom: de 1.000 a 30.000 avaliacoes", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "03_convergencia.png", dpi=150)
    plt.close(fig)


def fig_layouts(F, perm_q, perm_b, c_q, c_b):
    """(b) Layout QWERTY x melhor encontrado, cor = frequencia da tecla."""
    uni = F.sum(axis=1) + F.sum(axis=0)
    uni = 100 * uni / uni.sum()
    vmax = uni.max()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.0))
    for ax, perm, titulo in ((axs[0], perm_q, f"QWERTY (ABNT2)  custo = {c_q:.3f}"),
                             (axs[1], perm_b, f"Melhor encontrado  custo = {c_b:.3f}")):
        inv = np.argsort(perm)  # inv[posicao] = simbolo
        grade = np.array([[uni[inv[r * 10 + c]]
                         for c in range(10)] for r in range(3)])
        ax.imshow(grade, cmap="YlOrRd", vmin=0, vmax=vmax)
        for r in range(3):
            for c in range(10):
                s = inv[r * 10 + c]
                cor = "white" if uni[s] > 0.6 * vmax else "black"
                ax.text(c, r, SIMBOLOS[s].upper() if SIMBOLOS[s].isalpha() else SIMBOLOS[s],
                        ha="center", va="center", fontsize=15, color=cor, weight="bold")
                ax.text(c + 0.32, r + 0.34, f"{uni[s]:.1f}", ha="center", va="center",
                        fontsize=6.5, color=cor)
        for c in range(10):  # dedo atribuido por coluna
            ax.text(c, 2.78, ["M", "A", "Md", "I", "I", "I", "I", "Md", "A", "M"][c],
                    ha="center", va="top", fontsize=7, color="#444")
        ax.axvline(4.5, color="k", lw=1.2)   # separa as maos
        ax.axhline(0.5, color="w", lw=0.5)
        ax.axhline(1.5, color="w", lw=0.5)
        ax.set_title(titulo, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([1])
        ax.set_yticklabels(["linha base"], fontsize=7)
        ax.set_ylim(3.0, -0.5)
    fig.suptitle("Cor e numero pequeno = frequencia da tecla (% dos toques); "
                 "M=mindinho A=anelar Md=medio I=indicador", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "03_layouts.png", dpi=150)
    plt.close(fig)


def fig_analise(finais, otimos, c_qwerty, sens):
    """(c) Distribuicao dos resultados, otimos locais e sensibilidade a T0."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.0))
    nomes = list(finais)
    axs[0].boxplot([finais[n] for n in nomes], tick_labels=[
                   n.replace(" ", "\n") for n in nomes])
    axs[0].set_ylim(0.86, 0.97)   # zoom (QWERTY = 1,99 fica fora do quadro)
    axs[0].tick_params(axis="x", labelsize=6)
    axs[0].set_ylabel("custo final por bigrama")
    axs[0].set_title(f"Custo final ({N_SEEDS} sementes)", fontsize=10)
    axs[1].hist(otimos, bins=30, color="#4c72b0")
    axs[1].axvline(otimos.min(), color="green", ls="-.", lw=1, label="melhor otimo local")
    axs[1].set_xlabel("custo do otimo local")
    axs[1].set_ylabel("contagem")
    axs[1].set_title(f"{len(otimos)} otimos locais (HC first)", fontsize=10)
    axs[1].legend(fontsize=8)
    t0s = [s[0] for s in sens]
    axs[2].errorbar(t0s, [s[1] for s in sens], yerr=[s[2] for s in sens], marker="o",
                    capsize=3, color="#d62728")
    axs[2].set_ylim(0.86, 0.97)
    axs[2].set_xscale("log")
    axs[2].set_xlabel("T0 (Tf = T0/1000, SA geometrico)")
    axs[2].set_ylabel("custo final medio")
    axs[2].set_title("Sensibilidade a temperatura", fontsize=10)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "03_analise.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    t_ini = time.time()
    if not CORPUS.exists():
        raise SystemExit("corpus_pt.txt nao encontrado ao lado do script.")
    bruto = CORPUS.read_text(encoding="utf-8")
    texto = normalizar(bruto)
    F, ids = matriz_bigramas(texto)
    C = matriz_custo_posicoes()
    assert np.allclose(C, C.T) and abs(F.sum() - 1) < 1e-9

    say("=" * 72)
    say("PROJETO 3 - BUSCA COMPLEXA: LAYOUT DE TECLADO PARA PORTUGUES")
    say("=" * 72)
    say(f"Corpus: {len(bruto)} caracteres, {len(ids)} toques em teclas validas, "
        f"{int((F > 0).sum())} bigramas distintos de {N * N} possiveis.")
    uni = F.sum(axis=1) + F.sum(axis=0)
    uni = uni / uni.sum()
    top = np.argsort(-uni)[:8]
    say("Teclas mais frequentes: " +
        ", ".join(f"{SIMBOLOS[i]}={100 * uni[i]:.1f}%" for i in top))
    say(f"Espaco de busca: 30! = {math.factorial(N):.3e}; vizinhanca = {len(PARES)} trocas.")
    say(f"Orcamento = {ORCAMENTO} avaliacoes/execucao; {N_SEEDS} sementes.")

    # ---- baselines ----
    ev0 = Avaliador(F, C, 1)  # so para custo_puro (sem contagem)
    perm_q = layout_qwerty()
    c_q = ev0.custo_puro(perm_q)
    rng_b = np.random.default_rng(12345)
    c_alea = np.array([ev0.custo_puro(rng_b.permutation(N))
                      for _ in range(N_ALEATORIOS)])
    say("\n--- Baselines ---")
    say(f"QWERTY (ABNT2)      : custo = {c_q:.4f}")
    say(f"Aleatorio ({N_ALEATORIOS} layouts): media = {c_alea.mean():.4f}, desvio = "
        f"{c_alea.std():.4f}, min = {c_alea.min():.4f}, max = {c_alea.max():.4f}")
    say(f"QWERTY e {100 * (c_alea.mean() - c_q) / c_alea.mean():.1f}% melhor que o aleatorio medio.")

    # ---- calibracao de temperatura ----
    media_pior, T0 = calibrar_temperaturas(F, C, np.random.default_rng(7))
    Tf = T0 / 1000.0
    say(f"\nCalibracao SA: piora media de uma troca aleatoria = {media_pior:.4f}; "
        f"T0 = {T0:.4f} (aceita 50% da piora media), Tf = T0/1000 = {Tf:.6f}")

    # ---- algoritmos ----
    algoritmos = {
        "HC steepest": lambda ev, rng, ini: hc_steepest(ev, rng, ini),
        "HC first": lambda ev, rng, ini: hc_first(ev, rng, ini),
        "HC reinicios": lambda ev, rng, ini: hc_reinicios(ev, rng, ini),
        "SA geometrico": lambda ev, rng, ini: simulated_annealing(ev, rng, ini, "geometrico", T0, Tf),
        "SA linear": lambda ev, rng, ini: simulated_annealing(ev, rng, ini, "linear", T0, Tf),
        "SA logaritmico": lambda ev, rng, ini: simulated_annealing(ev, rng, ini, "logaritmico", T0, Tf),
        "Beam (k=8)": lambda ev, rng, ini: local_beam(ev, rng, ini, k=8, m=30),
    }
    finais = {n: [] for n in algoritmos}
    evals = {n: [] for n in algoritmos}
    curvas = {n: [] for n in algoritmos}
    perms = {n: [] for n in algoritmos}
    inicial_c = []
    for seed in range(N_SEEDS):
        inicial = np.random.default_rng(
            1000 + seed).permutation(N)  # mesmo inicio p/ todos
        inicial_c.append(ev0.custo_puro(inicial))
        for off, (nome, fn) in enumerate(algoritmos.items()):
            c, p, n_ev, curva = executar(nome, fn, F, C, seed, inicial, off)
            assert c <= inicial_c[-1] + 1e-12, f"{nome} pior que o inicial"
            finais[nome].append(c)
            evals[nome].append(n_ev)
            curvas[nome].append(curva)
            perms[nome].append(p)
    curvas = {n: np.array(v) for n, v in curvas.items()}

    melhor_global, perm_b, nome_b = math.inf, None, None
    for n in algoritmos:
        k = int(np.argmin(finais[n]))
        if finais[n][k] < melhor_global:
            melhor_global, perm_b, nome_b = finais[n][k], perms[n][k], n

    say("\n--- Resultados (custo por bigrama; menor e melhor) ---")
    say(
        f"Custo medio dos layouts iniciais (aleatorios, mesmas sementes): {np.mean(inicial_c):.4f}")
    cab = (f"{'algoritmo':<16}{'media':>8}{'desvio':>8}{'min':>8}{'max':>8}"
           f"{'evals(med)':>12}{'vs QWERTY':>11}{'gap/melhor':>12}{'bate QWERTY':>13}")
    say(cab)
    say("-" * len(cab))
    for n in algoritmos:
        f = np.array(finais[n])
        vs = 100 * (c_q - f.mean()) / c_q
        gap = 100 * (f.mean() - melhor_global) / melhor_global
        bate = int((f < c_q).sum())
        say(f"{n:<16}{f.mean():>8.4f}{f.std():>8.4f}{f.min():>8.4f}{f.max():>8.4f}"
            f"{np.mean(evals[n]):>12.0f}{vs:>10.1f}%{gap:>11.2f}%{bate:>10d}/{N_SEEDS}")
    say(f"Melhor layout global: custo = {melhor_global:.4f} (algoritmo: {nome_b}); "
        f"reducao vs QWERTY = {100 * (c_q - melhor_global) / c_q:.1f}%")

    # avaliacoes ate superar o QWERTY (media das sementes que superaram)
    say("\nAvaliacoes ate o custo ficar abaixo do QWERTY (media sobre sementes que conseguiram):")
    for n in algoritmos:
        alvo = []
        for M in curvas[n]:
            k = np.argmax(M < c_q)
            if M[k] < c_q:
                alvo.append(k + 1)
        say(f"  {n:<16}: {np.mean(alvo):8.0f}  ({len(alvo)}/{N_SEEDS})" if alvo else f"  {n:<16}: nunca")

    # melhores layouts
    say("\nQWERTY na grade 3x10:")
    say(layout_texto(perm_q))
    say("Melhor layout encontrado:")
    say(layout_texto(perm_b))
    sq, sb = estatisticas_layout(F, perm_q), estatisticas_layout(F, perm_b)
    say(f"\n{'metrica':<38}{'QWERTY':>10}{'melhor':>10}")
    say(
        f"{'bigramas no mesmo dedo (%)':<38}{sq['mesmo_dedo']:>10.2f}{sb['mesmo_dedo']:>10.2f}")
    say(
        f"{'bigramas alternando maos (%)':<38}{sq['alterna']:>10.2f}{sb['alterna']:>10.2f}")
    say(
        f"{'toques na linha base (%)':<38}{sq['linha_casa']:>10.2f}{sb['linha_casa']:>10.2f}")
    say(
        f"{'esforco individual medio por toque':<38}{sq['esforco_uni']:>10.3f}{sb['esforco_uni']:>10.3f}")

    # ---- otimos locais ----
    say(f"\n--- Otimos locais: {N_OTIMOS_LOCAIS} partidas aleatorias de HC first-improvement ---")
    rng_o = np.random.default_rng(99)
    otimos, layouts_otimos, evs_otimos = [], set(), []
    for _ in range(N_OTIMOS_LOCAIS):
        ev = Avaliador(F, C, 20_000)
        p, c = hc_first(ev, rng_o, rng_o.permutation(N))
        otimos.append(c)
        layouts_otimos.add(p.tobytes())
        evs_otimos.append(ev.n)
    otimos = np.array(otimos)
    distintos_custo = len(set(np.round(otimos, 6)))
    say(f"Custo dos otimos locais: media = {otimos.mean():.4f}, desvio = {otimos.std():.4f}, "
        f"min = {otimos.min():.4f}, max = {otimos.max():.4f}")
    say(f"Layouts distintos: {len(layouts_otimos)}/{N_OTIMOS_LOCAIS}; custos distintos "
        f"(6 casas): {distintos_custo}; avaliacoes medias ate o otimo local: {np.mean(evs_otimos):.0f}")
    say(f"Fracao dos otimos locais dentro de 1% do melhor global: "
        f"{100 * np.mean(otimos <= melhor_global * 1.01):.1f}%; "
        f"que atingem o melhor global: {100 * np.mean(otimos <= melhor_global + 1e-9):.1f}%; "
        f"piores que o QWERTY: {100 * np.mean(otimos > c_q):.1f}%")

    # ---- sensibilidade a temperatura ----
    say("\n--- Sensibilidade do SA geometrico a T0 (Tf = T0/1000; 10 sementes) ---")
    say(f"{'T0':>10}{'media':>9}{'desvio':>9}{'min':>9}")
    sens = []
    for fator in (0.001, 0.01, 0.1, 1, 10, 100):
        t0 = T0 * fator
        cs = []
        for seed in range(10):
            ini = np.random.default_rng(1000 + seed).permutation(N)
            c, *_ = executar("sa", lambda ev, rng, i: simulated_annealing(
                ev, rng, i, "geometrico", t0, t0 / 1000.0), F, C, seed, ini, 50)
            cs.append(c)
        sens.append((t0, float(np.mean(cs)), float(np.std(cs))))
        say(f"{t0:>10.5f}{np.mean(cs):>9.4f}{np.std(cs):>9.4f}{np.min(cs):>9.4f}"
            f"   (T0 = {fator} x calibrado)")

    # ---- figuras ----
    fig_convergencia(curvas, c_q, c_alea.mean(), melhor_global)
    fig_layouts(F, perm_q, perm_b, c_q, melhor_global)
    fig_analise(finais, otimos, c_q, sens)

    # ---- asserts finais (autoverificacao) ----
    assert melhor_global < c_q, "o melhor layout deveria ser melhor que o QWERTY"
    assert c_q < c_alea.mean(), "QWERTY deveria ser melhor que a media aleatoria"
    # simetrias do modelo: espelho entre maos e inversao cima/baixo preservam o custo
    for f in (lambda r, c: (r, 9 - c), lambda r, c: (2 - r, c), lambda r, c: (2 - r, 9 - c)):
        assert abs(ev0.custo_puro(transformar(perm_b, f)) - melhor_global) < 1e-9
    assert sorted(perm_b.tolist()) == list(
        range(N)), "layout invalido (nao e permutacao)"
    for n in algoritmos:
        assert np.all(np.array(finais[n]) <= np.array(inicial_c) + 1e-12)
    say(f"\nAsserts OK. Tempo total: {time.time() - t_ini:.1f} s")
    (FIGS / "03_saida.txt").write_text("\n".join(SAIDA) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
