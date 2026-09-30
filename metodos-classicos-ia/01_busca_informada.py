"""
Projeto 1 - Busca Informada: alinhamento de sequencias de DNA como busca A*.

PROBLEMA
    Dadas duas sequencias A (tamanho n) e B (tamanho m) sobre o alfabeto {A,C,G,T},
    encontrar o alinhamento global de MENOR CUSTO (distancia de edicao ponderada).

FORMULACAO COMO PROBLEMA DE BUSCA
    Estado   : (i, j)  -> i caracteres de A e j caracteres de B ja foram consumidos.
    Inicial  : (0, 0).   Objetivo: (n, m).
    Acoes    : - diagonal  (i+1, j+1): custo 0 se A[i]==B[j] (match), senao custo
                 MISMATCH (substituicao);
               - gap em B  (i+1, j)  : custo GAP (A[i] alinhado a um "-");
               - gap em A  (i,   j+1): custo GAP (B[j] alinhado a um "-").
    Custo    : soma dos custos das acoes (o grafo e um DAG em grade (n+1)x(m+1)).

HEURISTICA
    h(i, j) = GAP * |(n - i) - (m - j)|
    Admissivel: qualquer caminho ate (n, m) precisa de ao menos |diferenca de
    comprimentos restantes| gaps, cada um custando GAP. Tambem e CONSISTENTE
    (h(s) <= c(s,s') + h(s')), verificado por assert em verificar_heuristica().

ALGORITMOS IMPLEMENTADOS A MAO
    - Needleman-Wunsch (programacao dinamica) : verdade de referencia (ground truth).
    - UCS/Dijkstra  : prioridade = g            (h ignorada)
    - A*            : prioridade = g + h        (otimo, pois h e admissivel)
    - Weighted A*   : prioridade = g + w*h, w em {1.5, 2, 3}  (pode ser subotimo,
                      custo <= w * otimo)
    - Greedy best-first : prioridade = h        (rapido, sem garantia de otimalidade)

Todas as buscas compartilham UMA unica funcao (busca) parametrizada pelos pesos
(wg, wh) de g e h, o que garante comparacao justa (mesmo desempate, mesma fila).

Execucao: python 01_busca_informada.py   (gera figs/*.png e figs/01_saida.txt)
"""

import heapq
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela (salva apenas arquivos)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter

# ----------------------------------------------------------------------------
# Parametros globais do problema (parametrizaveis)
# ----------------------------------------------------------------------------
GAP = 2        # custo de abrir/estender um gap (indel)
MISMATCH = 1   # custo de uma substituicao
ALFABETO = "ACGT"
PASTA = Path(__file__).resolve().parent
FIGS = PASTA / "figs"

# Algoritmos: nome -> (peso de g, peso de h)
ALGORITMOS = {
    "UCS": (1.0, 0.0),
    "A*": (1.0, 1.0),
    "WA* w=1.5": (1.0, 1.5),
    "WA* w=2": (1.0, 2.0),
    "WA* w=3": (1.0, 3.0),
    "Greedy": (0.0, 1.0),
}

_saida = []  # acumula tudo o que e impresso, para salvar em figs/01_saida.txt


def log(texto=""):
    """Imprime na tela e guarda a linha para o arquivo de saida."""
    print(texto)
    _saida.append(str(texto))


# ----------------------------------------------------------------------------
# Geracao de dados sinteticos
# ----------------------------------------------------------------------------
def gerar_par(tamanho, divergencia, rng):
    """Gera (A, B): A aleatoria; B = A mutada.

    Para cada posicao de A, com probabilidade `divergencia` ocorre um evento:
    70% substituicao, 15% delecao, 15% insercao de uma base aleatoria.
    """
    a = "".join(rng.choice(ALFABETO) for _ in range(tamanho))
    b = []
    for c in a:
        if rng.random() < divergencia:
            r = rng.random()
            if r < 0.70:  # substituicao por base diferente
                b.append(rng.choice([x for x in ALFABETO if x != c]))
            elif r < 0.85:  # delecao: nao copia o caractere
                pass
            else:  # insercao: base nova antes do caractere original
                b.append(rng.choice(ALFABETO))
                b.append(c)
        else:
            b.append(c)
    return a, "".join(b)


# ----------------------------------------------------------------------------
# Verdade de referencia: Needleman-Wunsch (programacao dinamica)
# ----------------------------------------------------------------------------
def needleman_wunsch(a, b, gap=GAP, mism=MISMATCH):
    """Retorna (custo_otimo, caminho) preenchendo toda a matriz (n+1)x(m+1).

    D[i][j] = custo minimo para alinhar A[:i] com B[:j]. O caminho e recuperado
    por traceback a partir de (n, m), preferindo a diagonal em empates.
    """
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * gap
    for j in range(1, m + 1):
        D[0][j] = j * gap
    for i in range(1, n + 1):
        ai = a[i - 1]
        Di, Dp = D[i], D[i - 1]
        for j in range(1, m + 1):
            diag = Dp[j - 1] + (0 if ai == b[j - 1] else mism)
            Di[j] = min(diag, Dp[j] + gap, Di[j - 1] + gap)
    # traceback
    i, j = n, m
    caminho = [(i, j)]
    while (i, j) != (0, 0):
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else mism):
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
        caminho.append((i, j))
    caminho.reverse()
    return D[n][m], caminho


def custo_caminho(a, b, caminho, gap=GAP, mism=MISMATCH):
    """Recalcula o custo de um caminho (lista de estados) e valida cada passo."""
    assert caminho[0] == (0, 0) and caminho[-1] == (len(a),
                                                    len(b)), "caminho nao vai de (0,0) a (n,m)"
    total = 0
    for (i, j), (k, l) in zip(caminho, caminho[1:]):
        if (k, l) == (i + 1, j + 1):
            total += 0 if a[i] == b[j] else mism
        elif (k, l) in ((i + 1, j), (i, j + 1)):
            total += gap
        else:
            raise AssertionError(f"passo invalido {(i, j)} -> {(k, l)}")
    return total


# ----------------------------------------------------------------------------
# Busca generica em melhor-primeiro (UCS, A*, WA*, Greedy)
# ----------------------------------------------------------------------------
def busca(a, b, wg, wh, gap=GAP, mism=MISMATCH):
    """Busca melhor-primeiro com prioridade f = wg*g + wh*h.

    (wg,wh) = (1,0) UCS | (1,1) A* | (1,w) WA* | (0,1) Greedy.
    Desempate: prefere estados mais profundos (maior i+j), comum a todos.
    Retorna dicionario com custo, caminho, numero de expansoes, tempo (s) e a
    matriz booleana de celulas expandidas.
    """
    n, m = len(a), len(b)
    W = m + 1
    INF = float("inf")
    g = [INF] * ((n + 1) * W)     # melhor custo conhecido ate cada celula
    pai = [-1] * ((n + 1) * W)    # ponteiros para reconstruir o caminho
    exp_mat = np.zeros((n + 1, W), dtype=bool)

    def h(i, j):
        """Heuristica: GAP * |diferenca dos comprimentos restantes|."""
        return gap * abs((n - i) - (m - j))

    t0 = time.perf_counter()
    g[0] = 0
    fila = [(wh * h(0, 0), 0, 0, 0, 0)]  # (prioridade, desempate, g, i, j)
    expansoes = 0
    meta = n * W + m
    while fila:
        _, _, gc, i, j = heapq.heappop(fila)
        idx = i * W + j
        if gc > g[idx]:
            # entrada obsoleta na fila (foi encontrado caminho melhor)
            continue
        expansoes += 1
        exp_mat[i, j] = True
        if idx == meta:
            break
        # sucessores: (destino_i, destino_j, custo)
        suc = []
        if i < n and j < m:
            suc.append((i + 1, j + 1, 0 if a[i] == b[j] else mism))
        if i < n:
            suc.append((i + 1, j, gap))
        if j < m:
            suc.append((i, j + 1, gap))
        for k, l, c in suc:
            nidx = k * W + l
            ng = gc + c
            # Greedy (wg=0) nao reabre nos: so insere celulas nunca vistas.
            melhora = ng < g[nidx] if wg > 0 else g[nidx] == INF
            if melhora:
                g[nidx] = ng
                pai[nidx] = idx
                heapq.heappush(
                    fila, (wg * ng + wh * h(k, l), -(k + l), ng, k, l))
    tempo = time.perf_counter() - t0

    caminho = []
    idx = meta
    while idx != -1:
        caminho.append((idx // W, idx % W))
        idx = pai[idx]
    caminho.reverse()
    return {
        "custo": custo_caminho(a, b, caminho, gap, mism),
        "caminho": caminho,
        "exp": expansoes,
        "celulas": int(exp_mat.sum()),
        "tempo": tempo,
        "mat": exp_mat,
    }


# ----------------------------------------------------------------------------
# Autoverificacao da heuristica
# ----------------------------------------------------------------------------
def verificar_heuristica(gap=GAP, mism=MISMATCH):
    """Confere admissibilidade (h <= custo real restante via NW) e consistencia."""
    rng = random.Random(7)
    # 3 pares: dois mutados e um com B truncado (comprimentos bem diferentes,
    # onde h > 0 em muitas celulas e o teste realmente aperta a heuristica)
    pares = [gerar_par(30, 0.3, rng), gerar_par(30, 0.1, rng)]
    a0, b0 = gerar_par(30, 0.05, rng)
    pares.append((a0, b0[:20]))
    for a, b in pares:
        n, m = len(a), len(b)

        def h(i, j):
            """Mesma heuristica usada em busca()."""
            return gap * abs((n - i) - (m - j))

        for i in range(n + 1):
            for j in range(m + 1):
                # consistencia em cada aresta
                for k, l, c in ((i + 1, j + 1, mism), (i + 1, j, gap), (i, j + 1, gap)):
                    if k <= n and l <= m:
                        c_min = 0 if (k == i + 1 and l == j + 1 and a[i] == b[j]) else c
                        assert h(i, j) <= c_min + h(k, l), "heuristica inconsistente"
                # admissibilidade: custo real do sufixo via NW
                real, _ = needleman_wunsch(a[i:], b[j:], gap, mism)
                assert h(i, j) <= real, "heuristica inadmissivel"
    log("[ok] heuristica verificada: admissivel e consistente (3 pares de ~30 bases, todas as celulas)")


# ----------------------------------------------------------------------------
# Experimento principal
# ----------------------------------------------------------------------------
def experimento_principal(tamanhos, divergencias, reps):
    """Roda todos os algoritmos em pares sinteticos; retorna lista de registros."""
    regs = []
    for L in tamanhos:
        for dv in divergencias:
            for r in range(reps):
                rng = random.Random(1000 * L + int(dv * 100) * 10 + r)
                a, b = gerar_par(L, dv, rng)
                otimo, cam_nw = needleman_wunsch(a, b)
                assert custo_caminho(
                    a, b, cam_nw) == otimo, "traceback do NW inconsistente"
                for nome, (wg, wh) in ALGORITMOS.items():
                    res = busca(a, b, wg, wh)
                    tot = (len(a) + 1) * (len(b) + 1)
                    if nome in ("UCS", "A*"):
                        assert res["custo"] == otimo, f"{nome} nao otimo: {res['custo']} != {otimo}"
                    assert res["custo"] >= otimo, f"{nome} custo abaixo do otimo (impossivel)"
                    if nome.startswith("WA*"):
                        w = ALGORITMOS[nome][1]
                        assert res["custo"] <= w * otimo + \
                            1e-9, f"{nome} violou limite w*otimo"
                    regs.append({"L": L, "div": dv, "rep": r, "alg": nome, "otimo": otimo,
                                 "custo": res["custo"], "exp": res["exp"],
                                 "frac": res["celulas"] / tot, "tempo": res["tempo"]})
    return regs


def media(regs, **filtros):
    """Media de cada metrica sobre os registros que casam com os filtros."""
    sel = [r for r in regs if all(r[k] == v for k, v in filtros.items())]
    return {k: float(np.mean([r[k] for r in sel])) for k in ("exp", "frac", "tempo", "custo", "otimo")}


def imprimir_tabelas(regs, tamanhos, divergencias):
    """Tabelas: expansoes, fracao da matriz, tempo e subotimalidade."""
    nomes = list(ALGORITMOS)
    cab = f"{'L':>5} | " + " | ".join(f"{n:>10}" for n in nomes)
    for titulo, chave, fmt in (
        ("Nos expandidos (media de expansoes)", "exp", "{:10.0f}"),
        ("Fracao da matriz DP explorada (%)", "frac", "{:10.1f}"),
        ("Tempo (ms)", "tempo", "{:10.1f}"),
    ):
        log(f"\n=== {titulo} ===")
        for dv in divergencias:
            log(f"-- divergencia {int(dv * 100)}% --")
            log(cab)
            for L in tamanhos:
                vals = []
                for n in nomes:
                    v = media(regs, L=L, div=dv, alg=n)[chave]
                    v = v * 100 if chave == "frac" else v * 1000 if chave == "tempo" else v
                    vals.append(fmt.format(v))
                log(f"{L:>5} | " + " | ".join(vals))

    log("\n=== Subotimalidade (gap % = (custo - otimo)/otimo) ===")
    log(f"{'algoritmo':>10} | {'div':>4} | {'subotimos':>10} | {'gap medio %':>11} | {'gap max %':>9}")
    for n in nomes:
        for dv in divergencias:
            sel = [r for r in regs if r["alg"] == n and r["div"] == dv]
            gaps = [100 * (r["custo"] - r["otimo"]) / r["otimo"] for r in sel]
            sub = sum(1 for x in gaps if x > 1e-9)
            log(f"{n:>10} | {int(dv * 100):>3}% | {sub:>4}/{len(sel):<5} | {np.mean(gaps):>11.2f} | {max(gaps):>9.2f}")


# ----------------------------------------------------------------------------
# Experimento extra: sequencias com blocos ausentes (comprimentos diferentes)
# ----------------------------------------------------------------------------
def experimento_bloco(L=200, blocos=(0, 10, 30, 60), reps=3):
    """B = A com um bloco contiguo removido + 5% de divergencia.

    Aqui a heuristica (diferenca de comprimentos) carrega informacao real.
    """
    log(f"\n=== Experimento extra: bloco removido (L={L}, 5% de divergencia) ===")
    log(f"{'bloco':>6} | {'|n-m|':>6} | {'UCS exp':>9} | {'A* exp':>9} | {'A*/UCS':>7} | {'Greedy gap%':>11}")
    linhas = []
    for k in blocos:
        e_u, e_a, gg, dif = [], [], [], []
        for r in range(reps):
            rng = random.Random(555 + 10 * k + r)
            a, b = gerar_par(L, 0.05, rng)
            pos = rng.randrange(20, L - k - 20) if k else 0
            b = b[:pos] + b[pos + k:]
            otimo, _ = needleman_wunsch(a, b)
            u, s, gr = busca(a, b, 1, 0), busca(a, b, 1, 1), busca(a, b, 0, 1)
            assert u["custo"] == otimo and s["custo"] == otimo
            e_u.append(u["exp"])
            e_a.append(s["exp"])
            dif.append(abs(len(a) - len(b)))
            gg.append(100 * (gr["custo"] - otimo) / otimo)
        linhas.append((k, np.mean(dif), np.mean(
            e_u), np.mean(e_a), np.mean(gg)))
        log(f"{k:>6} | {np.mean(dif):>6.0f} | {np.mean(e_u):>9.0f} | {np.mean(e_a):>9.0f} | "
            f"{np.mean(e_a) / np.mean(e_u):>7.3f} | {np.mean(gg):>11.1f}")
    return linhas


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
CORES = {"UCS": "#7f7f7f", "A*": "#0072B2", "WA* w=1.5": "#56B4E9", "WA* w=2": "#009E73",
         "WA* w=3": "#E69F00", "Greedy": "#D55E00"}


def fig_heatmap(L=100):
    """Fig (a): celulas expandidas por A*, UCS e Greedy; linhas = 5% e 30% de divergencia."""
    fig, eixos = plt.subplots(2, 3, figsize=(11, 7.4))
    cmap = ListedColormap(["#f2f2f2", "#7fb3e0"])
    for row, dv in enumerate((0.05, 0.30)):
        a, b = gerar_par(L, dv, random.Random(2024 + row))
        otimo, cam = needleman_wunsch(a, b)
        for col, nome in enumerate(("A*", "UCS", "Greedy")):
            wg, wh = ALGORITMOS[nome]
            res = busca(a, b, wg, wh)
            ax = eixos[row, col]
            ax.imshow(res["mat"], cmap=cmap, aspect="equal",
                      interpolation="nearest")
            ax.plot([p[1] for p in cam], [p[0] for p in cam],
                    color="#c00000", lw=0.8, label="caminho otimo (NW)")
            if nome == "Greedy":
                ax.plot([p[1] for p in res["caminho"]], [p[0] for p in res["caminho"]],
                        color="#222222", lw=1.0, ls="--", label="caminho do Greedy")
            frac = 100 * res["celulas"] / ((len(a) + 1) * (len(b) + 1))
            ax.set_title(f"{nome} | div {int(dv * 100)}%\n{res['exp']} exp. ({frac:.1f}%), custo {res['custo']} (otimo {otimo})",
                         fontsize=9)
            ax.set_xlabel("j (caracteres de B)", fontsize=8)
            ax.set_ylabel("i (caracteres de A)", fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Celulas expandidas (azul) sobre a matriz de alinhamento, L=100", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "01_heatmap_expandidas.png", dpi=130)
    plt.close(fig)


def fig_expandidos(regs, tamanhos, divergencias):
    """Fig (b): nos expandidos vs tamanho, um painel por divergencia (escala log-log)."""
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, dv in zip(eixos, divergencias):
        for nome in ALGORITMOS:
            ys = [media(regs, L=L, div=dv, alg=nome)["exp"] for L in tamanhos]
            ax.plot(tamanhos, ys, marker="o", ms=4,
                    color=CORES[nome], label=nome)
        ax.plot(tamanhos, [(L + 1) ** 2 for L in tamanhos],
                "k:", lw=1, label="(L+1)^2 (matriz toda)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(tamanhos)
        ax.set_xticklabels(tamanhos)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"divergencia {int(dv * 100)}%", fontsize=10)
        ax.set_xlabel("comprimento L")
        ax.grid(alpha=0.3)
    eixos[0].set_ylabel("nos expandidos (media)")
    eixos[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "01_expandidos_vs_tamanho.png", dpi=130)
    plt.close(fig)


def fig_fracao(regs, divergencias, L):
    """Fig (c): fracao da matriz explorada vs divergencia para o maior L."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for nome in ("UCS", "A*", "WA* w=2", "Greedy"):
        ys = [100 * media(regs, L=L, div=dv, alg=nome)["frac"]
              for dv in divergencias]
        ax.plot([int(d * 100) for d in divergencias], ys,
                marker="o", color=CORES[nome], label=nome)
    ax.set_xticks([int(d * 100) for d in divergencias])
    ax.set_xlabel("divergencia (%)")
    ax.set_ylabel("matriz DP explorada (%)")
    ax.set_title(f"Fracao explorada vs divergencia (L={L})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "01_fracao_vs_divergencia.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    FIGS.mkdir(exist_ok=True)
    log("PROJETO 1 - Busca informada: alinhamento de DNA como A*")
    log(f"Parametros: GAP={GAP}, MISMATCH={MISMATCH}, alfabeto={ALFABETO}")
    verificar_heuristica()

    # Exemplo pequeno e legivel
    a, b = "GATTACA", "GCATGCU".replace("U", "T")
    otimo, cam = needleman_wunsch(a, b)
    for nome in ("A*", "UCS"):
        assert busca(a, b, *ALGORITMOS[nome])["custo"] == otimo
    log(f"[ok] exemplo {a} x {b}: custo otimo NW = {otimo} (A* e UCS conferem)")

    tamanhos, divergencias, reps = [50, 100, 200, 400], [0.05, 0.15, 0.30], 3
    log(f"\nExperimento principal: L={tamanhos}, divergencias={divergencias}, {reps} pares por celula")
    regs = experimento_principal(tamanhos, divergencias, reps)
    log("[ok] A* e UCS igualaram o custo do Needleman-Wunsch em todos os "
        f"{len(regs) // len(ALGORITMOS)} pares")
    imprimir_tabelas(regs, tamanhos, divergencias)

    # razao A*/UCS resumida
    log("\n=== Razao de expansoes A*/UCS (media) ===")
    log(f"{'L':>5} | " + " | ".join(f"div {int(d * 100):>2}%" for d in divergencias))
    for L in tamanhos:
        rs = [media(regs, L=L, div=d, alg="A*")["exp"] / media(regs,
                                                               L=L, div=d, alg="UCS")["exp"] for d in divergencias]
        log(f"{L:>5} | " + " | ".join(f"{x:7.3f}" for x in rs))

    experimento_bloco()

    fig_heatmap()
    fig_expandidos(regs, tamanhos, divergencias)
    fig_fracao(regs, divergencias, tamanhos[-1])
    log(f"\nFiguras salvas em {FIGS}")
    log(f"Tempo total: {time.perf_counter() - t0:.1f} s")
    (FIGS / "01_saida.txt").write_text("\n".join(_saida) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
