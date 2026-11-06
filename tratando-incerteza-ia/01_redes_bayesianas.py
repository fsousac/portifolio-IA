"""
Projeto 1 - Redes Bayesianas: diagnostico respiratorio com nos ruidosos (noisy-OR).

PROBLEMA
    Um consultorio quer estimar, a partir de fatores de risco e sintomas observados,
    a probabilidade de cada doenca respiratoria. A rede tem 15 variaveis binarias em
    3 camadas (fatores de risco -> doencas -> sintomas/exame):
        Fumante, Poluicao, Contato           (raizes, com probabilidade a priori)
        Gripe, Resfriado, Asma, Bronquite, Pneumonia
        Febre, Tosse, Dispneia, Dor_corpo, Coriza, Chiado, RaioX
    As relacoes causa -> efeito sao modeladas como NOISY-OR: cada causa presente
    "falha" em produzir o efeito com probabilidade independente (1 - p_i) e ha uma
    probabilidade de vazamento (leak) para o efeito aparecer sem causa listada.
    Isso reduz o numero de parametros de 2^k para k+1 por no (Tosse tem 5 pais:
    32 linhas na tabela cheia contra 6 parametros noisy-OR).

ALGORITMOS IMPLEMENTADOS A MAO (so numpy; nenhuma biblioteca de redes bayesianas)
    - Fatores (produto, soma, restricao) com np.einsum.
    - Inferencia exata por ENUMERACAO (Russell & Norvig, ENUMERATION-ASK).
    - Inferencia exata por ELIMINACAO DE VARIAVEIS com ordem gulosa (min-fill,
      min-size), poda de nos barren (nao ancestrais da consulta/evidencia) e contagem
      de operacoes.
    - Inferencia aproximada: amostragem com REJEICAO, PONDERACAO POR VEROSSIMILHANCA
      (likelihood weighting) e AMOSTRAGEM DE GIBBS (cadeia de Markov no manto de
      Markov, com estimador Rao-Blackwell).
    - APRENDIZADO de parametros por contagem (maxima verossimilhanca com suavizacao
      de Dirichlet/Laplace) a partir de dados amostrados.

VERDADE DE REFERENCIA
    A distribuicao conjunta completa (2^15 = 32768 entradas) e construida por forca
    bruta; toda posteriori exata e comparada a ela. A KL(P_real || P_aprendida) tambem
    e calculada sobre a conjunta completa.

AUTOVERIFICACAO (assert em main)
    CPTs somam 1; conjunta soma 1; enumeracao == eliminacao == forca bruta (todas as
    15 marginais posteriores); "explaining away" (Gripe explica a febre e reduz
    Pneumonia); amostradores convergem; KL do aprendizado decresce com N; custo da
    enumeracao cresce exponencialmente e o da eliminacao nao.

SIMPLIFICACOES E LIMITES
    - Parametros do consultorio sao plausiveis, mas inventados (nao sao dados
      clinicos reais): o objetivo e estudar inferencia, nao medicina.
    - Variaveis binarias; estrutura conhecida (so os parametros sao aprendidos).
    - Gibbs implementado em python puro (lento); por isso poucas sementes.

Execucao: python 01_redes_bayesianas.py  (gera figs/01_*.png e figs/01_saida.txt)
"""

import itertools
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PASTA = Path(__file__).resolve().parent
FIGS = PASTA / "figs"
FIGS.mkdir(exist_ok=True)

_LOG = []


def _saida(*args):
    """Imprime e acumula a linha para gravar em figs/01_saida.txt."""
    linha = " ".join(str(a) for a in args)
    print(linha)
    _LOG.append(linha)


# ----------------------------------------------------------------------------
# Fatores
# ----------------------------------------------------------------------------
class Fator:
    """Tabela sobre variaveis binarias: eixo k de `t` corresponde a vars[k]."""

    def __init__(self, vars_, t):
        self.vars = list(vars_)
        self.t = np.asarray(t, dtype=float)


def multiplicar(a, b):
    """Produto de fatores (alinhando variaveis por nome via einsum)."""
    vs = list(a.vars) + [v for v in b.vars if v not in a.vars]
    L = {v: chr(97 + i) for i, v in enumerate(vs)}
    eq = "".join(L[v] for v in a.vars) + "," + "".join(L[v] for v in b.vars) + "->" + "".join(L[v] for v in vs)
    return Fator(vs, np.einsum(eq, a.t, b.t))


def somar_fora(f, v):
    ax = f.vars.index(v)
    return Fator([u for u in f.vars if u != v], f.t.sum(axis=ax))


def restringir(f, v, val):
    """Fixa v=val (evidencia) e remove a variavel do fator."""
    if v not in f.vars:
        return f
    ax = f.vars.index(v)
    return Fator([u for u in f.vars if u != v], np.take(f.t, val, axis=ax))


# ----------------------------------------------------------------------------
# Rede bayesiana
# ----------------------------------------------------------------------------
def noisy_or(ps, leak):
    """CPT noisy-OR: P(efeito=0 | pais) = (1-leak) * prod_{i presente}(1-p_i).
    Retorna tabela de forma (2,)*k + (2,), ultimo eixo = valor do filho."""
    k = len(ps)
    cpt = np.zeros((2,) * k + (2,))
    for idx in itertools.product((0, 1), repeat=k):
        q = (1 - leak) * np.prod([(1 - p) for p, x in zip(ps, idx) if x])
        cpt[idx + (0,)] = q
        cpt[idx + (1,)] = 1 - q
    return cpt


class Rede:
    def __init__(self):
        self.order = []        # ordem topologica
        self.pais = {}
        self.cpt = {}
        self.filhos = {}

    def add(self, nome, pais, cpt):
        self.order.append(nome)
        self.pais[nome] = list(pais)
        self.cpt[nome] = np.asarray(cpt, dtype=float)
        self.filhos.setdefault(nome, [])
        for p in pais:
            self.filhos[p].append(nome)

    def fator(self, v):
        return Fator(self.pais[v] + [v], self.cpt[v])

    def p(self, v, estado):
        """P(v = estado[v] | pais = estado[pais]) para um dict de valores."""
        return self.cpt[v][tuple(estado[p] for p in self.pais[v]) + (estado[v],)]

    def n_parametros(self):
        return sum(2 ** len(self.pais[v]) for v in self.order)


def ancestrais(rede, alvo):
    vis, pilha = set(), list(alvo)
    while pilha:
        v = pilha.pop()
        if v in vis:
            continue
        vis.add(v)
        pilha.extend(rede.pais[v])
    return vis


def rede_clinica():
    r = Rede()
    r.add("Fumante", [], np.array([0.75, 0.25]))
    r.add("Poluicao", [], np.array([0.70, 0.30]))
    r.add("Contato", [], np.array([0.80, 0.20]))
    r.add("Gripe", ["Contato"], noisy_or([0.50], 0.02))
    r.add("Resfriado", ["Contato"], noisy_or([0.40], 0.05))
    r.add("Asma", ["Poluicao"], noisy_or([0.20], 0.03))
    r.add("Bronquite", ["Fumante", "Poluicao"], noisy_or([0.25, 0.15], 0.01))
    r.add("Pneumonia", ["Fumante", "Gripe", "Bronquite"], noisy_or([0.05, 0.08, 0.10], 0.003))
    r.add("Febre", ["Gripe", "Resfriado", "Pneumonia"], noisy_or([0.80, 0.30, 0.85], 0.02))
    r.add("Tosse", ["Gripe", "Resfriado", "Asma", "Bronquite", "Pneumonia"],
          noisy_or([0.60, 0.60, 0.50, 0.80, 0.80], 0.05))
    r.add("Dispneia", ["Asma", "Bronquite", "Pneumonia"], noisy_or([0.70, 0.50, 0.75], 0.01))
    r.add("Dor_corpo", ["Gripe", "Resfriado", "Pneumonia"], noisy_or([0.70, 0.10, 0.40], 0.05))
    r.add("Coriza", ["Resfriado", "Gripe"], noisy_or([0.80, 0.40], 0.05))
    r.add("Chiado", ["Asma", "Bronquite"], noisy_or([0.80, 0.40], 0.01))
    r.add("RaioX", ["Pneumonia", "Bronquite"], noisy_or([0.90, 0.15], 0.02))
    return r


def rede_aleatoria(n, seed):
    """Rede noisy-OR com n nos; pais escolhidos numa janela dos 5 nos anteriores
    (treewidth limitada), ate 3 pais. Usada so no experimento de escala."""
    rng = np.random.default_rng(seed)
    r = Rede()
    for i in range(n):
        janela = list(range(max(0, i - 5), i))
        k = int(rng.integers(0, 4)) if janela else 0
        k = min(k, len(janela))
        pais = sorted(rng.choice(janela, size=k, replace=False).tolist()) if k else []
        nome = f"X{i}"
        if k == 0:
            r.add(nome, [], np.array([0.7, 0.3]))
        else:
            r.add(nome, [f"X{j}" for j in pais], noisy_or(rng.uniform(0.3, 0.9, size=k), 0.05))
    return r


# ----------------------------------------------------------------------------
# Verdade de referencia: conjunta por forca bruta
# ----------------------------------------------------------------------------
def conjunta(rede):
    f = None
    for v in rede.order:
        f = rede.fator(v) if f is None else multiplicar(f, rede.fator(v))
    return f


def posteriori_bruta(rede, J, consulta, evid):
    f = J
    for v, val in evid.items():
        f = restringir(f, v, val)
    for v in [u for u in f.vars if u != consulta]:
        f = somar_fora(f, v)
    return f.t / f.t.sum()


# ----------------------------------------------------------------------------
# Inferencia exata 1: enumeracao
# ----------------------------------------------------------------------------
def enumeracao(rede, consulta, evid, stats=None):
    stats = stats if stats is not None else {}
    stats["chamadas"] = 0

    def enum_all(vars_, e):
        if not vars_:
            return 1.0
        stats["chamadas"] += 1
        y, resto = vars_[0], vars_[1:]
        if y in e:
            return rede.p(y, e) * enum_all(resto, e)
        tot = 0.0
        for val in (0, 1):
            e2 = dict(e)
            e2[y] = val
            tot += rede.p(y, e2) * enum_all(resto, e2)
        return tot

    q = np.array([enum_all(rede.order, {**evid, consulta: val}) for val in (0, 1)])
    return q / q.sum()


# ----------------------------------------------------------------------------
# Inferencia exata 2: eliminacao de variaveis
# ----------------------------------------------------------------------------
def _escolhe(fatores, ocultas, heuristica, rng):
    if heuristica == "aleatoria":
        return ocultas[int(rng.integers(len(ocultas)))]
    adj = {}
    for f in fatores:
        for a in f.vars:
            adj.setdefault(a, set()).update(f.vars)
    melhor, custo_melhor = None, None
    for v in ocultas:
        viz = adj.get(v, set()) - {v}
        if heuristica == "min-size":
            c = len(viz)
        else:  # min-fill: pares de vizinhos ainda nao ligados
            c = sum(1 for a, b in itertools.combinations(viz, 2) if b not in adj.get(a, set()))
            c = (c, len(viz))
        if custo_melhor is None or c < custo_melhor:
            melhor, custo_melhor = v, c
    return melhor


def eliminacao(rede, consulta, evid, heuristica="min-fill", poda=True, stats=None, rng=None):
    """Eliminacao de variaveis. `stats` recebe tam_max (maior fator criado, em
    entradas) e mults (soma dos tamanhos das tabelas-produto)."""
    stats = stats if stats is not None else {}
    stats["tam_max"], stats["mults"] = 0, 0
    manter = ancestrais(rede, [consulta, *evid]) if poda else set(rede.order)
    fatores = []
    for v in rede.order:
        if v not in manter:
            continue
        f = rede.fator(v)
        for e, val in evid.items():
            f = restringir(f, e, val)
        fatores.append(f)
    ocultas = [v for v in rede.order if v in manter and v != consulta and v not in evid]
    while ocultas:
        v = _escolhe(fatores, ocultas, heuristica, rng)
        ocultas.remove(v)
        com = [f for f in fatores if v in f.vars]
        fatores = [f for f in fatores if v not in f.vars]
        if not com:
            continue
        prod = com[0]
        for f in com[1:]:
            prod = multiplicar(prod, f)
            stats["mults"] += prod.t.size
        stats["tam_max"] = max(stats["tam_max"], prod.t.size)
        fatores.append(somar_fora(prod, v))
    res = fatores[0]
    for f in fatores[1:]:
        res = multiplicar(res, f)
    assert res.vars == [consulta]
    return res.t / res.t.sum()


# ----------------------------------------------------------------------------
# Inferencia aproximada
# ----------------------------------------------------------------------------
def amostra_direta(rede, N, rng):
    X = {}
    for v in rede.order:
        pa = rede.pais[v]
        if pa:
            p1 = rede.cpt[v][tuple(X[p] for p in pa)][:, 1]
        else:
            p1 = np.full(N, rede.cpt[v][1])
        X[v] = (rng.random(N) < p1).astype(np.int8)
    return X


def est_rejeicao(rede, consulta, evid, N, rng):
    """Retorna (prefixo de estimativas em N, aceitas)."""
    X = amostra_direta(rede, N, rng)
    ok = np.ones(N, bool)
    for v, val in evid.items():
        ok &= X[v] == val
    return X[consulta], ok


def est_ponderacao(rede, consulta, evid, N, rng):
    """Likelihood weighting: evidencia fixa, peso = produto de P(e|pais)."""
    X, w = {}, np.ones(N)
    for v in rede.order:
        pa = rede.pais[v]
        p1 = rede.cpt[v][tuple(X[p] for p in pa)][:, 1] if pa else np.full(N, rede.cpt[v][1])
        if v in evid:
            X[v] = np.full(N, evid[v], dtype=np.int8)
            w *= p1 if evid[v] == 1 else 1 - p1
        else:
            X[v] = (rng.random(N) < p1).astype(np.int8)
    return X[consulta], w


def gibbs(rede, consulta, evid, n_varreduras, rng, queima=200):
    """Gibbs: cada variavel oculta e reamostrada dado seu manto de Markov
    P(x | pais) * prod_filhos P(filho | pais do filho). Estimador Rao-Blackwell:
    media da condicional de `consulta` (menos variancia que contar amostras).
    Retorna o vetor de condicionais P(consulta=1 | manto) por varredura pos-queima."""
    estado = {v: int(rng.integers(2)) for v in rede.order}
    estado.update(evid)
    ocultas = [v for v in rede.order if v not in evid]

    def cond(v):
        pr = np.empty(2)
        for val in (0, 1):
            estado[v] = val
            p = rede.p(v, estado)
            for c in rede.filhos[v]:
                p *= rede.p(c, estado)
            pr[val] = p
        return pr[1] / pr.sum()

    saida = []
    for it in range(queima + n_varreduras):
        for v in ocultas:
            estado[v] = int(rng.random() < cond(v))
        if it >= queima:
            saida.append(cond(consulta))
    return np.array(saida)


# ----------------------------------------------------------------------------
# Aprendizado de parametros por contagem
# ----------------------------------------------------------------------------
def aprende(rede_ref, dados, alfa):
    """Mesma estrutura de rede_ref; CPT = (contagem + alfa) / (total + 2 alfa)."""
    nova = Rede()
    for v in rede_ref.order:
        pa = rede_ref.pais[v]
        cont = np.zeros((2,) * len(pa) + (2,))
        idx = tuple(dados[p] for p in pa) + (dados[v],)
        np.add.at(cont, idx, 1)
        cont = cont + alfa
        nova.add(v, pa, cont / cont.sum(axis=-1, keepdims=True))
    return nova


def kl(P, Q):
    p, q = P.t.ravel(), Q.t.ravel()
    m = p > 0
    if np.any(q[m] == 0):
        return np.inf
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


# ----------------------------------------------------------------------------
# Desenho da rede
# ----------------------------------------------------------------------------
POS = {
    "Fumante": (1, 2), "Poluicao": (4, 2), "Contato": (7.5, 2),
    "Gripe": (6.5, 1), "Resfriado": (8.5, 1), "Asma": (3.5, 1), "Bronquite": (2, 1), "Pneumonia": (0, 1),
    "Febre": (0, 0), "Tosse": (1.6, 0), "Dispneia": (3.2, 0), "Dor_corpo": (4.8, 0),
    "Coriza": (6.4, 0), "Chiado": (8.0, 0), "RaioX": (9.6, 0),
}


def desenha_rede(rede, post, evid, titulo, arq):
    fig, ax = plt.subplots(figsize=(13, 5.4))
    for v in rede.order:
        for p in rede.pais[v]:
            (x0, y0), (x1, y1) = POS[p], POS[v]
            ax.annotate("", xy=(x1, y1 + 0.13), xytext=(x0, y0 - 0.13),
                        arrowprops=dict(arrowstyle="->", color="#888", lw=0.9, shrinkA=6, shrinkB=6))
    cmap = plt.get_cmap("YlOrRd")
    for v in rede.order:
        x, y = POS[v]
        p1 = post[v]
        ev = v in evid
        ax.scatter([x], [y], s=2300, c=[cmap(p1)], edgecolors="black" if ev else "#555",
                   linewidths=3.0 if ev else 0.8, zorder=3)
        ax.text(x, y + 0.02, f"{v}\n{p1:.2f}" + ("*" if ev else ""), ha="center", va="center", fontsize=7.5, zorder=4)
    ax.set_xlim(-1, 10.6)
    ax.set_ylim(-0.6, 2.5)
    ax.axis("off")
    ax.set_title(titulo, fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, label="P(variavel = 1 | evidencia)")
    fig.tight_layout()
    fig.savefig(FIGS / arq, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    rede = rede_clinica()
    n_nos = len(rede.order)

    # ---- CPTs validas ------------------------------------------------------
    for v in rede.order:
        assert np.allclose(rede.cpt[v].sum(axis=-1), 1.0), v
    J = conjunta(rede)
    assert abs(J.t.sum() - 1) < 1e-12 and J.t.size == 2 ** n_nos
    cheio = sum(2 ** len(rede.pais[v]) for v in rede.order)
    _saida("=== REDE ===")
    _saida(f"nos={n_nos} arestas={sum(len(p) for p in rede.pais.values())} "
           f"parametros noisy-OR={sum(len(rede.pais[v]) + 1 for v in rede.order if rede.pais[v]) + 3} "
           f"(tabelas cheias: {cheio} linhas, conjunta sem estrutura: {2 ** n_nos - 1})")

    # ---- Cenario clinico ----------------------------------------------------
    evid = {"Fumante": 1, "Febre": 1, "Tosse": 1, "Dispneia": 1, "RaioX": 1, "Coriza": 0}
    prior = {v: posteriori_bruta(rede, J, v, {})[1] for v in rede.order}
    post_exato = {}
    for v in rede.order:
        if v in evid:
            post_exato[v] = float(evid[v])
            continue
        e1 = eliminacao(rede, v, evid)
        e2 = enumeracao(rede, v, evid)
        e3 = posteriori_bruta(rede, J, v, evid)
        assert np.allclose(e1, e3, atol=1e-10) and np.allclose(e2, e3, atol=1e-10), v
        post_exato[v] = e3[1]
    _saida("\n=== CENARIO: fumante, febre, tosse, dispneia, raio-X anormal, sem coriza ===")
    _saida(f"{'doenca':<11}{'a priori':>10}{'a posteriori':>14}")
    for v in ["Gripe", "Resfriado", "Asma", "Bronquite", "Pneumonia"]:
        _saida(f"{v:<11}{prior[v]:>10.4f}{post_exato[v]:>14.4f}")
    pe = float(np.sum(restringir_total(J, evid)))
    _saida(f"P(evidencia) = {pe:.5f}   (enumeracao == eliminacao == forca bruta nas 15 marginais: OK)")
    desenha_rede(rede, post_exato, evid,
                 "Posteriori exata (eliminacao de variaveis); cor = P(var=1); * = evidencia observada",
                 "01_rede_posteriori.png")

    # ---- Explaining away ----------------------------------------------------
    _saida("\n=== EXPLAINING AWAY: P(Pneumonia | ...) ===")
    casos = [
        ("nada observado", {}),
        ("Febre", {"Febre": 1}),
        ("Febre + Gripe confirmada", {"Febre": 1, "Gripe": 1}),
        ("Febre + Gripe confirmada + RaioX anormal", {"Febre": 1, "Gripe": 1, "RaioX": 1}),
        ("Febre + RaioX anormal", {"Febre": 1, "RaioX": 1}),
        ("Febre + Tosse + Dispneia + RaioX + Fumante", {"Febre": 1, "Tosse": 1, "Dispneia": 1, "RaioX": 1, "Fumante": 1}),
    ]
    ea = []
    for nome, e in casos:
        pr = eliminacao(rede, "Pneumonia", e)[1]
        assert abs(pr - posteriori_bruta(rede, J, "Pneumonia", e)[1]) < 1e-10
        ea.append((nome, pr))
        _saida(f"  {nome:<46}{pr:>9.4f}")
    assert ea[2][1] < ea[1][1], "explaining away: Gripe deveria reduzir P(Pneumonia | Febre)"

    # ---- Amostradores -------------------------------------------------------
    Ns = [100, 300, 1000, 3000, 10000, 30000]
    Gs = [100, 300, 1000, 3000, 10000]
    SEEDS = 20
    AM = {}
    for consulta in ("Pneumonia", "Bronquite"):
        ref = float(eliminacao(rede, consulta, evid)[1])
        _saida(f"\n=== AMOSTRADORES (consulta P({consulta}=1 | evidencia) = {ref:.4f}; P(evidencia)={pe:.5f}) ===")
        err = {"rejeicao": np.zeros((SEEDS, len(Ns))), "ponderacao": np.zeros((SEEDS, len(Ns))),
               "gibbs": np.zeros((SEEDS, len(Gs)))}
        aceitas = np.zeros((SEEDS, len(Ns)))
        ess = np.zeros((SEEDS, len(Ns)))
        t_gibbs = 0.0
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 + s)
            xq, ok = est_rejeicao(rede, consulta, evid, max(Ns), rng)
            xw, w = est_ponderacao(rede, consulta, evid, max(Ns), rng)
            for k, n in enumerate(Ns):
                a = ok[:n].sum()
                aceitas[s, k] = a
                est = xq[:n][ok[:n]].mean() if a > 0 else 0.5  # sem amostra aceita: palpite neutro
                err["rejeicao"][s, k] = abs(est - ref)
                ww = w[:n]
                err["ponderacao"][s, k] = abs((ww * xw[:n]).sum() / ww.sum() - ref)
                ess[s, k] = ww.sum() ** 2 / (ww ** 2).sum()
            t0 = time.perf_counter()
            c = gibbs(rede, consulta, evid, max(Gs), rng)
            t_gibbs += time.perf_counter() - t0
            for k, n in enumerate(Gs):
                err["gibbs"][s, k] = abs(c[:n].mean() - ref)
        _saida(f"erro absoluto medio em {SEEDS} sementes (mesma consulta/evidencia):")
        _saida(f"{'N':>7}{'rejeicao':>11}{'(aceitas)':>11}{'ponderacao':>12}{'(ESS)':>8}{'gibbs(varred.)':>16}")
        for k, n in enumerate(Ns):
            g = f"{err['gibbs'][:, Gs.index(n)].mean():.4f}" if n in Gs else "-"
            _saida(f"{n:>7}{err['rejeicao'][:, k].mean():>11.4f}{aceitas[:, k].mean():>11.1f}"
                   f"{err['ponderacao'][:, k].mean():>12.4f}{ess[:, k].mean():>8.0f}{g:>16}")
        _saida(f"Gibbs: {t_gibbs / SEEDS:.2f} s por cadeia de {max(Gs) + 200} varreduras (so tempo; varia)")
        i1000 = Ns.index(1000)
        assert err["ponderacao"][:, i1000].mean() < err["rejeicao"][:, i1000].mean()
        assert err["ponderacao"][:, -1].mean() < 0.01
        assert err["gibbs"][:, -1].mean() < 0.02
        assert err["ponderacao"][:, -1].mean() < err["ponderacao"][:, 0].mean()
        AM[consulta] = (err, aceitas, ess)
    consulta = "Pneumonia"
    ref = float(eliminacao(rede, consulta, evid)[1])
    err, aceitas, ess = AM[consulta]

    # ---- Aprendizado --------------------------------------------------------
    _saida("\n=== APRENDIZADO DE PARAMETROS (estrutura conhecida) ===")
    Nl = [50, 200, 1000, 5000, 20000, 100000]
    SL = 10
    klv = {a: np.zeros((SL, len(Nl))) for a in (0.01, 1.0)}
    err_post = np.zeros((SL, len(Nl)))
    for s in range(SL):
        rng = np.random.default_rng(2000 + s)
        for k, n in enumerate(Nl):
            dados = amostra_direta(rede, n, rng)
            for a in klv:
                aprendida = aprende(rede, dados, a)
                klv[a][s, k] = kl(J, conjunta(aprendida))
                if a == 1.0:
                    err_post[s, k] = abs(eliminacao(aprendida, consulta, evid)[1] - ref)
    _saida(f"{'N':>8}{'KL (a=0.01)':>14}{'KL (a=1)':>12}{'|erro posteriori|':>20}")
    for k, n in enumerate(Nl):
        _saida(f"{n:>8}{np.mean(klv[0.01][:, k]):>14.4f}{np.mean(klv[1.0][:, k]):>12.4f}{err_post[:, k].mean():>20.4f}")
    m1 = klv[1.0].mean(axis=0)
    assert np.all(np.diff(m1) < 0), "KL deveria decrescer com N"
    assert m1[-1] < 0.01
    assert klv[1.0][:, 0].mean() < klv[0.01][:, 0].mean(), "Laplace deve ajudar com poucos dados"

    # ---- Escala: enumeracao x eliminacao -----------------------------------
    _saida("\n=== ESCALA: custo de inferencia exata em redes noisy-OR aleatorias (consulta = ultimo no, 3 evidencias) ===")
    ns = [8, 10, 12, 14, 16, 18]
    ns_ve = [8, 12, 16, 20, 30, 40, 60, 80]
    linhas = {}
    for n in sorted(set(ns + ns_ve)):
        calls, mults, tam, mults_np = [], [], [], []
        for sd in range(5):
            r = rede_aleatoria(n, 50 + sd)
            rng = np.random.default_rng(sd)
            q = r.order[-1]
            cand = r.order[:-1]
            ev = {cand[int(i)]: 1 for i in rng.choice(len(cand), size=3, replace=False)}
            st = {}
            pv = eliminacao(r, q, ev, "min-fill", True, st)
            mults.append(st["mults"])
            tam.append(st["tam_max"])
            st2 = {}
            eliminacao(r, q, ev, "min-fill", False, st2)
            mults_np.append(st2["mults"])
            if n in ns:
                st3 = {}
                pe_ = enumeracao(r, q, ev, st3)
                assert np.allclose(pe_, pv, atol=1e-9)
                calls.append(st3["chamadas"])
        linhas[n] = (np.mean(calls) if calls else np.nan, np.mean(mults), np.mean(mults_np), np.mean(tam), max(tam))
    _saida(f"{'n':>4}{'enum: chamadas':>16}{'VE: mults':>11}{'VE sem poda':>13}{'tam max medio':>15}{'tam max':>9}")
    for n, (c, m, mnp, t, tmax) in linhas.items():
        _saida(f"{n:>4}{'-' if np.isnan(c) else f'{c:.0f}':>16}{m:>11.0f}{mnp:>13.0f}{t:>15.1f}{tmax:>9.0f}")
    assert linhas[18][0] > 20 * linhas[10][0], "enumeracao deveria crescer exponencialmente"
    assert linhas[80][1] < 5 * linhas[40][1] and linhas[80][4] <= 2 ** 8

    # ---- Ablacao da ordem de eliminacao ------------------------------------
    _saida("\n=== ABLACAO: ordem de eliminacao na rede clinica (consulta Pneumonia, evidencia do cenario) ===")
    _saida(f"{'heuristica':<12}{'poda':>6}{'tam max':>9}{'mults':>8}")
    ab = {}
    for h in ("min-fill", "min-size", "aleatoria"):
        for poda in (True, False):
            tms, mts = [], []
            for sd in range(20 if h == "aleatoria" else 1):
                st = {}
                eliminacao(rede, consulta, evid, h, poda, st, np.random.default_rng(sd))
                tms.append(st["tam_max"])
                mts.append(st["mults"])
            ab[(h, poda)] = (np.mean(tms), np.mean(mts))
            _saida(f"{h:<12}{('sim' if poda else 'nao'):>6}{np.mean(tms):>9.1f}{np.mean(mts):>8.0f}")
    assert ab[("min-fill", True)][1] <= ab[("aleatoria", True)][1]
    assert ab[("min-fill", True)][1] <= ab[("min-fill", False)][1]

    # ---- Figuras ------------------------------------------------------------
    # Fig 2: explaining away + prior/posteriori das doencas
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axs[0]
    doencas = ["Gripe", "Resfriado", "Asma", "Bronquite", "Pneumonia"]
    x = np.arange(len(doencas))
    ax.bar(x - 0.2, [prior[d] for d in doencas], 0.4, label="a priori", color="#9ab")
    ax.bar(x + 0.2, [post_exato[d] for d in doencas], 0.4, label="a posteriori (cenario)", color="#c44")
    ax.set_xticks(x, doencas)
    ax.set_ylabel("P(doenca = 1)")
    ax.set_title("Efeito da evidencia sobre cada doenca")
    ax.legend()
    ax = axs[1]
    nomes = [n.replace(" + ", "\n+ ") for n, _ in ea]
    ax.barh(range(len(ea)), [p for _, p in ea], color=["#9ab", "#d96", "#7a7", "#c44", "#c44", "#a22"])
    ax.set_yticks(range(len(ea)), [n if len(n) < 30 else n[:28] + ".." for n, _ in ea], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("P(Pneumonia = 1 | evidencia)")
    ax.set_title("Explaining away: Gripe confirmada 'explica' a febre")
    for i, (_, p) in enumerate(ea):
        ax.text(p + 0.005, i, f"{p:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "01_explaining_away.png", dpi=170)
    plt.close(fig)

    # Fig 3: erro dos amostradores + ESS
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, q in zip(axs[:2], ("Pneumonia", "Bronquite")):
        e_q = AM[q][0]
        for nome, xs, cor in (("rejeicao", Ns, "#d62"), ("ponderacao", Ns, "#26a"), ("gibbs", Gs, "#2a6")):
            m = e_q[nome].mean(axis=0)
            ax.errorbar(xs, np.maximum(m, 1e-5), yerr=e_q[nome].std(axis=0) / np.sqrt(SEEDS), marker="o",
                        color=cor, label=nome, capsize=3)
        ax.loglog(Ns, 0.5 / np.sqrt(Ns), "k:", label=r"$\propto 1/\sqrt{N}$")
        ax.set_xlabel("amostras (rejeicao, ponderacao) ou varreduras (Gibbs)")
        ax.set_ylabel(f"erro absoluto medio de P({q} | e)")
        ax.set_title(f"Convergencia: {q} ({SEEDS} sementes)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    ax = axs[2]
    ax.loglog(Ns, aceitas.mean(axis=0), "o-", color="#d62", label="rejeicao: amostras aceitas")
    ax.loglog(Ns, ess.mean(axis=0), "s-", color="#26a", label="ponderacao: ESS")
    ax.loglog(Ns, Ns, "k:", label="N")
    ax.set_xlabel("N")
    ax.set_ylabel("amostras uteis")
    ax.set_title(f"Eficiencia amostral (P(evidencia) = {pe:.4f})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "01_amostradores.png", dpi=170)
    plt.close(fig)

    # Fig 4: aprendizado + escala
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    for a, cor in ((0.01, "#d62"), (1.0, "#26a")):
        m = klv[a].mean(axis=0)
        ax.errorbar(Nl, m, yerr=klv[a].std(axis=0) / np.sqrt(SL), marker="o", color=cor, capsize=3,
                    label=f"alfa = {a}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("exemplos de treino N")
    ax.set_ylabel(r"$KL(P_{real} \| P_{aprendida})$ (nats)")
    ax.set_title("Aprendizado por contagem: suavizacao importa com poucos dados")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax = axs[1]
    ne = list(ns)
    ax.semilogy(ne, [linhas[n][0] for n in ne], "o-", color="#d62", label="enumeracao (chamadas)")
    ax.semilogy(list(linhas), [linhas[n][1] for n in linhas], "s-", color="#26a", label="elim. variaveis (mults, com poda)")
    ax.semilogy(list(linhas), [linhas[n][2] for n in linhas], "^--", color="#7ab", label="elim. variaveis (sem poda)")
    ax.set_xlabel("numero de nos n")
    ax.set_ylabel("operacoes (contagem)")
    ax.set_title("Custo da inferencia exata x tamanho da rede")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "01_aprendizado_escala.png", dpi=170)
    plt.close(fig)

    _saida("\nTodos os assert passaram.")
    (FIGS / "01_saida.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")


def restringir_total(J, evid):
    f = J
    for v, val in evid.items():
        f = restringir(f, v, val)
    return f.t


if __name__ == "__main__":
    main()
