"""
Projeto 3 - Aprendizado por reforco: uma padaria aprende quanto encomendar de um
produto perecivel-de-estoque com demanda que muda ao longo da semana.

PROBLEMA
    Todo fim de dia a loja olha o estoque (0 a 20 unidades, capacidade da prateleira)
    e decide quantas unidades encomendar; a entrega chega na manha seguinte, antes dos
    clientes. A demanda diaria e Poisson com media que depende do dia da semana
    (seg..dom = 3, 3, 4, 4, 7, 10, 8). Cada venda rende 10, cada unidade comprada custa 4,
    cada pedido tem custo FIXO de 12 (frete), cada unidade que dorme na prateleira custa
    0,5 e cada cliente nao atendido custa 2 (perda de reputacao) alem da venda perdida.
    Encomendar muito paga armazenagem; pouco perde vendas; pedir todo dia paga frete.
    O agente NAO conhece a demanda nem os custos: so ve estoque, dia e lucro do dia.

MODELAGEM (MDP)
    Estado s = (estoque, dia da semana): 21 x 7 = 147 estados. Acao a = unidades
    encomendadas, 0 <= a <= 20 - estoque. Recompensa = lucro do dia. Desconto
    gamma = 0.95 (tarefa continua, uma unica trajetoria longa, sem episodios).
    Como o simulador e nosso, o MDP completo (P(s'|s,a) e R(s,a)) e conhecido e
    serve de VERDADE: a politica otima e calculada exatamente.

ALGORITMOS IMPLEMENTADOS A MAO (numpy)
    - Iteracao de valor e iteracao de politica (exatas, usam o modelo).
    - Q-learning (off-policy), SARSA (on-policy) e Expected SARSA (alvo = media de
      Q(s',.) sob a politica epsilon-gulosa, menos variancia que o SARSA), tabulares,
      epsilon decrescente de 1 a 0.01, passo alfa constante (ablacao) ou 1/n(s,a)^0.7.
      1 milhao de dias reais por execucao, 10 sementes em processos paralelos.
    - Dyna-Q (conceito alem da aula): cada passo real e seguido de n atualizacoes
      "imaginadas" com transicoes reamostradas da memoria do agente (modelo amostral).
    - R-MAX (Brafman e Tennenholtz, 2002), aprendizado BASEADO EM MODELO com exploracao
      otimista: conta transicoes, trata pares (s,a) vistos menos de m vezes como
      valendo o maximo possivel e replaneja por iteracao de valor no MDP estimado.
    - Linha de base classica de pesquisa operacional: politica (s, S) unica (pede ate
      S quando o estoque cai abaixo de s), com o melhor par achado por busca exaustiva.
    - Toda politica e avaliada EXATAMENTE (sistema linear de Bellman e distribuicao
      estacionaria da cadeia), sem ruido de simulacao.

VERDADE DE REFERENCIA / AUTOVERIFICACAO
    - Iteracao de valor == iteracao de politica (mesmo V*, mesma politica).
    - Linhas de P somam 1; R(s,a) == media de Monte Carlo do simulador (6 sigma).
    - V* satisfaz Bellman (residuo < 1e-8) e domina toda politica avaliada
      (V^pi <= V* em todos os estados).
    - Lucro medio estacionario exato da politica otima == media de uma simulacao longa.
    - Estatisticos (10 sementes pareadas, mesmas demandas): ver os assert no main()
      (alfa decrescente x constante, Dyna-Q e R-MAX x Q-learning com o mesmo numero
      de dias reais, todo agente acima da politica aleatoria).

Execucao: python 03_reforco_estoque.py  (gera figs/03_*.png e figs/03_saida.txt)
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

PASTA = Path(__file__).resolve().parent
FIGS = PASTA / "figs"
FIGS.mkdir(exist_ok=True)

_LOG = []


def _saida(*args):
    """Imprime a linha e a acumula em _LOG para gravar em figs/03_saida.txt."""
    linha = " ".join(str(a) for a in args)
    print(linha)
    _LOG.append(linha)


# ----------------------------------------------------------------------------
# Ambiente
# ----------------------------------------------------------------------------
CAP = 20
LAM = np.array([3, 3, 4, 4, 7, 10, 8], float)  # demanda media seg..dom
DIAS = ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]
PRECO, CUSTO, FRETE, ARMAZ, MULTA = 10.0, 4.0, 12.0, 0.5, 2.0
GAMMA = 0.95
NS, NA = (CAP + 1) * 7, CAP + 1
DMAX = 45  # truncamento da Poisson (massa da cauda somada no ultimo valor)


def idx(estoque, dia):
    return estoque * 7 + dia


def lucro(estoque, a, d):
    """Lucro do dia: estoque + a disponivel, demanda d."""
    disp = estoque + a
    vend = min(d, disp)
    sobra = disp - vend
    return PRECO * vend - CUSTO * a - FRETE * (a > 0) - ARMAZ * sobra - MULTA * (d - vend), sobra


def poisson_pmf(lam):
    p = np.zeros(DMAX + 1)
    p[0] = np.exp(-lam)
    for k in range(1, DMAX + 1):
        p[k] = p[k - 1] * lam / k
    p[-1] += 1 - p.sum()
    return p


def modelo():
    """MDP exato: P[s, a, s'], R[s, a] e mascara de acoes validas."""
    P = np.zeros((NS, NA, NS))
    R = np.full((NS, NA), -1e9)
    valida = np.zeros((NS, NA), bool)
    for dia in range(7):
        pmf = poisson_pmf(LAM[dia])
        for e in range(CAP + 1):
            s = idx(e, dia)
            for a in range(CAP - e + 1):
                valida[s, a] = True
                R[s, a] = 0.0
                for d in range(DMAX + 1):
                    r, sobra = lucro(e, a, d)
                    R[s, a] += pmf[d] * r
                    P[s, a, idx(sobra, (dia + 1) % 7)] += pmf[d]
    return P, R, valida


def iteracao_valor(P, R, valida, tol=1e-10):
    V = np.zeros(NS)
    while True:
        Q = np.where(valida, R + GAMMA * P @ V, -np.inf)
        Vn = Q.max(1)
        if np.abs(Vn - V).max() < tol * (1 - GAMMA) / GAMMA:
            return Vn, Q.argmax(1), Q
        V = Vn


def avaliar(P, R, pi):
    """V^pi exato: resolve (I - gamma P_pi) V = R_pi."""
    Pp = P[np.arange(NS), pi]
    return np.linalg.solve(np.eye(NS) - GAMMA * Pp, R[np.arange(NS), pi])


def iteracao_politica(P, R, valida):
    pi = np.zeros(NS, int)
    while True:
        V = avaliar(P, R, pi)
        Q = np.where(valida, R + GAMMA * P @ V, -np.inf)
        novo = Q.argmax(1)
        melhora = Q[np.arange(NS), novo] > Q[np.arange(NS), pi] + 1e-9
        if not melhora.any():
            return V, pi
        pi = np.where(melhora, novo, pi)


def lucro_medio(P, R, pi):
    """Lucro medio por dia no regime estacionario da cadeia induzida por pi (exato)."""
    Pp = P[np.arange(NS), pi]
    w, v = np.linalg.eig(Pp.T)
    mu = np.real(v[:, np.argmin(np.abs(w - 1))])
    mu = mu / mu.sum()
    return mu @ R[np.arange(NS), pi]


def politica_sS(s_, S_):
    pi = np.zeros(NS, int)
    for e in range(CAP + 1):
        for dia in range(7):
            pi[idx(e, dia)] = S_ - e if e < s_ else 0
    return pi


# ----------------------------------------------------------------------------
# Agentes tabulares
# ----------------------------------------------------------------------------
def treinar(metodo, passos, semente, valida, alfa="const", n_plan=0, eps_fim=0.01, avaliar_cada=None, P=None, R=None, Vstar=None):
    """Uma trajetoria continua de `passos` dias reais. Devolve Q e a curva (passos, gap %) se pedido.

    metodo: "q" (Q-learning), "sarsa" ou "esarsa" (Expected SARSA: alvo = media de Q(s', .)
    sob a politica epsilon-gulosa). epsilon decai linearmente de 1 a eps_fim na 1a metade
    (GLIE na pratica quando eps_fim e pequeno). alfa: 0.1 constante ou 1/n(s,a)^0.7."""
    rng = np.random.default_rng(semente)
    dem = rng.poisson(LAM[np.arange(passos) % 7])
    u_eps = rng.random(passos + 1)
    a_rand = rng.random(passos + 1)
    Q = np.where(valida, 0.0, -np.inf)
    N = np.zeros((NS, NA))
    nval = valida.sum(1)
    mem_s, mem_a, mem_s2 = (np.zeros(passos, int) for _ in range(3))
    mem_r = np.zeros(passos)
    curva = []
    e, dia = 0, 0
    s = idx(e, dia)

    def eps_t(t):
        return max(eps_fim, 1.0 - t / (0.5 * passos))   # decai linearmente na 1a metade

    def escolhe(s, t):
        if u_eps[t] < eps_t(t):
            return int(a_rand[t] * nval[s])               # acoes validas sao 0..CAP-estoque
        return int(Q[s].argmax())

    a = escolhe(s, 0)
    for t in range(passos):
        r, e = lucro(e, a, dem[t])
        dia = (dia + 1) % 7
        s2 = idx(e, dia)
        a2 = escolhe(s2, t + 1)
        N[s, a] += 1
        lr = 0.1 if alfa == "const" else 1.0 / N[s, a] ** 0.7
        if metodo == "sarsa":
            boot = Q[s2, a2]
        elif metodo == "esarsa":
            ep = eps_t(t + 1)
            qv = Q[s2, :nval[s2]]
            boot = (1 - ep) * qv.max() + ep * qv.mean()
        else:
            boot = Q[s2].max()
        Q[s, a] += lr * (r + GAMMA * boot - Q[s, a])
        if n_plan:
            mem_s[t], mem_a[t], mem_r[t], mem_s2[t] = s, a, r, s2
            for k in rng.integers(0, t + 1, n_plan):     # planejamento com transicoes lembradas
                ps, pa = mem_s[k], mem_a[k]
                Q[ps, pa] += 0.1 * (mem_r[k] + GAMMA * Q[mem_s2[k]].max() - Q[ps, pa])
        s, a = s2, a2
        if avaliar_cada and (t + 1) % avaliar_cada == 0:
            curva.append((t + 1, gap(P, R, Q.argmax(1), Vstar)))
    return Q, curva


def rmax(passos, semente, valida, m=5, replan=250, avaliar_cada=None, P=None, R=None, Vstar=None):
    """Agente baseado em modelo com exploracao otimista R-MAX (Brafman e Tennenholtz, 2002).

    Conta transicoes N(s,a,s') e soma recompensas; pares (s,a) vistos menos de m vezes sao
    'desconhecidos' e recebem o valor otimista VMAX = RMAX/(1-gamma), o que os torna
    atraentes ate serem experimentados m vezes. A cada `replan` dias o agente resolve o
    MDP estimado por iteracao de valor e passa a agir de forma gulosa (sem epsilon).
    Usa as MESMAS demandas que treinar() com a mesma semente."""
    rng = np.random.default_rng(semente)
    dem = rng.poisson(LAM[np.arange(passos) % 7])
    RMAX = PRECO * CAP
    VMAX = RMAX / (1 - GAMMA)
    Nsas = np.zeros((NS, NA, NS))
    Rsum = np.zeros((NS, NA))
    V = np.zeros(NS)
    curva = []

    def planejar():
        n = Nsas.sum(2)
        conhecido = n >= m
        Ph = Nsas / np.maximum(n, 1)[:, :, None]
        Rh = Rsum / np.maximum(n, 1)
        Vp = np.full(NS, VMAX)
        while True:
            Q = np.where(conhecido, Rh + GAMMA * Ph @ Vp, VMAX)
            Q = np.where(valida, Q, -np.inf)
            Vn = Q.max(1)
            if np.abs(Vn - Vp).max() < 1e-6:
                return Q
            Vp = Vn

    Q = planejar()
    e, dia = 0, 0
    for t in range(passos):
        s = idx(e, dia)
        a = int(Q[s].argmax())
        r, e = lucro(e, a, dem[t])
        dia = (dia + 1) % 7
        Nsas[s, a, idx(e, dia)] += 1
        Rsum[s, a] += r
        if (t + 1) % replan == 0:
            Q = planejar()
        if avaliar_cada and (t + 1) % avaliar_cada == 0:
            pi = planejar_guloso(Nsas, Rsum, valida)
            curva.append((t + 1, gap(P, R, pi, Vstar)))
    return planejar_guloso(Nsas, Rsum, valida), curva


def planejar_guloso(Nsas, Rsum, valida):
    """Politica 'certeza-equivalente' do modelo estimado (sem otimismo): e a politica entregue.
    Pares nunca vistos ficam com Q = -inf (nao sao escolhidos se houver alternativa vista)."""
    n = Nsas.sum(2)
    Ph = Nsas / np.maximum(n, 1)[:, :, None]
    Rh = Rsum / np.maximum(n, 1)
    vis = valida & (n > 0)
    V = np.zeros(NS)
    for _ in range(2000):
        Q = np.where(vis, Rh + GAMMA * Ph @ V, -np.inf)
        Q[~vis.any(1), 0] = 0.0
        Vn = Q.max(1)
        if np.abs(Vn - V).max() < 1e-8:
            break
        V = Vn
    return Q.argmax(1)


def _job(args):
    """Executado num processo filho (mesmo resultado que em serie: a semente fixa tudo)."""
    tipo, nome, semente, kw, P, R, Vstar, valida = args
    if tipo == "rmax":
        pi, c = rmax(semente=semente, valida=valida, P=P, R=R, Vstar=Vstar, **kw)
    else:
        Q, c = treinar(semente=semente, valida=valida, P=P, R=R, Vstar=Vstar, **kw)
        pi = Q.argmax(1)
    return nome, semente, pi, c


def gap(P, R, pi, Vstar):
    """Perda media de valor descontado em % de V*: media_s (V* - V^pi) / media_s V*."""
    return 100 * np.mean(Vstar - avaliar(P, R, pi)) / np.mean(Vstar)


def acerto(pi, Qstar, valida):
    """% de estados cuja acao escolhida e otima (empates ate 1e-6 contam)."""
    return 100 * np.mean(Qstar[np.arange(NS), pi] >= Qstar.max(1) - 1e-6)


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    t_ini = time.perf_counter()
    P, R, valida = modelo()
    _saida(f"MDP: {NS} estados (estoque 0..{CAP} x 7 dias), ate {NA} acoes, {int(valida.sum())} pares (s,a) validos, gamma = {GAMMA}")

    # --- verificacoes do modelo e do otimo ---
    assert np.allclose(P[valida].sum(-1), 1)
    rng = np.random.default_rng(123)
    for (e, dia, a) in ((0, 5, 12), (6, 0, 0), (3, 4, 9)):
        ds = rng.poisson(LAM[dia], 200000)
        rs = np.array([lucro(e, a, d)[0] for d in ds])
        assert abs(rs.mean() - R[idx(e, dia), a]) < 6 * rs.std() / np.sqrt(len(rs))
    Vv, piv, Qstar = iteracao_valor(P, R, valida)
    Vp, pip = iteracao_politica(P, R, valida)
    _saida(f"Iteracao de valor x iteracao de politica: max |V_vi - V_pi| = {np.abs(Vv - Vp).max():.2e}, politicas iguais em {np.mean(piv == pip) * 100:.0f}% dos estados")
    assert np.abs(Vv - Vp).max() < 1e-6 and acerto(pip, Qstar, valida) == 100
    Vstar = Vp
    res = np.abs(np.where(valida, R + GAMMA * P @ Vstar, -np.inf).max(1) - Vstar).max()
    assert res < 1e-8
    _saida(f"Residuo de Bellman de V*: {res:.1e}; P estocastica e R == Monte Carlo (200 mil amostras, 3 pares): OK")
    g_ot = lucro_medio(P, R, pip)
    ds = np.random.default_rng(7).poisson(LAM[np.arange(700000) % 7])
    e, tot = 0, 0.0
    for t, d in enumerate(ds):
        r, e = lucro(e, pip[idx(e, t % 7)], d)
        tot += r
    _saida(f"Lucro medio/dia da politica otima: exato (estacionario) = {g_ot:.3f}; simulacao de 700 mil dias = {tot / len(ds):.3f}")
    assert abs(g_ot - tot / len(ds)) < 0.1

    # politica otima: nivel "pede ate" por dia
    _saida("\nPolitica otima por dia: pede quando estoque < s_d, ate o nivel S_d (estrutura (s,S) verificada):")
    estrut = True
    for dia in range(7):
        acoes = [int(pip[idx(e, dia)]) for e in range(CAP + 1)]
        pede = [e for e in range(CAP + 1) if acoes[e] > 0]
        s_d = max(pede) + 1 if pede else 0
        alvos = {e + acoes[e] for e in pede}
        estrut &= pede == list(range(s_d)) and len(alvos) == 1
        _saida(f"  {DIAS[dia]} (demanda media {LAM[dia]:.0f}): s = {s_d:2d}, S = {sorted(alvos)}  (acao por estoque 0..: {acoes[:12]}...)")
    _saida(f"Estrutura (s,S) por dia (teorema de Scarf): {'confirmada' if estrut else 'NAO confirmada'}")

    # --- linhas de base ---
    melhor = max(((s_, S_) for S_ in range(1, CAP + 1) for s_ in range(0, S_ + 1)), key=lambda p: lucro_medio(P, R, politica_sS(*p)))
    pi_sS = politica_sS(*melhor)
    pi_nunca = np.zeros(NS, int)
    rng_a = np.random.default_rng(0)
    pi_aleat = np.array([rng_a.integers(0, valida[s].sum()) for s in range(NS)])
    pi_diario = np.array([max(0, int(LAM[(s % 7 + 1) % 7]) + 2 - s // 7) for s in range(NS)])  # repoe a media de amanha + 2
    _saida("\nPoliticas de referencia (avaliacao exata):")
    _saida(f"{'politica':<38}{'lucro/dia':>11}{'perda V* %':>12}{'acao otima %':>14}")
    base = {}
    for nome, pi in (("otima (iteracao de politica)", pip), (f"(s,S) unica s={melhor[0]}, S={melhor[1]}", pi_sS), ("repoe media de amanha + 2", pi_diario), ("aleatoria (fixa)", pi_aleat), ("nunca pede", pi_nunca)):
        V = avaliar(P, R, pi)
        assert np.all(V <= Vstar + 1e-6), "nenhuma politica pode superar V*"
        base[nome] = (lucro_medio(P, R, pi), gap(P, R, pi, Vstar), acerto(pi, Qstar, valida))
        _saida(f"{nome:<38}{base[nome][0]:>11.3f}{base[nome][1]:>12.2f}{base[nome][2]:>14.1f}")

    # --- aprendizado sem modelo: 1 milhao de dias, 10 sementes, em paralelo ---
    from concurrent.futures import ProcessPoolExecutor
    T = 1_000_000
    cada = 25000
    sementes = range(10)
    config = {
        "Q-learning (alfa 0.1)": dict(metodo="q"),
        "Q-learning (alfa 1/n^0.7)": dict(metodo="q", alfa="visita"),
        "SARSA (alfa 1/n^0.7)": dict(metodo="sarsa", alfa="visita"),
        "Expected SARSA (alfa 1/n^0.7)": dict(metodo="esarsa", alfa="visita"),
        "Q-learning eps fixo 0.3": dict(metodo="q", alfa="visita", eps_fim=0.3),
    }
    T_d, cada_d = 20000, 1000
    config_d = {
        "Q-learning": ("q", dict(metodo="q", n_plan=0)),
        "Dyna-Q n=5": ("q", dict(metodo="q", n_plan=5)),
        "Dyna-Q n=20": ("q", dict(metodo="q", n_plan=20)),
        "Dyna-Q n=50": ("q", dict(metodo="q", n_plan=50)),
        "R-MAX (modelo, m=5)": ("rmax", dict(m=5)),
        "R-MAX (modelo, m=20)": ("rmax", dict(m=20)),
    }
    jobs = [("q", n, s_, dict(passos=T, avaliar_cada=cada, **cfg), P, R, Vstar, valida) for n, cfg in config.items() for s_ in sementes]
    jobs += [(tp, "D:" + n, s_, dict(passos=T_d, avaliar_cada=cada_d, **cfg), P, R, Vstar, valida) for n, (tp, cfg) in config_d.items() for s_ in sementes]
    with ProcessPoolExecutor(4) as ex:
        saidas = list(ex.map(_job, jobs))
    pis_por, curva_por = {}, {}
    for nome, s_, pi, c in saidas:
        pis_por.setdefault(nome, [None] * 10)[s_] = pi
        curva_por.setdefault(nome, [None] * 10)[s_] = [g for _, g in c]

    def resumo(nome):
        pis = pis_por[nome]
        return {"lucro": [lucro_medio(P, R, p) for p in pis], "gap": [gap(P, R, p, Vstar) for p in pis], "ac": [acerto(p, Qstar, valida) for p in pis]}

    _saida(f"\nAprendizado sem modelo: {T} dias reais por execucao, 10 sementes (mesmas demandas para todos), epsilon 1 -> 0.01 na 1a metade, politica gulosa final:")
    _saida(f"{'metodo':<32}{'lucro/dia':>18}{'perda V* %':>18}{'acao otima %':>18}")
    finais, curvas = {}, {}
    for nome in config:
        finais[nome] = f = resumo(nome)
        curvas[nome] = np.array(curva_por[nome])
        _saida(f"{nome:<32}" + "".join(f"{np.mean(f[k]):>10.3f} +-{np.std(f[k], ddof=1):>6.3f}" for k in ("lucro", "gap", "ac")))
    pi_q = pis_por["Q-learning (alfa 1/n^0.7)"][0]
    assert min(min(f["lucro"]) for f in finais.values()) > base["aleatoria (fixa)"][0]
    d_al = np.array(finais["Q-learning (alfa 1/n^0.7)"]["gap"]) - np.array(finais["Q-learning (alfa 0.1)"]["gap"])
    d_qs = np.array(finais["SARSA (alfa 1/n^0.7)"]["gap"]) - np.array(finais["Q-learning (alfa 1/n^0.7)"]["gap"])
    d_es = np.array(finais["Expected SARSA (alfa 1/n^0.7)"]["gap"]) - np.array(finais["SARSA (alfa 1/n^0.7)"]["gap"])
    _saida(f"Pareado: perda(alfa 1/n^0.7) - perda(alfa 0.1) = {d_al.mean():.2f} p.p. ({np.sum(d_al < 0)}/10 sementes melhores)")
    _saida(f"Pareado: perda(SARSA) - perda(Q-learning) = {d_qs.mean():.2f} p.p. ({np.sum(d_qs > 0)}/10 com SARSA pior); perda(Expected SARSA) - perda(SARSA) = {d_es.mean():.2f} p.p. ({np.sum(d_es < 0)}/10)")
    assert np.all(d_al < 0), "alfa decrescente deveria reduzir o ruido final em todas as sementes"
    assert np.mean(finais["Q-learning (alfa 1/n^0.7)"]["gap"]) < 2.5, "Q-learning (alfa decrescente) deveria chegar a menos de 2,5% de V*"
    assert d_qs.mean() > 3, "neste problema o SARSA (on-policy) termina pior que o Q-learning"

    # --- eficiencia de amostra: Dyna-Q e R-MAX com poucos dias reais ---
    _saida(f"\nCom so {T_d} dias reais (10 sementes, mesmas demandas): Dyna-Q (modelo amostral) e R-MAX (modelo de contagens + otimismo):")
    _saida(f"{'agente':<22}{'lucro/dia':>18}{'perda V* %':>18}{'acao otima %':>18}")
    dyna, curvas_d = {}, {}
    for nome in config_d:
        dyna[nome] = f = resumo("D:" + nome)
        curvas_d[nome] = np.array(curva_por["D:" + nome])
        _saida(f"{nome:<22}" + "".join(f"{np.mean(f[k]):>10.3f} +-{np.std(f[k], ddof=1):>6.3f}" for k in ("lucro", "gap", "ac")))
    dif = np.array(dyna["Dyna-Q n=20"]["gap"]) - np.array(dyna["Q-learning"]["gap"])
    dif_r = np.array(dyna["R-MAX (modelo, m=5)"]["gap"]) - np.array(dyna["Q-learning"]["gap"])
    _saida(f"Pareado: perda(Dyna n=20) - perda(Q-learning) = {dif.mean():.2f} p.p. (Dyna melhor em {np.sum(dif < 0)}/10); perda(R-MAX) - perda(Q-learning) = {dif_r.mean():.2f} p.p. ({np.sum(dif_r < 0)}/10)")
    assert np.all(dif < 0) and np.all(dif_r < 0), "Dyna-Q n=20 e R-MAX deveriam superar o Q-learning em todas as sementes"
    assert np.mean(dyna["R-MAX (modelo, m=5)"]["gap"]) < np.mean(dyna["Dyna-Q n=20"]["gap"])

    # ------------------------------------------------------------------------
    # Figuras
    # ------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (tit, pi) in zip(axs, (("Otima (iteracao de politica)", pip), ("Q-learning alfa 1/n^0.7, 1 milhao de dias (sem. 0)", pi_q), (f"(s,S) unica s={melhor[0]}, S={melhor[1]}", pi_sS))):
        M = np.array([[pi[idx(e, d)] for d in range(7)] for e in range(CAP + 1)])
        im = ax.imshow(M, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=CAP)
        for e in range(0, CAP + 1):
            for d in range(7):
                if M[e, d]:
                    ax.text(d, e, str(M[e, d]), ha="center", va="center", fontsize=6.5, color="w" if M[e, d] < 12 else "k")
        ax.set_xticks(range(7), DIAS)
        ax.set_xlabel("dia da decisao (entrega chega no dia seguinte)")
        ax.set_ylabel("estoque no fim do dia")
        ax.set_title(f"{tit}\nlucro/dia = {lucro_medio(P, R, pi):.2f}")
    fig.colorbar(im, ax=axs, fraction=0.02, label="unidades encomendadas")
    fig.savefig(FIGS / "03_politicas.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axs[0]
    xs = np.arange(1, T // cada + 1) * cada
    for (nome, c), cor in zip(curvas.items(), ["#d62", "#26a", "#2a6", "#e90", "#a4c"]):
        m, sd = c.mean(0), c.std(0, ddof=1)
        ax.plot(xs, m, color=cor, label=nome)
        ax.fill_between(xs, np.maximum(m - sd, 1e-2), m + sd, color=cor, alpha=0.15)
    ax.axhline(base[f"(s,S) unica s={melhor[0]}, S={melhor[1]}"][1], color="k", ls=":", lw=1, label="melhor (s,S) unica")
    ax.set_yscale("log")
    ax.set_xlabel("dias reais de experiencia")
    ax.set_ylabel("perda de valor vs. otimo (%)")
    ax.set_title("Sem modelo: media +- desvio, 10 sementes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axs[1]
    xs = np.arange(1, T_d // cada_d + 1) * cada_d
    for n, cor in zip(config_d, ["#26a", "#e90", "#a4c", "#2a6", "#d62", "#955"]):
        m, sd = curvas_d[n].mean(0), curvas_d[n].std(0, ddof=1)
        ax.plot(xs, m, color=cor, label=n, lw=2 if n.startswith("R-MAX") else 1.5)
        ax.fill_between(xs, np.maximum(m - sd, 1e-2), m + sd, color=cor, alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("dias reais de experiencia")
    ax.set_ylabel("perda de valor vs. otimo (%)")
    ax.set_title("Poucos dias reais: Dyna-Q e R-MAX usam um modelo")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "03_aprendizado.png", dpi=170)
    plt.close(fig)

    # simulacao de 6 semanas com a mesma demanda para todas as politicas
    dsim = np.random.default_rng(2026).poisson(LAM[np.arange(42) % 7])
    fig, axs = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True)
    for nome, pi, cor in (("otima", pip, "#2a6"), ("Q-learning", pi_q, "#26a"), (f"(s,S) {melhor}", pi_sS, "#d62")):
        e, est, acum = 0, [], [0.0]
        for t, d in enumerate(dsim):
            a = pi[idx(e, t % 7)]
            est.append(e + a)
            r, e = lucro(e, a, d)
            acum.append(acum[-1] + r)
        axs[0].step(range(42), est, where="mid", color=cor, label=f"{nome}: estoque disponivel")
        axs[1].plot(range(43), acum, color=cor, label=f"{nome}: lucro acumulado = {acum[-1]:.0f}")
    axs[0].bar(range(42), dsim, color="#bbb", alpha=0.6, label="demanda do dia")
    axs[0].set_ylabel("unidades")
    axs[0].set_ylim(0, 27)
    axs[0].legend(fontsize=8, ncol=4, loc="upper center")
    axs[1].set_ylabel("lucro acumulado")
    axs[1].set_xlabel("dia (0 = segunda)")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)
        for k in range(5, 42, 7):
            ax.axvspan(k - 0.5, k + 1.5, color="#fd8", alpha=0.2, lw=0)
    axs[0].set_title("Seis semanas com a mesma demanda (faixas = fim de semana)")
    fig.tight_layout()
    fig.savefig(FIGS / "03_simulacao.png", dpi=170)
    plt.close(fig)

    _saida(f"\nTodos os assert passaram. (tempo total ~ {time.perf_counter() - t_ini:.0f} s)")
    (FIGS / "03_saida.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
