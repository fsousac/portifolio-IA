"""
Projeto 6 - Base de Conhecimento: Campo Minado por inferencia logica.

PROBLEMA
    Jogar Campo Minado (R x C, n minas) descobrindo, por RACIOCINIO, quais
    celulas sao seguras e quais sao minas. O agente so enxerga os numeros das
    celulas ja reveladas; uma jogada em mina perde o jogo.

BASE DE CONHECIMENTO (KB) PROPOSICIONAL EM CNF
    Variavel  : M(r,c) = "a celula (r,c) tem mina"; a variavel do indice i=r*C+c
                e o inteiro i+1 (literal negativo = negacao). Clausula = tupla
                de literais; a KB e a conjuncao das clausulas.
    TELL      : ao revelar a celula i com numero k (sem mina, obviamente):
                  - unidade  not M(i);
                  - "exatamente k das n vizinhas NAO reveladas sao minas",
                    em CNF por enumeracao de combinacoes (n <= 8):
                      * pelo menos k : para todo subconjunto de tamanho n-k+1,
                        uma clausula positiva (alguma dessas e mina);
                      * no maximo k  : para todo subconjunto de tamanho k+1,
                        uma clausula negativa (alguma dessas nao e mina).
    ASK       : KB |= alfa  <=>  KB ^ not(alfa) e INSATISFATIVEL (refutacao).
                  "x e segura" : KB |= not M(x)  <=>  KB ^ M(x)  insatisfativel
                  "x e mina"   : KB |= M(x)      <=>  KB ^ not M(x) insatisfativel
    DPLL      : implementado a mao (propagacao unitaria + literal puro +
                ramificacao). Para ENTAILMENT, resolucao e DPLL sao equivalentes
                em poder: ambos sao refutacionalmente completos para logica
                proposicional (uma refutacao DPLL e uma refutacao por resolucao
                em arvore) e ambos sao exponenciais no pior caso (SAT e
                NP-completo). DPLL e so a forma pratica: nao gera clausulas
                novas, usa memoria linear e explora unit propagation.
    Otimizacoes seguras (nao mudam o que e deduzido):
      - "fatos" (unidades) sao propagados uma vez sobre a KB inteira, e a KB e
        substituida por uma forma equivalente e menor (compactacao);
      - as clausulas restantes sao separadas em COMPONENTES CONEXAS (variaveis
        ligadas por clausulas); como a KB e satisfativel (o tabuleiro real e um
        modelo), KB |= alfa so depende da componente de alfa;
      - cada modelo achado e reaproveitado como TESTEMUNHA: se x foi verdadeira
        em algum modelo, KB nao implica not M(x); se foi falsa, nao implica M(x).
    Orcamento de nos por chamada DPLL (LIMITE_NOS): se estourar, a consulta e
    tratada como "nao deduzido" (continua SONORO, so perde completude); o numero
    de estouros e reportado.

ENCADEAMENTO BARATO (antes do DPLL), so com numeros do tabuleiro
    Cada numero revelado vira uma restricao (U, rem): "exatamente rem minas em U"
    (U = vizinhas desconhecidas, rem = numero - minas ja conhecidas).
      - contagem trivial : rem = 0 -> todo U e seguro; rem = |U| -> todo U e mina;
      - subconjunto      : se U1 e subconjunto proprio de U2, entao U2-U1 tem
                           rem2-rem1 minas; 0 -> seguras, |U2-U1| -> minas.
    Sao regras de inferencia sonoras, mas INCOMPLETAS (nao veem, por exemplo, o
    padrao 1-2-1 misto com varias soma-restricoes); o DPLL fecha a lacuna.

AGENTES (mesmas sementes de tabuleiro para todos: comparacao pareada)
    aleatorio        : so chutes uniformes entre as celulas nao reveladas.
    regras           : encadeamento + chute aleatorio (sem DPLL).
    regras+DPLL      : encadeamento + DPLL + chute aleatorio (ablacao do chute).
    KB completo      : encadeamento + DPLL + chute de MENOR PROBABILIDADE de mina.
    so DPLL          : sem encadeamento (compara custo; deve decidir igual).
    Primeira jogada: o gerador nunca poe mina na celula inicial nem em suas
    vizinhas (jogada segura garantida e abertura por cascata).
    Chute (KB completo): P(mina) por CONTAGEM PONDERADA DE MODELOS em cada
    componente conexa da fronteira: cada modelo com m minas pesa w^m,
    w = rho/(1-rho), rho = minas restantes / celulas desconhecidas (aproxima a
    contagem global de minas). Celulas fora da fronteira recebem
    rho_int = (minas restantes - minas esperadas na fronteira)/n_interior.
    Vale a menor probabilidade. Se a enumeracao passar de LIMITE_MODELOS folhas,
    cai em heuristica local (maior rem/|U| entre os numeros vizinhos).

AUTOVERIFICACAO
    - DPLL contra forca bruta em CNFs aleatorias;
    - deducoes da KB (seguras e minas) contra enumeracao exaustiva da fronteira
      em tabuleiros pequenos (sonoridade E completude);
    - toda deducao dos agentes e conferida contra o tabuleiro real: 0 falsos
      positivos (nunca "segura" numa mina, nunca "mina" numa segura).

Execucao: python 06_base_conhecimento.py > figs/06_saida.txt 2>&1
          (gera figs/06_*.png; progresso em stderr; ~2 min)
"""

import itertools
import random
import sys
import time
from collections import defaultdict, namedtuple
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela (salva apenas arquivos)
import matplotlib.pyplot as plt
import numpy as np

PASTA = Path(__file__).resolve().parent
FIGS = PASTA / "figs"

LIMITE_NOS = 200_000      # orcamento de nos por chamada DPLL
LIMITE_MODELOS = 30_000   # folhas maximas na contagem de modelos (chute)
DENSIDADES = [0.10, 0.15, 0.20, 0.25]


def progresso(msg):
    """Progresso em stderr (nao polui a saida das tabelas)."""
    print(msg, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# DPLL a mao
# ----------------------------------------------------------------------------
class Orcamento(Exception):
    """Levantada quando uma chamada DPLL passa de LIMITE_NOS nos."""


class Contadores:
    """Contadores de custo da inferencia: chamadas DPLL, nos e estouros."""

    def __init__(self):
        self.chamadas = 0   # chamadas de alto nivel a sat()
        self.nos = 0        # nos de busca (total)
        self.atual = 0      # nos da chamada corrente (para o orcamento)
        self.estouros = 0   # consultas DPLL abandonadas por orcamento
        self.reserva = 0    # chutes que cairam na heuristica local (contagem estourou)


def simplifica(clausulas, lit):
    """Aplica o literal 'lit' verdadeiro: remove clausulas satisfeitas e apaga
    -lit das demais. Devolve None se surgir clausula vazia (conflito)."""
    saida, neg = [], -lit
    for c in clausulas:
        if lit in c:
            continue
        if neg in c:
            c2 = tuple(x for x in c if x != neg)
            if not c2:
                return None
            saida.append(c2)
        else:
            saida.append(c)
    return saida


def dpll(clausulas, modelo, ctr):
    """DPLL recursivo. Devolve um modelo PARCIAL (dict var->bool; variaveis
    ausentes sao livres, pois todas as clausulas ja estao satisfeitas) ou None
    se a formula for insatisfativel."""
    ctr.nos += 1
    ctr.atual += 1
    if ctr.atual > LIMITE_NOS:
        raise Orcamento
    modelo = dict(modelo)
    cls = clausulas
    while True:
        mudou = False
        # propagacao unitaria: clausula de 1 literal forca o valor
        unit = next((c[0] for c in cls if len(c) == 1), None)
        if unit is not None:
            modelo[abs(unit)] = unit > 0
            cls = simplifica(cls, unit)
            if cls is None:
                return None
            continue
        if not cls:
            return modelo
        # literal puro: aparece so com uma polaridade -> pode ser satisfeito
        lits = {l for c in cls for l in c}
        puros = [l for l in lits if -l not in lits]
        if puros:
            for l in puros:
                modelo[abs(l)] = l > 0
            ps = set(puros)
            cls = [c for c in cls if not any(x in ps for x in c)]
            mudou = True
        if not mudou:
            break
    if not cls:
        return modelo
    # ramificacao: primeiro literal da menor clausula (falha rapido)
    lit = min(cls, key=len)[0]
    for l in (lit, -lit):
        c2 = simplifica(cls, l)
        if c2 is not None:
            r = dpll(c2, {**modelo, abs(l): l > 0}, ctr)
            if r is not None:
                return r
    return None


def sat(clausulas, ctr):
    """Chamada de alto nivel ao DPLL (conta chamada, zera o orcamento de nos)."""
    ctr.chamadas += 1
    ctr.atual = 0
    return dpll(list(clausulas), {}, ctr)


def contar_modelos(clausulas, vars_, w, limite):
    """Probabilidade de mina de cada variavel de uma componente: enumera os
    modelos por DPLL (sem literal puro) com peso w^(#minas). Variaveis livres
    numa folha entram de forma analitica ((1+w) por variavel livre). Levanta
    Orcamento se passar de 'limite' folhas."""
    soma = dict.fromkeys(vars_, 0.0)
    z = [0.0]
    folhas = [0]

    def rec(cls, atrib):
        atrib = dict(atrib)
        while True:
            unit = next((c[0] for c in cls if len(c) == 1), None)
            if unit is None:
                break
            atrib[abs(unit)] = unit > 0
            cls = simplifica(cls, unit)
            if cls is None:
                return
        if not cls:
            folhas[0] += 1
            if folhas[0] > limite:
                raise Orcamento
            m = sum(atrib.values())
            livres = [v for v in vars_ if v not in atrib]
            f = len(livres)
            base = w ** m
            total = base * (1 + w) ** f
            z[0] += total
            for v, val in atrib.items():
                if val:
                    soma[v] += total
            if f:
                cada = base * w * (1 + w) ** (f - 1)
                for v in livres:
                    soma[v] += cada
            return
        lit = min(cls, key=len)[0]
        for l in (lit, -lit):
            c2 = simplifica(cls, l)
            if c2 is not None:
                rec(c2, {**atrib, abs(l): l > 0})

    rec(list(clausulas), {})
    assert z[0] > 0
    return {v: soma[v] / z[0] for v in vars_}


# ----------------------------------------------------------------------------
# KB do Campo Minado
# ----------------------------------------------------------------------------
class KB:
    """Base de conhecimento em CNF: 'fatos' (unidades ja conhecidas, var->bool)
    mais 'clausulas' (residuais). TELL adiciona conhecimento; deduzir() faz os
    ASKs de todas as celulas da fronteira e devolve (seguras, minas)."""

    def __init__(self, ctr):
        self.ctr = ctr
        self.fatos = {}
        self.clausulas = []
        self.comps = []   # componentes da ultima varredura: [(vars, clausulas)]

    def tell_unidade(self, cel, mina):
        """TELL de um fato: 'cel tem mina' (True) ou 'cel e segura' (False)."""
        assert self.fatos.get(cel + 1, mina) == mina, "KB inconsistente"
        self.fatos[cel + 1] = mina

    def tell_revelada(self, cel, k, desconhecidas):
        """TELL de uma celula revelada com numero k: nao-mina + restricao
        'exatamente k minas' entre as vizinhas nao reveladas (CNF)."""
        self.tell_unidade(cel, False)
        vs = [j + 1 for j in desconhecidas]
        n = len(vs)
        assert n <= 8 and 0 <= k <= n, "restricao impossivel"
        if k > 0:   # pelo menos k: todo grupo de n-k+1 tem alguma mina
            self.clausulas += [tuple(g) for g in itertools.combinations(vs, n - k + 1)]
        if k < n:   # no maximo k: todo grupo de k+1 tem alguma nao-mina
            self.clausulas += [tuple(-x for x in g) for g in itertools.combinations(vs, k + 1)]

    def propagar(self):
        """Propagacao unitaria sobre a KB inteira; a KB vira uma forma
        equivalente e menor (fatos + clausulas residuais sem duplicatas)."""
        fatos, cls = self.fatos, set(self.clausulas)
        mudou = True
        while mudou:
            mudou, novas = False, set()
            for c in cls:
                if any(fatos.get(abs(l)) == (l > 0) for l in c):
                    continue   # clausula satisfeita
                resto = tuple(l for l in c if abs(l) not in fatos)
                assert resto, "KB inconsistente (clausula vazia)"
                if len(resto) == 1:
                    fatos[abs(resto[0])] = resto[0] > 0
                    mudou = True
                else:
                    novas.add(resto)
            cls = novas
        self.clausulas = sorted(cls)

    def componentes(self):
        """Componentes conexas (variaveis ligadas por clausulas)."""
        pai = {}

        def achar(x):
            while pai.setdefault(x, x) != x:
                pai[x] = pai[pai[x]]
                x = pai[x]
            return x

        for c in self.clausulas:
            r0 = achar(abs(c[0]))
            for l in c[1:]:
                pai[achar(abs(l))] = r0
        grupos = defaultdict(list)
        for c in self.clausulas:
            grupos[achar(abs(c[0]))].append(c)
        comps = []
        for cl in grupos.values():
            vs = sorted({abs(l) for c in cl for l in c})
            comps.append((vs, cl))
        return comps

    def deduzir(self):
        """ASK de todas as variaveis da fronteira. Devolve (seguras, minas) como
        conjuntos de indices de celula (inclui fatos ja conhecidos)."""
        self.propagar()
        seguras = {v - 1 for v, m in self.fatos.items() if not m}
        minas = {v - 1 for v, m in self.fatos.items() if m}
        self.comps = self.componentes()
        for vs, cls in self.comps:
            vt, vf = set(), set()   # testemunhas: valores ja vistos em modelos

            def registra(modelo):
                for v in vs:
                    if v not in modelo:      # variavel livre: os dois valores
                        vt.add(v)
                        vf.add(v)
                    elif modelo[v]:
                        vt.add(v)
                    else:
                        vf.add(v)

            try:
                m0 = sat(cls, self.ctr)
                assert m0 is not None, "KB insatisfativel (inconsistente)"
                registra(m0)
                for v in vs:
                    if v not in vt:   # ASK: KB |= not M(v) ?  refuta KB ^ M(v)
                        m = sat(cls + [(v,)], self.ctr)
                        if m is None:
                            seguras.add(v - 1)
                            self.fatos[v] = False
                            continue
                        registra(m)
                    if v not in vf:   # ASK: KB |= M(v) ?  refuta KB ^ not M(v)
                        m = sat(cls + [(-v,)], self.ctr)
                        if m is None:
                            minas.add(v - 1)
                            self.fatos[v] = True
                            continue
                        registra(m)
            except Orcamento:
                self.ctr.estouros += 1   # consulta abandonada: sem deducao (sonoro)
        return seguras, minas


# ----------------------------------------------------------------------------
# Tabuleiro e estado do agente
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def vizinhanca(R, C):
    """Lista de vizinhas (8-conexo) de cada celula de um tabuleiro R x C."""
    viz = []
    for r in range(R):
        for c in range(C):
            viz.append([rr * C + cc
                        for rr in range(max(0, r - 1), min(R, r + 2))
                        for cc in range(max(0, c - 1), min(C, c + 2))
                        if (rr, cc) != (r, c)])
    return viz


class Tabuleiro:
    """Tabuleiro real (oculto ao agente). Nao ha mina na celula inicial nem em
    suas vizinhas: a primeira jogada e segura e abre uma cascata."""

    def __init__(self, R, C, n_minas, rng):
        self.R, self.C, self.N = R, C, R * C
        self.viz = vizinhanca(R, C)
        self.inicio = (R // 2) * C + C // 2
        proibidas = {self.inicio, *self.viz[self.inicio]}
        livres = [i for i in range(self.N) if i not in proibidas]
        self.minas = frozenset(rng.sample(livres, n_minas))
        self.cont = [sum(j in self.minas for j in self.viz[i]) for i in range(self.N)]


class Estado:
    """Estado do agente: celulas reveladas (com numero), minas e seguras
    DEDUZIDAS (cel -> origem 'regra'/'dpll'). marcar_* confere cada deducao
    contra o tabuleiro real (assert = 0 falsos positivos)."""

    def __init__(self, tab, kb):
        self.tab, self.kb = tab, kb
        self.rev, self.num = set(), {}
        self.minas, self.seguras = {}, {}
        self.verificadas = 0

    def marcar_segura(self, i, origem):
        if i in self.seguras or i in self.rev:
            return False
        assert i not in self.tab.minas, "FALSO POSITIVO: 'segura' numa mina"
        self.verificadas += 1
        self.seguras[i] = origem
        if self.kb:
            self.kb.tell_unidade(i, False)
        return True

    def marcar_mina(self, i, origem):
        if i in self.minas:
            return False
        assert i in self.tab.minas, "FALSO POSITIVO: 'mina' numa celula segura"
        self.verificadas += 1
        self.minas[i] = origem
        if self.kb:
            self.kb.tell_unidade(i, True)
        return True


def revelar(est, i):
    """Revela a celula i (cascata se numero 0) e faz o TELL das novas celulas
    na KB. Devolve True se pisou numa mina."""
    tab = est.tab
    if i in tab.minas:
        return True
    novas, pilha = [], [i]
    while pilha:
        j = pilha.pop()
        if j in est.rev:
            continue
        est.rev.add(j)
        est.num[j] = tab.cont[j]
        est.seguras.pop(j, None)
        novas.append(j)
        if tab.cont[j] == 0:
            pilha.extend(k for k in tab.viz[j] if k not in est.rev)
    if est.kb:
        for j in novas:   # depois da cascata: vizinhas nao reveladas finais
            est.kb.tell_revelada(j, tab.cont[j], [k for k in tab.viz[j] if k not in est.rev])
    return False


# ----------------------------------------------------------------------------
# Encadeamento barato (regras de contagem e subconjunto)
# ----------------------------------------------------------------------------
def regras(est):
    """Aplica as regras ate o ponto fixo. Devolve True se deduziu algo novo."""
    tab, novo = est.tab, False
    while True:
        restr = set()
        for i in est.rev:
            u = frozenset(j for j in tab.viz[i]
                          if j not in est.rev and j not in est.minas and j not in est.seguras)
            if u:
                restr.add((u, est.num[i] - sum(j in est.minas for j in tab.viz[i])))
        lista = list(restr)
        achou = False
        for u, rem in lista:   # contagem trivial
            if rem == 0:
                for j in u:
                    achou |= est.marcar_segura(j, "regra")
            elif rem == len(u):
                for j in u:
                    achou |= est.marcar_mina(j, "regra")
        if not achou:          # subconjunto: so entre restricoes que se tocam
            idx = defaultdict(list)
            for k, (u, _) in enumerate(lista):
                for j in u:
                    idx[j].append(k)
            for k, (u, rem) in enumerate(lista):
                for m in {m for j in u for m in idx[j]}:
                    u2, rem2 = lista[m]
                    if m != k and u < u2:
                        dif, dr = u2 - u, rem2 - rem
                        if dr == 0:
                            for j in dif:
                                achou |= est.marcar_segura(j, "regra")
                        elif dr == len(dif):
                            for j in dif:
                                achou |= est.marcar_mina(j, "regra")
        if not achou:
            return novo
        novo = True


def varredura_kb(est):
    """ASK ao DPLL de todas as celulas da fronteira; registra o que deduziu."""
    seguras, minas = est.kb.deduzir()
    achou = False
    for i in seguras:
        if i not in est.rev:
            achou |= est.marcar_segura(i, "dpll")
    for i in minas:
        achou |= est.marcar_mina(i, "dpll")
    return achou


# ----------------------------------------------------------------------------
# Chute e agente
# ----------------------------------------------------------------------------
def prob_local(est, i):
    """Heuristica de reserva: maior rem/|U| entre os numeros vizinhos de i."""
    tab, melhor = est.tab, 0.0
    for j in tab.viz[i]:
        if j in est.rev:
            u = [k for k in tab.viz[j] if k not in est.rev and k not in est.minas]
            rem = est.num[j] - sum(k in est.minas for k in tab.viz[j])
            melhor = max(melhor, rem / len(u))
    return melhor


def escolher_chute(est, rng, prob_modelos, ctr):
    """Escolhe a celula do chute. Sem prob_modelos: uniforme. Com: menor P(mina)
    (contagem ponderada de modelos por componente da fronteira). Devolve
    (celula, P(mina) estimada ou None, dict de probabilidades)."""
    cand = [i for i in range(est.tab.N) if i not in est.rev and i not in est.minas]
    if not prob_modelos:
        return rng.choice(cand), None, {}
    m_rest = len(est.tab.minas) - len(est.minas)
    rho = min(max(m_rest / len(cand), 1e-6), 1 - 1e-6)
    w = rho / (1 - rho)
    prob = {}
    for vs, cls in est.kb.comps:
        try:
            p = contar_modelos(cls, vs, w, LIMITE_MODELOS)
            prob.update({v - 1: q for v, q in p.items()})
        except Orcamento:
            ctr.reserva += 1
            prob.update({v - 1: prob_local(est, v - 1) for v in vs})
    interior = [i for i in cand if i not in prob]
    if interior:
        esperado = sum(prob.values())
        p_int = min(max((m_rest - esperado) / len(interior), 0.0), 1.0)
        prob.update(dict.fromkeys(interior, p_int))
    pmin = min(prob[i] for i in cand)
    melhores = [i for i in cand if prob[i] <= pmin + 1e-12]
    return rng.choice(melhores), pmin, prob


Agente = namedtuple("Agente", "nome regras dpll prob")
AGENTES = [
    Agente("aleatorio", False, False, False),
    Agente("regras", True, False, False),
    Agente("regras+DPLL", True, True, False),
    Agente("KB completo", True, True, True),
]
AG_SO_DPLL = Agente("so DPLL", False, True, True)


def jogar(tab, ag, rng, ctr, gancho=None):
    """Joga uma partida completa. Devolve dict de metricas. 'gancho' (opcional)
    recebe eventos ('dpll'|'chute', estado, ...) para desenhar figuras."""
    est = Estado(tab, KB(ctr) if ag.dpll else None)
    r = dict(venceu=False, regra=0, dpll=0, chute=0, t_regras=0.0, t_dpll=0.0,
             t_chute=0.0, p_chutes=[], n_regras=0, n_dpll=0)
    t_ini = time.perf_counter()
    revelar(est, tab.inicio)   # primeira jogada: segura por construcao
    alvo = tab.N - len(tab.minas)
    visto_dpll = False
    while len(est.rev) < alvo:
        if est.seguras:        # 1) revela uma deduzida segura
            i, origem = est.seguras.popitem()
            if i in est.rev:
                continue
            assert not revelar(est, i), "deducao segura pisou em mina"
            r[origem] += 1
            continue
        if ag.regras:          # 2) encadeamento barato
            t0 = time.perf_counter()
            achou = regras(est)
            r["t_regras"] += time.perf_counter() - t0
            r["n_regras"] += 1
            if achou:
                continue
        if ag.dpll:            # 3) DPLL (ASK de toda a fronteira)
            t0 = time.perf_counter()
            achou = varredura_kb(est)
            r["t_dpll"] += time.perf_counter() - t0
            r["n_dpll"] += 1
            if achou:
                if gancho and not visto_dpll and ag.regras:
                    gancho("dpll", est)
                    visto_dpll = True
                continue
        t0 = time.perf_counter()   # 4) nada deduzivel: chute
        i, p, prob = escolher_chute(est, rng, ag.prob, ctr)
        r["t_chute"] += time.perf_counter() - t0
        if gancho:
            gancho("chute", est, i, prob)
        r["chute"] += 1
        if p is not None:
            r["p_chutes"].append(p)
        if revelar(est, i):
            break
    else:
        r["venceu"] = True
    r["tempo"] = time.perf_counter() - t_ini
    r["verificadas"] = est.verificadas
    return r


# ----------------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------------
def rodar_config(R, C, dens, n_jogos, agentes, semente):
    """Roda 'n_jogos' tabuleiros (mesmos para todos os agentes). Devolve
    {nome: dict agregado}."""
    n_minas = round(dens * R * C)
    res = {}
    for ag in agentes:
        ctr = Contadores()
        partidas = []
        for j in range(n_jogos):
            tab = Tabuleiro(R, C, n_minas, random.Random(semente * 100000 + j))
            partidas.append(jogar(tab, ag, random.Random(semente * 100000 + j + 7), ctr))
        n = len(partidas)
        venc = np.array([p["venceu"] for p in partidas], float)
        moves = sum(p["regra"] + p["dpll"] + p["chute"] for p in partidas)
        pc = [q for p in partidas for q in p["p_chutes"]]
        res[ag.nome] = dict(
            n=n, venc=venc, taxa=venc.mean(),
            ep=1.96 * np.sqrt(venc.mean() * (1 - venc.mean()) / n),
            regra=sum(p["regra"] for p in partidas), dpll=sum(p["dpll"] for p in partidas),
            chute=sum(p["chute"] for p in partidas), moves=moves,
            ms=1000 * np.mean([p["tempo"] for p in partidas]),
            t_regras=1000 * sum(p["t_regras"] for p in partidas) / n,
            t_dpll=1000 * sum(p["t_dpll"] for p in partidas) / n,
            t_chute=1000 * sum(p["t_chute"] for p in partidas) / n,
            chamadas=ctr.chamadas / n, nos=ctr.nos / n, estouros=ctr.estouros, reserva=ctr.reserva,
            sem_chute=np.mean([p["chute"] == 0 for p in partidas]),
            p_chute=float(np.mean(pc)) if pc else float("nan"),
            verificadas=sum(p["verificadas"] for p in partidas),
            ok_chute=sum(p["chute"] for p in partidas if p["venceu"]),
            n_regras=sum(p["n_regras"] for p in partidas),
            n_dpll=sum(p["n_dpll"] for p in partidas),
            resultados=[p["venceu"] for p in partidas],
        )
    return res


def tabela_experimento(titulo, R, C, resultados):
    """Imprime as tabelas de um tamanho de tabuleiro (uma linha por
    densidade x agente)."""
    print(f"\n=== {titulo} ===")
    print("dens  minas  agente        jogos  vitoria(+-IC95)  %deduz  regra  dpll  chute"
          "  ms/jogo  chamDPLL  nosDPLL  estouros")
    for d, res in resultados.items():
        for nome, a in res.items():
            ded = a["regra"] + a["dpll"]
            pct = 100 * ded / a["moves"] if a["moves"] else 0.0
            print(f"{d:4.0%}  {round(d * R * C):5d}  {nome:12s}  {a['n']:5d}  "
                  f"{100 * a['taxa']:5.1f}% +-{100 * a['ep']:4.1f}   {pct:5.1f}  {a['regra']:5d} "
                  f"{a['dpll']:5d} {a['chute']:6d}  {a['ms']:7.1f}  {a['chamadas']:8.1f} "
                  f"{a['nos']:8.1f} {a['estouros']:5d}")


# ----------------------------------------------------------------------------
# Autoverificacao
# ----------------------------------------------------------------------------
def verificar_dpll(rng):
    """DPLL contra forca bruta em 400 CNFs aleatorias (8 vars, 3-CNF)."""
    n_sat = 0
    for _ in range(400):
        n_cl = rng.randint(15, 45)
        cls = [tuple(rng.choice((-1, 1)) * v for v in rng.sample(range(1, 9), 3))
               for _ in range(n_cl)]
        modelo = dpll(cls, {}, Contadores())
        existe = any(all(any((a >> (abs(l) - 1) & 1) == (l > 0) for l in c) for c in cls)
                     for a in range(256))
        assert (modelo is not None) == existe, "DPLL diverge da forca bruta"
        if modelo is not None:   # o modelo parcial satisfaz todas as clausulas
            n_sat += 1
            assert all(any(modelo.get(abs(l)) == (l > 0) for l in c) for c in cls)
    return n_sat


def verificar_kb(rng):
    """Deducoes da KB e das regras contra enumeracao exaustiva da fronteira em
    tabuleiros 6x6: KB deve dar EXATAMENTE o entailment (sonora e completa) e as
    regras devem ser um subconjunto dele. Devolve (estados, deduzidas, regras)."""
    estados = n_ded = n_reg = 0
    for _ in range(300):
        R = C = 6
        tab = Tabuleiro(R, C, rng.randint(4, 8), rng)
        kb = KB(Contadores())
        est, est2 = Estado(tab, kb), Estado(tab, None)
        for e in (est, est2):
            revelar(e, tab.inicio)
        seguras_extra = [i for i in range(tab.N) if i not in tab.minas and i not in est.rev]
        for i in rng.sample(seguras_extra, min(3, len(seguras_extra))):
            for e in (est, est2):
                if i not in e.rev:
                    revelar(e, i)
        fronteira = sorted({j for i in est.rev for j in tab.viz[i] if j not in est.rev})
        if len(fronteira) > 16 or not fronteira:
            continue
        restr = [(i, [j for j in tab.viz[i] if j not in est.rev]) for i in est.rev]
        modelos = []
        for a in range(1 << len(fronteira)):
            val = {j: (a >> k) & 1 for k, j in enumerate(fronteira)}
            if all(sum(val[j] for j in u) == tab.cont[i] for i, u in restr if u):
                modelos.append(val)
        ent_seg = {j for j in fronteira if all(m[j] == 0 for m in modelos)}
        ent_min = {j for j in fronteira if all(m[j] == 1 for m in modelos)}
        seg, minas = kb.deduzir()
        seg = {i for i in seg if i not in est.rev}
        assert seg == ent_seg and minas == ent_min, "KB diverge do entailment exato"
        regras(est2)
        assert set(est2.seguras) <= ent_seg and set(est2.minas) <= ent_min
        estados += 1
        n_ded += len(ent_seg) + len(ent_min)
        n_reg += len(est2.seguras) + len(est2.minas)
    return estados, n_ded, n_reg


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
ESTILOS = {"aleatorio": ("#888888", "o"), "regras": ("#0072b2", "s"),
           "regras+DPLL": ("#009e73", "^"), "KB completo": ("#d55e00", "D")}
COR = {"rev": "#f4f4f4", "seg": "#4c9be8", "min": "#d55e00", "desc": "#9a9a9a", "chute": "#f0c020"}


def desenhar_tabuleiro(ax, tab, snap, titulo):
    """Desenha um instantaneo: reveladas (numero), seguras/minas deduzidas
    (com origem), desconhecidas e a celula do chute (com P(mina))."""
    R, C = tab.R, tab.C
    ax.set_xlim(0, C)
    ax.set_ylim(R, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(tab.N):
        r, c = divmod(i, C)
        txt, cor, tc = "", COR["desc"], "black"
        if i in snap["rev"]:
            cor, n = COR["rev"], snap["num"][i]
            txt, tc = ("" if n == 0 else str(n)), "black"
        elif i == snap.get("chute"):
            cor = COR["chute"]
            txt = "?\n%.0f%%" % (100 * snap["prob"][i])
        elif i in snap["minas"]:
            cor, txt, tc = COR["min"], "M", "white"
        elif i in snap["seguras"]:
            cor, txt, tc = COR["seg"], "S", "white"
        ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=cor, edgecolor="white", lw=1.5))
        if txt:
            ax.text(c + .5, r + .5, txt, ha="center", va="center", color=tc,
                    fontsize=8.5 if "\n" in txt else 11, fontweight="bold")
        # borda escura marca o que foi deduzido pelo DPLL (regras nao viam)
        for tipo in ("minas", "seguras"):
            if snap[tipo].get(i) == "dpll" and i not in snap["rev"]:
                ax.add_patch(plt.Rectangle((c + .06, r + .06), .88, .88, fill=False,
                                           edgecolor="black", lw=2.2))
    ax.set_title(titulo, fontsize=10)


def figura_tabuleiro(caminho):
    """Procura uma partida 9x9 (15%) com um evento DPLL e um chute e desenha os
    dois instantaneos lado a lado."""
    for s in range(400):
        eventos = {}

        def gancho(tipo, est, i=None, prob=None):
            if tipo in eventos:
                return
            eventos[tipo] = dict(rev=set(est.rev), num=dict(est.num), minas=dict(est.minas),
                                 seguras=dict(est.seguras), chute=i, prob=prob)

        tab = Tabuleiro(9, 9, 12, random.Random(555000 + s))
        ag = AGENTES[3]
        jogar(tab, ag, random.Random(s), Contadores(), gancho)
        e = eventos.get("dpll")
        if e and "chute" in eventos and len(e["seguras"]) + len(e["minas"]) >= 2 \
                and eventos["chute"]["rev"] != e["rev"]:
            break
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6))
    desenhar_tabuleiro(axs[0], tab, eventos["dpll"],
                       "Regras travaram; o DPLL deduziu (borda preta)")
    desenhar_tabuleiro(axs[1], tab, eventos["chute"],
                       "Nada dedutivel: chute na menor P(mina)")
    leg = [plt.Rectangle((0, 0), 1, 1, fc=COR[k], ec="white")
           for k in ("rev", "seg", "min", "desc", "chute")]
    fig.legend(leg, ["revelada (numero)", "S: deduzida segura", "M: deduzida mina",
                     "desconhecida", "chute (P estimada)"], loc="lower center", ncol=5,
               frameon=False, fontsize=9)
    fig.suptitle("Campo Minado 9x9, 15% de minas: instantaneos de uma partida do agente KB completo",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def figura_vitoria(dados, caminho):
    """Taxa de vitoria x densidade por agente (um painel por tamanho)."""
    fig, axs = plt.subplots(1, len(dados), figsize=(6 * len(dados), 4.4), squeeze=False)
    for ax, (tam, resultados) in zip(axs[0], dados.items()):
        ds = list(resultados)
        for ag in AGENTES:
            y = [100 * resultados[d][ag.nome]["taxa"] for d in ds]
            e = [100 * resultados[d][ag.nome]["ep"] for d in ds]
            cor, mk = ESTILOS[ag.nome]
            ax.errorbar([100 * d for d in ds], y, yerr=e, color=cor, marker=mk, capsize=3,
                        lw=1.6, label=ag.nome)
        ax.set_title(f"{tam} ({resultados[ds[0]]['KB completo']['n']} jogos por ponto)")
        ax.set_xlabel("densidade de minas (%)")
        ax.set_ylabel("taxa de vitoria (%)")
        ax.set_xticks([100 * d for d in ds])
        ax.set_ylim(-3, 103)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8.5)
    fig.suptitle("Taxa de vitoria por agente (barras: IC 95% binomial)", fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def figura_custo(res9, caminho):
    """Esquerda: origem das jogadas do KB completo (regra/DPLL/chute).
    Direita: tempo medio por jogo por agente (escala log) e chamadas DPLL."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    ds = list(res9)
    x = np.arange(len(ds))
    fr = np.array([[res9[d]["KB completo"][k] / res9[d]["KB completo"]["moves"] * 100
                    for k in ("regra", "dpll", "chute")] for d in ds])
    base = np.zeros(len(ds))
    for k, (nome, cor) in enumerate([("dedução por regras", "#0072b2"),
                                     ("dedução por DPLL", "#009e73"), ("chute", "#e69f00")]):
        axs[0].bar(x, fr[:, k], bottom=base, color=cor, label=nome, width=.6)
        for xi, v, b in zip(x, fr[:, k], base):
            if v >= 4:
                axs[0].text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                            color="white", fontsize=9)
        base += fr[:, k]
    axs[0].set_xticks(x, [f"{d:.0%}" for d in ds])
    axs[0].set_xlabel("densidade de minas")
    axs[0].set_ylabel("% das jogadas (KB completo, 9x9)")
    axs[0].set_title("De onde vem cada jogada")
    axs[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.17), ncol=3, frameon=False)
    larg = .2
    for k, ag in enumerate(AGENTES):
        axs[1].bar(x + (k - 1.5) * larg, [res9[d][ag.nome]["ms"] for d in ds], larg, label=ag.nome,
                   color=ESTILOS[ag.nome][0])
    for xi, d in zip(x, ds):
        axs[1].text(xi + 1.5 * larg, res9[d]["KB completo"]["ms"] * 1.15,
                    f"{res9[d]['KB completo']['chamadas']:.1f}\nchamadas", ha="center", fontsize=8)
    axs[1].set_yscale("log")
    axs[1].set_xticks(x, [f"{d:.0%}" for d in ds])
    axs[1].set_xlabel("densidade de minas")
    axs[1].set_ylabel("tempo medio por jogo (ms, log)")
    axs[1].set_title("Tempo por jogo (texto: chamadas DPLL/jogo)")
    axs[1].legend(fontsize=8.5, loc="upper left")
    axs[1].set_ylim(top=axs[1].get_ylim()[1] * 3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    t_ini = time.perf_counter()
    FIGS.mkdir(exist_ok=True)
    rng = random.Random(6)

    print("PROJETO 6 - Base de conhecimento: Campo Minado por inferencia logica")
    print("\n--- Autoverificacao ---")
    n_sat = verificar_dpll(rng)
    print(f"DPLL x forca bruta: 400 CNFs aleatorias iguais ({n_sat} satisfativeis, "
          f"{400 - n_sat} insatisfativeis)")
    est_n, ded, reg = verificar_kb(rng)
    print(f"KB x enumeracao exaustiva da fronteira (6x6): {est_n} estados, KB == entailment "
          f"exato em todos ({ded} celulas deduziveis; as regras acharam {reg} = "
          f"{100 * reg / max(ded, 1):.1f}%)")
    progresso("autoverificacao ok")

    # Experimento principal: 9x9
    plano = [("Tabuleiro 9x9", 9, 9, 500), ("Tabuleiro 16x16", 16, 16, 200)]
    dados = {}
    for titulo, R, C, n_jogos in plano:
        resultados = {}
        for d in DENSIDADES:
            t0 = time.perf_counter()
            resultados[d] = rodar_config(R, C, d, n_jogos, AGENTES, semente=R + int(d * 100))
            progresso(f"{titulo} dens {d:.0%}: {time.perf_counter() - t0:.1f}s")
        dados[titulo.split()[-1]] = resultados
        tabela_experimento(f"{titulo}, {n_jogos} jogos por densidade", R, C, resultados)

    # Sonoridade: contagem global de deducoes conferidas
    total_verif = sum(a["verificadas"] for res in dados.values()
                      for r in res.values() for a in r.values())
    print(f"\nDeducoes conferidas contra o tabuleiro real: {total_verif}; falsos positivos: 0 "
          f"(um assert aborta na primeira ocorrencia)")
    todos = [a for res in dados.values() for r in res.values() for a in r.values()]
    print(f"Consultas DPLL abandonadas por orcamento: {sum(a['estouros'] for a in todos)}; "
          f"chutes com contagem de modelos estourada (heuristica local): "
          f"{sum(a['reserva'] for a in todos)}")

    # Custo: regras x DPLL no KB completo 9x9 e comparacao com 'so DPLL'
    print("\n=== Custo por etapa (KB completo, 9x9, ms por jogo) ===")
    print("dens  t_regras  t_DPLL  t_chute  chamadas_regras  chamadas_DPLL  ded_regras  ded_DPLL")
    for d, r in dados["9x9"].items():
        a = r["KB completo"]
        print(f"{d:4.0%}  {a['t_regras']:8.2f}  {a['t_dpll']:6.2f}  {a['t_chute']:7.2f}  "
              f"{a['n_regras'] / a['n']:15.1f}  {a['n_dpll'] / a['n']:13.1f}  "
              f"{a['regra']:10d}  {a['dpll']:8d}")

    print("\n=== Ablacao: 'so DPLL' (sem encadeamento) x 'KB completo' (9x9, 15%, 100 jogos) ===")
    sub = 100
    par = rodar_config(9, 9, 0.15, sub, [AGENTES[3], AG_SO_DPLL], semente=9 + 15)
    for nome, a in par.items():
        print(f"{nome:12s}  vitoria {100 * a['taxa']:5.1f}%  ms/jogo {a['ms']:7.1f}  "
              f"chamadas DPLL/jogo {a['chamadas']:7.1f}  nos/jogo {a['nos']:9.1f}")
    ig = par["KB completo"]["resultados"] == par["so DPLL"]["resultados"]
    print(f"Mesmo resultado (vitoria/derrota) em todas as {sub} partidas: {ig}")
    assert ig, "so DPLL e KB completo deveriam decidir igual (deducao completa)"
    fator = par["so DPLL"]["ms"] / par["KB completo"]["ms"]
    print(f"Custo do 'so DPLL' / 'KB completo': {fator:.2f}x em tempo; "
          f"{par['so DPLL']['chamadas'] / par['KB completo']['chamadas']:.2f}x em chamadas DPLL")

    # Probabilidade de sobreviver ao chute
    print("\n=== Chutes do KB completo (9x9) ===")
    print("dens  chutes/jogo  P(mina) media no chute  partidas sem chute")
    for d, r in dados["9x9"].items():
        a = r["KB completo"]
        print(f"{d:4.0%}  {a['chute'] / a['n']:11.2f}  {100 * a['p_chute']:20.1f}%  "
              f"{100 * a['sem_chute']:16.1f}%")

    # Assercoes estatisticas robustas
    for tam, res in dados.items():
        for d, r in res.items():
            assert r["KB completo"]["taxa"] >= r["aleatorio"]["taxa"], (tam, d)
    r9 = dados["9x9"]
    assert r9[0.10]["KB completo"]["taxa"] > r9[0.25]["KB completo"]["taxa"]

    figura_tabuleiro(FIGS / "06_tabuleiro.png")
    figura_vitoria(dados, FIGS / "06_vitoria.png")
    figura_custo(r9, FIGS / "06_custo.png")
    print(f"\nFiguras: figs/06_tabuleiro.png, figs/06_vitoria.png, figs/06_custo.png")
    print(f"Tempo total: {time.perf_counter() - t_ini:.0f}s")


if __name__ == "__main__":
    main()
