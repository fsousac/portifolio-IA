"""
Projeto 2 - Busca NAO informada: o quebra-cabeca Rush Hour
==========================================================

PROBLEMA
--------
Tabuleiro 6x6 com carros (2x1 ou 3x1) que so deslizam ao longo do proprio eixo
(horizontal ou vertical). O carro vermelho 'X' (horizontal, linha 3, indice 2)
deve alcancar a borda direita, isto e, ocupar as colunas 4-5 da linha 3.

FORMULACAO
----------
* Estado   : tupla imutavel (hashable) com a posicao de cada carro ao longo do
             seu eixo (coluna da extremidade esquerda p/ horizontais, linha da
             extremidade superior p/ verticais). A geometria (orientacao, linha
             ou coluna fixa, comprimento) fica no objeto Tabuleiro, fora do estado.
* Acoes    : deslocar UM carro por qualquer numero k >= 1 de celulas livres em
             uma das duas direcoes ("movimento" = um deslize de qualquer distancia).
* Custo    : 1 por movimento (definicao principal).  Variante comparada com a UCS:
             1 por celula percorrida.  A saida final do carro vermelho (sair do
             tabuleiro) NAO conta como movimento: o objetivo e o vermelho em X=4.
* Objetivo : posicao do carro vermelho == 4.  O objetivo e um CONJUNTO de estados
             (os demais carros podem estar em qualquer lugar).

ALGORITMOS (todos implementados a mao)
--------------------------------------
BFS, DFS (busca em grafo com deteccao de repetidos), DLS, IDDFS, UCS e busca
bidirecional.  Metricas: nos expandidos, tamanho maximo da fronteira, estruturas
armazenadas, tempo e comprimento/custo da solucao; timeout registrado como TIMEOUT.

INSTANCIAS
----------
10 tabuleiros gerados por BUSCA REVERSA: a partir de um layout aleatorio (seed fixa),
enumera-se a componente conexa de estados e faz-se uma BFS multi-fonte a partir de
todos os estados resolvidos; escolhe-se um estado cuja distancia ate o conjunto
solucao e a profundidade desejada.  Assim a solucao otima (em movimentos) e conhecida
por construcao.  Os tabuleiros gerados foram gravados no codigo (INSTANCIAS);
`python 02_busca_nao_informada.py --gerar` reexecuta o gerador.

Execucao: python 02_busca_nao_informada.py   (gera figs/02_*.png e figs/02_saida.txt)
"""

import heapq
import random
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, FancyArrow  # noqa: E402

LADO = 6                 # tabuleiro 6x6
LINHA_SAIDA = 2          # linha (0-indexada) do carro vermelho
# posicao do vermelho (comprimento 2) encostado na direita
POS_OBJETIVO = LADO - 2
TIMEOUT_S = 10.0         # limite de tempo padrao por execucao (segundos)
LIMITES = {"IDDFS": 120.0, "IDDFS-puro": 120.0}  # tempo: so rede de seguranca (o corte real e o orcamento de nos)
# Orcamento em nos expandidos (deterministico, ao contrario do tempo): a IDDFS sem tabela
# e cortada aqui e registrada como TIMEOUT, o que torna seus numeros reprodutiveis.
# O da IDDFS com tabela (maximo observado: ~307 mil) so evita laco infinito.
ORCAMENTO_NOS = {"IDDFS-puro": 150_000, "IDDFS": 2_000_000}
LIMITE_DLS = 14          # limite de profundidade da DLS "pura" (demonstrativo)

RAIZ = Path(__file__).resolve().parent
FIGS = RAIZ / "figs"

# --------------------------------------------------------------------------
# Instancias (tabuleiro inicial, otimo em movimentos conhecido por construcao).
# '.' = vazio, 'X' = carro vermelho, letras = demais carros.
# Geradas por gerar_instancias() (busca reversa); 'meta' e o estado resolvido mais proximo.
# --------------------------------------------------------------------------
INSTANCIAS = [
    dict(nome="R01", otimo=5,
         tab=['..EJBB', 'IIEJ..', '..XXDC', '..F.DC', 'HGF.DC', 'HGAAA.'],
         meta=['..EJBB', 'IIEJ..', '....XX', '.GF.DC', 'HGF.DC', 'HAAADC']),
    dict(nome="R02", otimo=8,
         tab=['..BB..', 'IIEJ..', 'XXEJDC', '..F.DC', 'HGF.DC', 'HGAAA.'],
         meta=['..EJBB', 'IIEJ..', '....XX', '.GF.DC', 'HGF.DC', 'HAAADC']),
    dict(nome="R03", otimo=11,
         tab=['H.BB..', 'H.IID.', 'XXEJDC', '..EJDC', '.GF..C', '.GFAAA'],
         meta=['H.E.BB', 'H.EII.', '....XX', '.GFJDC', '.GFJDC', '.AAADC']),
    dict(nome="R04", otimo=14,
         tab=['.K.II.', '.KGDF.', 'XXGDFC', 'AHHBBC', 'A.JEE.', '..J...'],
         meta=['AKGDII', 'AKGD..', '....XX', 'HHBBFC', 'EEJ.FC', '..J...']),
    dict(nome="R05", otimo=17,
         tab=['.KIIF.', '.KGDF.', 'XXGD.C', 'HHJBBC', 'A.JEE.', 'A.....'],
         meta=['AKGDII', 'AKGD..', '....XX', 'HHBBFC', 'EEJ.FC', '..J...']),
    dict(nome="R06", otimo=20,
         tab=['.GBBII', '.GEJJ.', 'XXE.D.', '.FF.DC', 'H.KKDC', 'H.AAAC'],
         meta=['HGBBII', 'HGJJ..', '....XX', 'FFE.DC', 'KKE.DC', '.AAADC']),
    dict(nome="R07", otimo=23,
         tab=['HBBII.', 'H.EJJ.', 'XXE.D.', '.GFFDC', '.GKKDC', '..AAAC'],
         meta=['HGBBII', 'HGJJ..', '....XX', 'FFE.DC', 'KKE.DC', '.AAADC']),
    dict(nome="R08", otimo=26,
         tab=['LLDFFF', '..DII.', 'JXXG..', 'JM.GCC', 'JMAA.B', 'HHHKKB'],
         meta=['LLFFFB', 'JII..B', 'J...XX', 'JMDGCC', '.MDGAA', 'HHHKK.']),
    dict(nome="R09", otimo=28,
         tab=['.LGIIC', '.LGDFC', '.XXDF.', 'HHJBBE', 'A.JM.E', 'AKKM..'],
         meta=['IIGDFC', '..GDFC', '....XX', 'ALHHBB', 'ALJM.E', 'KKJM.E']),
    dict(nome="R10", otimo=30,
         tab=['.LIIFC', '.LGDFC', 'XXGD..', 'HHJBBE', 'A.JM.E', 'AKKM..'],
         meta=['IIGDFC', '..GDFC', '....XX', 'ALHHBB', 'ALJM.E', 'KKJM.E']),
]
# 'meta' = estado resolvido mais proximo (usado so pela busca bidirecional).


# ==========================================================================
# Modelo do problema
# ==========================================================================
class Tabuleiro:
    """Geometria fixa do puzzle + funcoes de transicao sobre estados (tuplas)."""

    def __init__(self, carros):
        """Guarda a geometria de cada carro e pre-calcula as mascaras de ocupacao."""
        # carros: lista de (letra, horizontal, fixo, comprimento); indice 0 = 'X'
        self.letras = [c[0] for c in carros]
        self.horiz = [c[1] for c in carros]
        self.fixo = [c[2] for c in carros]
        self.comp = [c[3] for c in carros]
        # mascara de ocupacao (36 bits) de cada carro em cada posicao possivel
        self.mascara = []
        for i in range(len(carros)):
            ms = []
            for p in range(LADO - self.comp[i] + 1):
                m = 0
                for cel in self.celulas(i, p):
                    m |= 1 << cel
                ms.append(m)
            self.mascara.append(ms)

    def celulas(self, i, p):
        """Indices lineares (0..35) das celulas do carro i na posicao p."""
        if self.horiz[i]:
            return [self.fixo[i] * LADO + p + j for j in range(self.comp[i])]
        return [(p + j) * LADO + self.fixo[i] for j in range(self.comp[i])]

    @staticmethod
    def de_texto(linhas):
        """Le um tabuleiro em texto (6 strings de 6 chars). Retorna (Tabuleiro, estado)."""
        assert len(linhas) == LADO and all(len(l) == LADO for l in linhas)
        celulas = {}
        for r, l in enumerate(linhas):
            for c, ch in enumerate(l):
                if ch != ".":
                    celulas.setdefault(ch, []).append((r, c))
        letras = ["X"] + sorted(k for k in celulas if k != "X")
        carros, estado = [], []
        for ch in letras:
            cs = celulas[ch]
            rs = {r for r, _ in cs}
            cols = {c for _, c in cs}
            if len(rs) == 1:                       # horizontal
                assert max(cols) - min(cols) + 1 == len(cs)
                carros.append((ch, True, cs[0][0], len(cs)))
                estado.append(min(cols))
            else:                                  # vertical
                assert len(cols) == 1 and max(rs) - min(rs) + 1 == len(cs)
                carros.append((ch, False, cs[0][1], len(cs)))
                estado.append(min(rs))
        assert carros[0][1] and carros[0][2] == LINHA_SAIDA and carros[0][3] == 2
        return Tabuleiro(carros), tuple(estado)

    def ocupacao(self, e):
        """Mascara de bits (36 bits) das celulas ocupadas no estado e."""
        occ = 0
        for i, p in enumerate(e):
            occ |= self.mascara[i][p]
        return occ

    def sucessores(self, e):
        """Lista de (novo_estado, movimento); movimento = (carro, pos_de, pos_para).
        Cada deslize de qualquer distancia gera UM sucessor."""
        occ = self.ocupacao(e)
        saida = []
        for i, p in enumerate(e):
            h, fx, cp = self.horiz[i], self.fixo[i], self.comp[i]
            passo = 1 if h else LADO
            base = fx * LADO if h else fx
            q = p - 1                              # para tras (esq./cima)
            while q >= 0 and not ((occ >> (base + passo * q)) & 1):
                saida.append((e[:i] + (q,) + e[i + 1:], (i, p, q)))
                q -= 1
            q = p + 1                              # para frente (dir./baixo)
            while q + cp <= LADO and not ((occ >> (base + passo * (q + cp - 1))) & 1):
                saida.append((e[:i] + (q,) + e[i + 1:], (i, p, q)))
                q += 1
        return saida

    def objetivo(self, e):
        """Teste de objetivo: carro vermelho encostado na borda direita."""
        return e[0] == POS_OBJETIVO

    def texto(self, e):
        """Converte o estado em 6 strings de 6 caracteres ('.' = vazio)."""
        g = [["."] * LADO for _ in range(LADO)]
        for i, p in enumerate(e):
            for cel in self.celulas(i, p):
                g[cel // LADO][cel % LADO] = self.letras[i]
        return ["".join(l) for l in g]


def custo_mov(m, modo):
    """Custo de um movimento: 'mov' -> 1; 'celulas' -> distancia deslizada."""
    return 1 if modo == "mov" else abs(m[2] - m[1])


def validar(tab, e0, caminho, meta=None):
    """Verificador INDEPENDENTE dos sucessores (que usam mascaras de bits): reexecuta a
    solucao numa grade de caracteres, celula por celula.  Cada movimento deve partir da
    posicao atual, deslizar so por celulas livres e o fim deve ser objetivo."""
    e = list(e0)
    g = [list(l) for l in tab.texto(e0)]
    for i, de, para in caminho:
        assert e[i] == de and de != para, "movimento nao parte da posicao atual"
        assert 0 <= para <= LADO - tab.comp[i], "movimento sai do tabuleiro"
        for cel in tab.celulas(i, de):
            g[cel // LADO][cel % LADO] = "."
        # celulas que o carro passa a ocupar (union dos dois trechos) devem estar livres
        for cel in set(tab.celulas(i, para)) | {c2 for q in range(min(de, para), max(de, para) + 1)
                                                for c2 in tab.celulas(i, q)}:
            assert g[cel // LADO][cel % LADO] == ".", "movimento atravessa outro carro"
        e[i] = para
        for cel in tab.celulas(i, para):
            g[cel // LADO][cel % LADO] = tab.letras[i]
    e = tuple(e)
    assert tab.objetivo(e), "estado final nao e objetivo"
    if meta is not None:
        assert e == meta
    return e


def conferir_sucessores(tab, e0, n=300, seed=0):
    """Autoteste: em um passeio aleatorio de n estados, compara `sucessores` (mascaras de
    bits) com uma enumeracao por forca bruta na grade de caracteres (compara o CONJUNTO
    de (carro, nova_posicao), nao so a quantidade)."""
    rng = random.Random(seed)
    e = e0
    for _ in range(n):
        g = tab.texto(e)
        esperado = set()
        for i, p in enumerate(e):
            for q in range(LADO - tab.comp[i] + 1):
                if q == p:
                    continue
                gg = [list(l) for l in g]
                for cel in tab.celulas(i, p):
                    gg[cel // LADO][cel % LADO] = "."
                if all(gg[cel // LADO][cel % LADO] == "."
                       for r in range(min(p, q), max(p, q) + 1) for cel in tab.celulas(i, r)):
                    esperado.add((i, q))
        suc = tab.sucessores(e)
        assert esperado == {(m[0], m[2]) for _, m in suc} and len(esperado) == len(suc), (
            "sucessores divergem da forca bruta")
        e = rng.choice(suc)[0]


# ==========================================================================
# Infraestrutura de medicao
# ==========================================================================
class Timeout(Exception):
    """Sinaliza estouro do limite de tempo ou do orcamento de nos."""

    pass


class Contador:
    """Acumula metricas e aborta por tempo (verifica o relogio a cada 256 visitas)
    ou por orcamento de nos expandidos (max_exp).  'Expandido' = no cujos sucessores
    foram gerados (folhas da DLS, so testadas, nao contam: igual ao teste de objetivo
    na geracao da BFS)."""

    def __init__(self, limite_s, max_exp=None):
        """Zera as metricas e inicia o cronometro."""
        self.t0 = time.perf_counter()
        self.limite = limite_s
        self.max_exp = max_exp  # orcamento opcional de nos expandidos
        self.expandidos = 0   # nos visitados/retirados da fronteira
        self.gerados = 0      # sucessores gerados
        # maior tamanho da fronteira (pilha/fila/heap/profundidade)
        self.front_max = 0
        # maior numero de estados guardados (visitados/tabelas)
        self.armaz = 0

    def visitar(self):
        """Conta uma expansao e aborta (Timeout) se passou do tempo ou do orcamento."""
        self.expandidos += 1
        if self.max_exp is not None and self.expandidos > self.max_exp:
            raise Timeout()
        if self.expandidos & 255 == 0 and time.perf_counter() - self.t0 > self.limite:
            raise Timeout()


def reconstruir(pai, s):
    """Segue os ponteiros pai[s] = (anterior, movimento) ate a raiz."""
    cam = []
    while pai[s] is not None:
        s, m = pai[s][0], pai[s][1]
        cam.append(m)
    cam.reverse()
    return cam


# ==========================================================================
# Algoritmos.  Assinatura: alg(tab, e0, c, **kw) -> (status, caminho)
# ==========================================================================
def bfs(tab, e0, c):
    """Busca em largura; teste de objetivo na GERACAO (poupa uma camada)."""
    if tab.objetivo(e0):
        return "OK", []
    pai = {e0: None}
    fila = deque([e0])
    while fila:
        s = fila.popleft()
        c.visitar()
        for t, m in tab.sucessores(s):
            c.gerados += 1
            if t not in pai:
                pai[t] = (s, m)
                if tab.objetivo(t):
                    c.armaz = len(pai)
                    return "OK", reconstruir(pai, t)
                fila.append(t)
        c.front_max = max(c.front_max, len(fila))
        c.armaz = len(pai)
    return "FALHA", None


def dfs(tab, e0, c):
    """Busca em profundidade em GRAFO: conjunto de explorados evita repeticoes.
    Marca o estado ao SAIR da pilha (comportamento DFS fiel); pilha guarda duplicatas."""
    pai = {}
    pilha = [(e0, None, None)]
    while pilha:
        s, ant, m = pilha.pop()
        if s in pai:
            continue
        pai[s] = None if ant is None else (ant, m)
        c.visitar()
        if tab.objetivo(s):
            c.armaz = len(pai)
            return "OK", reconstruir(pai, s)
        for t, mm in reversed(tab.sucessores(s)):
            c.gerados += 1
            if t not in pai:
                pilha.append((t, s, mm))
        c.front_max = max(c.front_max, len(pilha))
        c.armaz = len(pai)
    return "FALHA", None


def dls(tab, e0, c, limite, modo="tabela"):
    """Busca em profundidade limitada (recursiva).
    modo='caminho': so evita ciclos no caminho atual (memoria O(profundidade), mas
                    explode exponencialmente pois estados repetem por caminhos distintos);
    modo='tabela' : guarda a menor profundidade em que cada estado foi alcancado e so
                    reexpande se chegar mais raso (mantem o teorema de otimalidade da
                    IDDFS, ao custo de memoria proporcional aos estados visitados).
    Retorna ('OK', caminho) | ('CORTE', None) se o limite impediu a busca | ('FALHA', None)."""
    melhor = {e0: 0}
    ancestrais = {e0}
    caminho = []
    cortou = [False]

    def rec(s, d):
        """Recursao da DLS: True se achou o objetivo abaixo de s (profundidade d)."""
        c.visitar()
        c.front_max = max(c.front_max, d + 1)
        if tab.objetivo(s):
            return True
        if d == limite:
            cortou[0] = True
            return False
        folha = d + 1 == limite                    # filhos ja estao no limite de profundidade
        for t, m in tab.sucessores(s):
            c.gerados += 1
            if modo == "tabela":
                if melhor.get(t, 1 << 30) <= d + 1:
                    continue
                melhor[t] = d + 1
            else:
                if t in ancestrais:
                    continue
            if folha:                              # otimizacao: testa a folha sem recursao
                c.front_max = max(c.front_max, d + 2)
                if tab.objetivo(t):
                    caminho.append(m)
                    return True
                cortou[0] = True
                continue
            if modo == "caminho":
                ancestrais.add(t)
            caminho.append(m)
            if rec(t, d + 1):
                return True
            caminho.pop()
            if modo == "caminho":
                ancestrais.discard(t)
        return False

    try:
        ok = rec(e0, 0)
    finally:                                       # registra memoria mesmo em TIMEOUT
        c.armaz = max(c.armaz, len(melhor) if modo == "tabela" else limite + 1)
    if ok:
        return "OK", list(caminho)
    return ("CORTE" if cortou[0] else "FALHA"), None


def dls_fixa(tab, e0, c):
    """DLS com limite fixo LIMITE_DLS (adaptador para `executar`)."""
    return dls(tab, e0, c, LIMITE_DLS, "tabela")


def iddfs(tab, e0, c, modo="tabela"):
    """Aprofundamento iterativo: DLS com limite 0, 1, 2, ... ate achar solucao.
    Otima em numero de movimentos (custo unitario)."""
    for lim in range(0, 200):
        st, cam = dls(tab, e0, c, lim, modo)
        if st == "OK":
            return "OK", cam
        if st == "FALHA":
            return "FALHA", None
    return "FALHA", None


def iddfs_puro(tab, e0, c):
    """IDDFS so com deteccao de ciclo no caminho (versao "de livro")."""
    return iddfs(tab, e0, c, "caminho")


def ucs(tab, e0, c, modo="mov"):
    """Busca de custo uniforme (Dijkstra): heap por custo acumulado g; teste de
    objetivo ao RETIRAR da fila (garante otimalidade para custos nao unitarios)."""
    g = {e0: 0}
    pai = {e0: None}
    heap = [(0, 0, e0)]
    n = 0
    fechados = set()
    while heap:
        gs, _, s = heapq.heappop(heap)
        if s in fechados:
            continue
        fechados.add(s)
        c.visitar()
        if tab.objetivo(s):
            c.armaz = len(g)
            return "OK", reconstruir(pai, s)
        for t, m in tab.sucessores(s):
            c.gerados += 1
            gt = gs + custo_mov(m, modo)
            if gt < g.get(t, 1 << 30):
                g[t] = gt
                pai[t] = (s, m)
                n += 1
                heapq.heappush(heap, (gt, n, t))
        c.front_max = max(c.front_max, len(heap))
        c.armaz = len(g)
    return "FALHA", None


def ucs_celulas(tab, e0, c):
    """UCS com custo = numero de celulas percorridas (variante de custo)."""
    return ucs(tab, e0, c, "celulas")


def bidirecional(tab, e0, c, meta):
    """BFS bidirecional entre e0 e um estado-objetivo CANONICO `meta`.
    Como todo movimento e reversivel, a busca reversa usa os mesmos sucessores.
    Expande sempre a camada inteira da menor fronteira; ao encontrar interseccao
    termina a camada e devolve o menor caminho (otimo ATE `meta`).
    Limitacao: exige um objetivo unico e explicito; o objetivo real e um conjunto."""
    if e0 == meta:
        return "OK", []
    pf = {e0: None}
    pb = {meta: None}
    df = {e0: 0}
    db = {meta: 0}
    ff, fb = [e0], [meta]
    while ff and fb:
        fwd = len(ff) <= len(fb)
        frente, pa, po, da, do = (
            ff, pf, pb, df, db) if fwd else (fb, pb, pf, db, df)
        novo = []
        melhor = None
        for s in frente:
            c.visitar()
            for t, m in tab.sucessores(s):
                c.gerados += 1
                if t in pa:
                    continue
                pa[t] = (s, m)
                da[t] = da[s] + 1
                novo.append(t)
                if t in po:
                    tot = da[t] + do[t]
                    if melhor is None or tot < melhor[0]:
                        melhor = (tot, t)
        if fwd:
            ff = novo
        else:
            fb = novo
        c.front_max = max(c.front_max, len(ff) + len(fb))
        c.armaz = len(pf) + len(pb)
        if melhor is not None:
            x = melhor[1]
            cam = reconstruir(pf, x)               # e0 -> x
            n = x
            # x -> meta (movimentos invertidos)
            while pb[n] is not None:
                ant, m = pb[n]
                cam.append((m[0], m[2], m[1]))
                n = ant
            return "OK", cam
    return "FALHA", None


def estados_resolvidos(tab, e0):
    """Todos os estados-objetivo da componente conexa de e0 (BFS simples, ordem determinista)."""
    vistos = {e0}
    fila = [e0]
    for s in fila:                                 # a lista cresce durante a iteracao
        for t, _ in tab.sucessores(s):
            if t not in vistos:
                vistos.add(t)
                fila.append(t)
    return [s for s in fila if tab.objetivo(s)]


def executar(nome, fn, tab, e0, limite=TIMEOUT_S, max_exp=None, **kw):
    """Roda um algoritmo com timeout/orcamento e devolve um dict de metricas."""
    c = Contador(limite, max_exp)
    try:
        st, cam = fn(tab, e0, c, **kw)
    except Timeout:
        st, cam = "TIMEOUT", None
    t = time.perf_counter() - c.t0
    return {
        "alg": nome, "status": st, "caminho": cam,
        "movs": None if cam is None else len(cam),
        "cel": None if cam is None else sum(custo_mov(m, "celulas") for m in cam),
        "exp": c.expandidos, "ger": c.gerados, "front": c.front_max,
        "armaz": c.armaz, "tempo": t,
    }


# ==========================================================================
# Gerador de instancias por busca reversa (offline; usado por --gerar)
# ==========================================================================
def analisar_layout(linhas, cap=60000):
    """Enumera a componente conexa do layout e calcula a distancia de cada estado ao
    conjunto de estados resolvidos (BFS multi-fonte). Retorna None se nao ha solucao
    ou se a componente excede `cap`."""
    tab, e0 = Tabuleiro.de_texto(linhas)
    vistos = {e0}
    fila = [e0]
    i = 0
    while i < len(fila):
        for t, _ in tab.sucessores(fila[i]):
            if t not in vistos:
                vistos.add(t)
                fila.append(t)
        i += 1
        if len(fila) > cap:
            return None
    objs = [s for s in fila if tab.objetivo(s)]
    if not objs:
        return None
    dist = {s: 0 for s in objs}
    origem = {s: s for s in objs}
    q = deque(objs)
    while q:
        s = q.popleft()
        for t, _ in tab.sucessores(s):
            if t not in dist:
                dist[t] = dist[s] + 1
                origem[t] = origem[s]
                q.append(t)
    return tab, dist, origem


def _colocar(rng, grade, letra):
    """Tenta colocar um carro aleatorio (nao vermelho) na grade; devolve True se conseguiu."""
    for _ in range(40):
        h = rng.random() < 0.5
        cp = 2 if rng.random() < 0.7 else 3
        fx = rng.randrange(LADO)
        if h and fx == LINHA_SAIDA:
            continue
        p = rng.randrange(LADO - cp + 1)
        cels = [(fx, p + j) if h else (p + j, fx) for j in range(cp)]
        if all(grade[r][cc] == "." for r, cc in cels):
            for r, cc in cels:
                grade[r][cc] = letra
            return True
    return False


def layout_aleatorio(rng, n):
    """Layout aleatorio: vermelho na linha de saida (fora da posicao final) e n tentativas de carros."""
    grade = [["."] * LADO for _ in range(LADO)]
    p = rng.randrange(0, POS_OBJETIVO)
    grade[LINHA_SAIDA][p] = grade[LINHA_SAIDA][p + 1] = "X"
    for k in range(n):
        _colocar(rng, grade, chr(ord("A") + k))
    return ["".join(l) for l in grade]


def mutar(rng, linhas):
    """Perturba o layout: realoca 1-2 carros, ou remove um, ou adiciona um novo."""
    grade = [list(l) for l in linhas]
    letras = sorted({ch for l in linhas for ch in l if ch not in ".X"})
    op = rng.random()
    # adiciona um carro (letra livre)
    if op < 0.15 and len(letras) < 13:
        livre = next(
            ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWZ" if ch not in letras)
        _colocar(rng, grade, livre)
        return ["".join(l) for l in grade]
    for ch in rng.sample(letras, min(len(letras), rng.choice([1, 1, 2]))):
        for r in range(LADO):
            for cc in range(LADO):
                if grade[r][cc] == ch:
                    grade[r][cc] = "."
        # realoca (se falhar, o carro some)
        _colocar(rng, grade, ch)
    return ["".join(l) for l in grade]


def gerar_instancias(alvos, seed=7, orcamento_s=240.0):
    """Hill climbing sobre layouts maximizando a profundidade do estado mais dificil.
    Devolve {profundidade: (tab_inicial, tab_meta, n_estados)} e imprime progresso."""
    rng = random.Random(seed)
    achados = {}
    t0 = time.perf_counter()

    def registrar(res):
        """Guarda, por profundidade, o primeiro estado inedito; devolve a profundidade maxima."""
        tab, dist, origem = res
        por_prof = {}
        for s, d in dist.items():
            por_prof.setdefault(d, s)
        for d, s in por_prof.items():
            if d not in achados:
                achados[d] = (tab.texto(s), tab.texto(origem[s]), len(dist))
        return max(dist.values())

    while time.perf_counter() - t0 < orcamento_s:
        lay = layout_aleatorio(rng, rng.randint(10, 13))
        res = analisar_layout(lay)
        if res is None:
            continue
        atual = registrar(res)
        for _ in range(300):
            cand = mutar(rng, lay)
            r2 = analisar_layout(cand)
            if r2 is None:
                continue
            d2 = registrar(r2)
            if d2 >= atual:
                lay, atual = cand, d2
            if time.perf_counter() - t0 > orcamento_s:
                break
    return achados


# ==========================================================================
# Figuras
# ==========================================================================
def desenhar_tabuleiro(ax, tab, e, titulo=""):
    """Desenha o estado `e` em `ax`: carros coloridos, vermelho = X, seta = saida."""
    cmap = plt.get_cmap("tab20")
    # paleta sem tons de vermelho (reservado ao carro X)
    paleta = [cmap(k) for k in range(20) if k not in (6, 7)]
    ax.set_xlim(0, LADO + 0.35)
    ax.set_ylim(LADO, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), LADO, LADO,
                 fc="#f2f2f2", ec="black", lw=1.5))
    for k in range(1, LADO):
        ax.plot([k, k], [0, LADO], color="#d0d0d0", lw=0.5)
        ax.plot([0, LADO], [k, k], color="#d0d0d0", lw=0.5)
    for i, p in enumerate(e):
        cor = "#d62728" if i == 0 else paleta[(i * 5) % len(paleta)]
        cels = tab.celulas(i, p)
        r0, c0 = cels[0] // LADO, cels[0] % LADO
        w, h = (tab.comp[i], 1) if tab.horiz[i] else (1, tab.comp[i])
        ax.add_patch(Rectangle((c0 + 0.06, r0 + 0.06), w - 0.12, h - 0.12,
                               fc=cor, ec="black", lw=0.8))
        ax.text(c0 + w / 2, r0 + h / 2, tab.letras[i], ha="center", va="center",
                fontsize=8, color="white", weight="bold")
    ax.add_patch(FancyArrow(LADO, LINHA_SAIDA + 0.5, 0.3, 0, width=0.12,
                            head_width=0.4, head_length=0.15, color="#d62728"))
    ax.set_title(titulo, fontsize=8)


def fig_solucao(inst, tab, e0, caminho, arq):
    """Figura com a sequencia de tabuleiros da solucao (um painel por movimento)."""
    estados = [e0]
    for m in caminho:
        estados.append(
            [t for t, mm in tab.sucessores(estados[-1]) if mm == m][0])
    n = len(estados)
    cols = 5
    lin = (n + cols - 1) // cols
    fig, axs = plt.subplots(lin, cols, figsize=(2.3 * cols, 2.3 * lin))
    axs = axs.ravel()
    for k, ax in enumerate(axs):
        if k < n:
            if k == 0:
                tit = "inicio"
            else:
                i, de, para = caminho[k - 1]
                tit = f"{k}: {tab.letras[i]} {de}->{para}"
            desenhar_tabuleiro(ax, tab, estados[k], tit)
        else:
            ax.axis("off")
    fig.suptitle(f"Solucao otima da instancia {inst['nome']} ({inst['otimo']} movimentos, BFS)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / arq, dpi=130)
    plt.close(fig)


ALGS_ORDEM = ["BFS", "DFS",
              "DLS(L=%d)" % LIMITE_DLS, "IDDFS", "IDDFS-puro", "UCS", "Bidir."]
CORES = {"BFS": "#1f77b4", "DFS": "#ff7f0e", "DLS(L=%d)" % LIMITE_DLS: "#2ca02c",
         "IDDFS": "#d62728", "IDDFS-puro": "#9467bd", "UCS": "#8c564b", "Bidir.": "#17becf"}


def fig_barras(resultados, campo, ylabel, titulo, arq):
    """Barras agrupadas (eixo y logaritmico). Barra hachurada = parcial (TIMEOUT ou CORTE)."""
    nomes = [i["nome"] for i in INSTANCIAS]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / len(ALGS_ORDEM)
    for k, alg in enumerate(ALGS_ORDEM):
        xs, ys, hs = [], [], []
        for j, nm in enumerate(nomes):
            r = resultados[(nm, alg)]
            xs.append(j + (k - len(ALGS_ORDEM) / 2 + 0.5) * w)
            ys.append(max(r[campo], 1))
            hs.append("//" if r["status"] in ("TIMEOUT", "CORTE") else None)
        bars = ax.bar(xs, ys, w, label=alg,
                      color=CORES[alg], ec="black", lw=0.4)
        for b, h in zip(bars, hs):
            if h:
                b.set_hatch(h)
                b.set_alpha(0.55)
    ax.set_yscale("log")
    ax.set_xticks(range(len(nomes)))
    ax.set_xticklabels(
        [f"{i['nome']}\n(otimo {i['otimo']})" for i in INSTANCIAS], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 6)   # folga para a legenda
    ax.legend(ncol=7, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGS / arq, dpi=130)
    plt.close(fig)


# ==========================================================================
# Experimentos
# ==========================================================================
SAIDA = []


def log(msg=""):
    """Imprime e guarda a linha para gravar em figs/02_saida.txt."""
    print(msg)
    SAIDA.append(msg)


def celula(r, campo):
    """Formata um campo de resultado para as tabelas (marca TIMEOUT/CORTE)."""
    if campo == "movs":
        return r["status"] if r["status"] != "OK" else str(r["movs"])
    if campo == "tempo":
        return f"{r['tempo']:.2f}"
    v = r[campo]
    return f"{v}*" if r["status"] in ("TIMEOUT", "CORTE") else str(v)


def tabela(titulo, resultados, campo):
    """Imprime uma tabela instancia x algoritmo para o campo pedido."""
    log(f"\n{titulo}")
    cab = f"{'inst':<6}{'otimo':>6}" + "".join(f"{a:>12}" for a in ALGS_ORDEM)
    log(cab)
    log("-" * len(cab))
    for inst in INSTANCIAS:
        lin = f"{inst['nome']:<6}{inst['otimo']:>6}"
        for a in ALGS_ORDEM:
            lin += f"{celula(resultados[(inst['nome'], a)], campo):>12}"
        log(lin)


def main():
    """Roda todos os algoritmos nas 10 instancias, valida, confere asserts e gera as figuras."""
    FIGS.mkdir(exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t_ini = time.perf_counter()
    log("Projeto 2 - Busca nao informada (Rush Hour 6x6)")
    log(f"Timeout por execucao: {TIMEOUT_S}s (excecoes: {LIMITES}; orcamento de nos: {ORCAMENTO_NOS}) | DLS: L={LIMITE_DLS} | custo: 1 por movimento")

    # --- prepara instancias ---
    preparadas = []
    for inst in INSTANCIAS:
        tab, e0 = Tabuleiro.de_texto(inst["tab"])
        _, meta = Tabuleiro.de_texto(inst["meta"])
        assert tab.objetivo(meta) and not tab.objetivo(e0)
        conferir_sucessores(tab, e0)
        preparadas.append((inst, tab, e0, meta))

    log("\nInstancias:")
    for inst, tab, e0, _ in preparadas:
        log(f"  {inst['nome']}: {len(e0)} carros, otimo conhecido = {inst['otimo']} movimentos")

    algs = [
        ("BFS", bfs, {}),
        ("DFS", dfs, {}),
        ("DLS(L=%d)" % LIMITE_DLS, dls_fixa, {}),
        ("IDDFS", iddfs, {}),
        ("IDDFS-puro", iddfs_puro, {}),
        ("UCS", ucs, {}),
        ("Bidir.", bidirecional, "meta"),
    ]

    resultados = {}
    for inst, tab, e0, meta in preparadas:
        for nome, fn, kw in algs:
            kwargs = {"meta": meta} if kw == "meta" else kw
            r = executar(nome, fn, tab, e0, limite=LIMITES.get(nome, TIMEOUT_S),
                         max_exp=ORCAMENTO_NOS.get(nome), **kwargs)
            if r["status"] == "OK":
                validar(tab, e0, r["caminho"],
                        meta if nome == "Bidir." else None)
            resultados[(inst["nome"], nome)] = r
        log(f"  [{inst['nome']}] concluida ({time.perf_counter() - t_ini:.0f}s desde o inicio)")

    tabela(
        "Comprimento da solucao (movimentos)  [TIMEOUT/CORTE = sem solucao]", resultados, "movs")
    tabela(
        "Nos expandidos (visitados)  [* = parcial ate TIMEOUT/CORTE]", resultados, "exp")
    tabela("Tamanho maximo da fronteira  [* = parcial]", resultados, "front")
    tabela("Estados armazenados (max)  [* = parcial]", resultados, "armaz")
    tabela("Tempo (s)", resultados, "tempo")

    # --- variante de custo: 1 por celula (UCS) ---
    log("\nVariante de custo: UCS com custo = celulas percorridas vs. solucao da BFS")
    log(f"{'inst':<6}{'BFS movs':>10}{'BFS cel':>9}{'UCS-cel movs':>14}{'UCS-cel cel':>13}{'exp':>8}")
    for inst, tab, e0, _ in preparadas:
        rb = resultados[(inst["nome"], "BFS")]
        ru = executar("UCS-cel", ucs_celulas, tab, e0)
        validar(tab, e0, ru["caminho"])
        log(f"{inst['nome']:<6}{rb['movs']:>10}{rb['cel']:>9}{ru['movs']:>14}{ru['cel']:>13}{ru['exp']:>8}")
        assert ru["cel"] <= rb["cel"], "UCS por celulas deve minimizar celulas"
        assert ru["movs"] >= rb["movs"], "BFS minimiza movimentos"
        resultados[(inst["nome"], "UCS-cel")] = ru

    # --- bidirecional SEM a meta privilegiada: meta = estado resolvido qualquer (5 sorteados, seed 0) ---
    log("\nBidirecional com meta ARBITRARIA (5 estados resolvidos sorteados, seed 0) vs. meta mais proxima")
    log(f"{'inst':<6}{'resolvidos':>11}{'otimo':>7}{'movs medio':>12}{'movs max':>10}{'nos medio':>11}{'nos meta prox.':>15}{'nos BFS':>9}")
    rng = random.Random(0)
    for inst, tab, e0, _ in preparadas:
        alvos = estados_resolvidos(tab, e0)
        rs = []
        for meta_q in rng.sample(alvos, 5):
            rq = executar("Bidir.-q", bidirecional, tab, e0, meta=meta_q)
            assert rq["status"] == "OK"
            validar(tab, e0, rq["caminho"], meta_q)
            assert rq["movs"] >= inst["otimo"], "caminho ate uma meta nao pode ser menor que o otimo"
            rs.append(rq)
        rb, rm = resultados[(inst["nome"], "BFS")], resultados[(inst["nome"], "Bidir.")]
        log(f"{inst['nome']:<6}{len(alvos):>11}{inst['otimo']:>7}{sum(r['movs'] for r in rs) / 5:>12.1f}"
            f"{max(r['movs'] for r in rs):>10}{sum(r['exp'] for r in rs) / 5:>11.0f}{rm['exp']:>15}{rb['exp']:>9}")

    # --- asserts de autoverificacao ---
    for inst, tab, e0, meta in preparadas:
        nm, ot = inst["nome"], inst["otimo"]
        def R(a):
            """Atalho: resultado do algoritmo `a` na instancia corrente."""
            return resultados[(nm, a)]
        for a in ("BFS", "UCS", "Bidir."):
            assert R(a)["status"] == "OK" and R(a)["movs"] == ot, (nm, a, R(a)["status"], R(a)["movs"])
        # IDDFS: pode dar TIMEOUT (reexpansoes), mas quando termina deve ser otima
        assert R("IDDFS")["status"] in ("OK", "TIMEOUT")
        if R("IDDFS")["status"] == "OK":
            assert R("IDDFS")["movs"] == ot, (nm, "IDDFS")
        assert R("DFS")["status"] == "OK" and R("DFS")["movs"] >= ot
        if R("DLS(L=%d)" % LIMITE_DLS)["status"] == "OK":
            assert R("DLS(L=%d)" % LIMITE_DLS)["movs"] >= ot
        else:
            assert ot > LIMITE_DLS or R("DLS(L=%d)" % LIMITE_DLS)[
                "status"] == "TIMEOUT"
        if R("IDDFS-puro")["status"] == "OK":
            assert R("IDDFS-puro")["movs"] == ot
    n_id = sum(resultados[(i["nome"], "IDDFS")]["status"] == "OK" for i in INSTANCIAS)
    assert n_id >= 5, "IDDFS deveria resolver ao menos as instancias faceis"
    razao_dfs = [resultados[(i["nome"], "DFS")]["movs"] / i["otimo"] for i in INSTANCIAS]
    log(f"\nAsserts OK: BFS = UCS = Bidir. = otimo em todas as instancias; IDDFS = otimo nas {n_id} "
        "em que terminou; DFS/DLS >= otimo.")
    log("Comprimento DFS / otimo por instancia: " + ", ".join(f"{r:.1f}" for r in razao_dfs))

    # --- estatisticas resumo ---
    log("\nResumo: razao (nos expandidos)/(nos expandidos da BFS), quando terminou")
    for a in ALGS_ORDEM:
        rz = [resultados[(i['nome'], a)]["exp"] / resultados[(i['nome'], 'BFS')]["exp"]
              for i in INSTANCIAS if resultados[(i['nome'], a)]["status"] == "OK"]
        to = sum(1 for i in INSTANCIAS if resultados[(
            i['nome'], a)]["status"] == "TIMEOUT")
        ct = sum(1 for i in INSTANCIAS if resultados[(
            i['nome'], a)]["status"] == "CORTE")
        med = sum(rz) / len(rz) if rz else float("nan")
        log(f"  {a:<12} concluidas={len(rz):>2}  TIMEOUT={to}  CORTE={ct}  razao media={med:.2f}")

    # --- figuras ---
    cand = [p for p in preparadas if p[0]["otimo"] <= 15]
    inst, tab, e0, _ = max(cand, key=lambda p: p[0]["otimo"])
    fig_solucao(inst, tab, e0, resultados[(
        inst["nome"], "BFS")]["caminho"], "02_solucao.png")
    fig_barras(resultados, "exp", "nos expandidos (escala log)",
               "Nos expandidos por algoritmo e instancia (hachurado = parcial: TIMEOUT ou CORTE)",
               "02_expandidos.png")
    fig_barras(resultados, "front", "tamanho maximo da fronteira (escala log)",
               "Fronteira maxima (proxy de memoria) por algoritmo e instancia (hachurado = parcial)",
               "02_fronteira.png")
    log(
        f"\nFiguras: 02_solucao.png (instancia {inst['nome']}), 02_expandidos.png, 02_fronteira.png")
    log(f"Tempo total: {time.perf_counter() - t_ini:.1f}s")

    (FIGS / "02_saida.txt").write_text("\n".join(SAIDA) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--gerar" in sys.argv:
        alvos = [5, 8, 11, 14, 17, 20, 23, 26, 29, 32]
        ach = gerar_instancias(alvos)
        for d in sorted(ach):
            print(d, ach[d][2])
            for a, b in zip(ach[d][0], ach[d][1]):
                print("   ", a, b)
    else:
        main()
