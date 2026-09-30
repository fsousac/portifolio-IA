"""
Projeto 5 - CSP: geracao de cruzadinha (palavras cruzadas) como Problema de
Satisfacao de Restricoes.

PROBLEMA
    Dada uma grade-modelo (template ASCII com '#' = casa bloqueada e '.' = casa
    livre) e um dicionario de palavras do portugues (sem acentos), preencher
    TODAS as casas livres de modo que cada palavra horizontal/vertical (corrida
    de 2 ou mais casas livres consecutivas) seja uma palavra do dicionario.

FORMULACAO COMO CSP
    Variaveis  : os "slots" (corridas horizontais e verticais de tamanho >= 2).
    Dominios   : palavras do dicionario com o mesmo comprimento do slot.
    Restricoes : todas BINARIAS
                 (a) cruzamento: se os slots X e Y compartilham uma casa, a letra
                     de X naquela posicao deve ser igual a letra de Y;
                 (b) diferenca: palavras de slots distintos de mesmo comprimento
                     nao podem se repetir (all-different decomposto em pares).

ALGORITMOS IMPLEMENTADOS A MAO (sem bibliotecas de CSP/IA)
    CSP generico (classe CSP + Restricao binaria com predicado ok(a, b)):
      - Backtracking puro (ordem estatica de variaveis, ordem estatica de valores)
      - + MRV (minimum remaining values)      + desempate por GRAU
      - + LCV (least constraining value)
      - + Forward Checking (FC)
      - AC-3 (pre-processamento) e MAC (Maintaining Arc Consistency = AC-3 a cada
        atribuicao)
      - Min-Conflicts (busca local, com ruido; avalia todos os valores do dominio)
    Os mesmos algoritmos rodam sobre a cruzadinha e sobre um problema de outra
    natureza (8 rainhas) para mostrar que o nucleo e realmente generico.

METRICAS
    nos        = numero de atribuicoes consistentes tentadas (extensoes do estado);
    backtracks = numero de atribuicoes desfeitas (falhas de subarvore);
    trabalho   = valores de dominio inspecionados (proxy deterministico do custo);
    tempo      = segundos de CPU (process_time; resolucao de ~15 ms no Windows).
    Cada execucao tem limite de tempo e de nos; ao estourar registra-se TIMEOUT
    (os valores de nos/backtracks nesse caso sao LIMITES INFERIORES).

VERIFICACAO
    Toda solucao passa por um verificador INDEPENDENTE (verificar_solucao) que
    reescreve a grade letra a letra, le de novo as corridas com expressao regular e
    confere: cruzamentos, palavras no dicionario e palavras todas distintas.

Execucao: python 05_csp.py   (gera figs/05_*.png e figs/05_saida.txt)
"""

import math
import random
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela (salva apenas arquivos); antes do pyplot
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# Parametros globais
# ----------------------------------------------------------------------------
PASTA = Path(__file__).resolve().parent
FIGS = PASTA / "figs"
ARQ_DICIONARIO = PASTA / "palavras_pt.txt"
SEMENTE = 2024          # semente global (reprodutibilidade)
LIMITE_TRAB = 30_000_000 # orcamento de "trabalho" por execucao (valores de dominio inspecionados)
LIMITE_TEMPO_MURO = 60.0  # rede de seguranca (segundos de CPU); nao deve ser atingida
LIMITE_NOS = 400_000    # limite de nos por execucao
PASSOS_MC = 6000        # orcamento de passos do Min-Conflicts
N_GRADES_DENS = 12      # grades sorteadas por densidade (experimento de densidade)
LIMITE_TRAB_DENS = 5_000_000   # orcamento de trabalho por grade no experimento de densidade
GRADE_DICIONARIO = "9x9-A"   # grade usada no experimento de tamanho do dicionario
REPS_DIC = 10           # repeticoes por fracao do dicionario
LIMITE_TRAB_DIC = 5_000_000    # orcamento de trabalho por execucao no experimento de dicionario
AMOSTRA_MC = 500        # candidatos avaliados por passo no Min-Conflicts (>= maior dominio: avalia TODOS)
RUIDO_MC = 0.04         # probabilidade de passo aleatorio no Min-Conflicts

# ----------------------------------------------------------------------------
# Grades-modelo ('#' bloqueada, '.' livre), duas por tamanho. Foram escolhidas
# (fora deste script, por amostragem de grades simetricas aleatorias) entre as
# que possuem solucao com o dicionario embutido e comprimento maximo de slot <= 6;
# ha muitas outras grades da mesma densidade SEM solucao (ver relatorio).
# ----------------------------------------------------------------------------
TEMPLATES = {
    "5x5-A": ["##...", ".....", ".....", ".....", "...##"],
    "5x5-B": ["....#", ".....", "#...#", ".....", "#...."],
    "7x7-A": [".....#.", "..#....", "#......", "..###..", "......#", "....#..", ".#....."],
    "7x7-B": ["##.....", "......#", "....#..", "...#...", "..#....", "#......", ".....##"],
    "9x9-A": ["...#..#..", ".....#.#.", "....#..#.", "..#......", "#...#...#",
              "......#..", ".#..#....", ".#.#.....", "..#..#..."],
    "9x9-B": ["..#......", ".....##..", "...#..#.#", ".#......#", "....#....",
              "#......#.", "#.#..#...", "..##.....", "......#.."],
    "11x11-A": [".....#.....", "......#....", ".#...#.....", "..#....#...", "......#..#.",
                "#.##...##.#", ".#..#......", "...#....#..", ".....#...#.", "....#......",
                ".....#....."],
    "11x11-B": [".#.....#...", "......#....", "..##....#.#", "...#.#.....", "....##.....",
                "##...#...##", ".....##....", ".....#.#...", "#.#....##..", "....#......",
                "...#.....#."],
}


# ----------------------------------------------------------------------------
# Nucleo generico: restricoes binarias e CSP
# ----------------------------------------------------------------------------
class Restricao:
    """Restricao binaria orientada X -> Y. Subclasses implementam ok(a, b),
    onde a e valor de X e b e valor de Y. Os demais metodos possuem versao
    generica (baseada apenas em ok) e podem ser especializados por eficiencia."""

    # False = restricao "fraca" (so age com dominios unitarios)
    informativa = True
    # True = so vale a pena revisar quando |D(Y)| == 1
    somente_unitario = False

    def ok(self, a, b):
        """Predicado da restricao: True se (X = a, Y = b) e permitido."""
        raise NotImplementedError

    def revisar(self, dx, dy):
        """Passo REVISE do AC-3: devolve os valores de dx com algum suporte em dy."""
        return [a for a in dx if any(self.ok(a, b) for b in dy)]

    def filtrar(self, a, dy):
        """Valores de dy compativeis com X = a (usado no Forward Checking)."""
        return [b for b in dy if self.ok(a, b)]

    def suporte(self, dy):
        """Devolve f(a) = quantos valores de dy sao compativeis com X = a (LCV)."""
        return lambda a: sum(1 for b in dy if self.ok(a, b))

    def inversa(self):
        """Mesma restricao vista na direcao Y -> X."""
        return Predicado(lambda b, a: self.ok(a, b))


class Predicado(Restricao):
    """Restricao definida por uma funcao qualquer fn(a, b) -> bool."""

    def __init__(self, fn):
        """Guarda a funcao fn(a, b) -> bool."""
        self.fn = fn

    def ok(self, a, b):
        """Delega ao predicado fornecido."""
        return self.fn(a, b)


class Cruzamento(Restricao):
    """Letra i da palavra de X == letra j da palavra de Y. Se `distintas`, as
    palavras tambem devem ser diferentes (slots de mesmo comprimento)."""

    def __init__(self, i, j, distintas):
        """Guarda as posicoes i (em X) e j (em Y) da casa comum e se as palavras devem diferir."""
        self.i, self.j, self.distintas = i, j, distintas

    def ok(self, a, b):
        """Mesma letra na casa comum (e palavras distintas, se exigido)."""
        return a[self.i] == b[self.j] and not (self.distintas and a == b)

    def revisar(self, dx, dy):
        """Passo REVISE especializado: filtra dx pelas letras que dy oferece na posicao j."""
        # Especializacao O(|dx|+|dy|): conjunto de letras disponiveis em dy na
        # posicao j; a so tem suporte se a[i] estiver nesse conjunto.
        i, j = self.i, self.j
        letras = set(map(itemgetter(j), dy))
        novo = [a for a in dx if a[i] in letras]
        if not self.distintas:
            return novo
        # Caso raro (mesmo comprimento): a palavra a e o UNICO suporte de si mesma
        # (a esta em dy, a[i] == a[j] e nenhuma outra palavra de dy tem essa letra
        # na posicao j). Como palavras iguais sao proibidas, ela perde o suporte.
        conj = cont = None
        saida = []
        for a in novo:
            if a[i] == a[j]:
                if conj is None:              # constroi (uma vez) so quando necessario
                    conj, cont = set(dy), Counter(map(itemgetter(j), dy))
                if a in conj and cont[a[i]] == 1:
                    continue
            saida.append(a)
        return saida

    def filtrar(self, a, dy):
        """Forward checking especializado: valores de dy com a letra certa (e != a, se exigido)."""
        c, j = a[self.i], self.j
        if self.distintas:
            return [b for b in dy if b[j] == c and b != a]
        return [b for b in dy if b[j] == c]

    def suporte(self, dy):
        """Suporte de cada valor = quantos valores de dy tem a mesma letra na casa comum (LCV)."""
        cnt = Counter(map(itemgetter(self.j), dy))
        i = self.i
        return lambda a: cnt[a[i]]

    def inversa(self):
        """Restricao na direcao oposta (Y -> X)."""
        return Cruzamento(self.j, self.i, self.distintas)


class Diferente(Restricao):
    """a != b (all-different decomposto em pares). Fraca: so poda com |D(Y)| = 1."""

    informativa = False
    somente_unitario = True

    def ok(self, a, b):
        """a e b devem ser diferentes."""
        return a != b

    def revisar(self, dx, dy):
        """So poda quando dy tem um unico valor b: remove b de dx."""
        if len(dy) == 1:
            b = dy[0]
            return [a for a in dx if a != b]
        return dx

    def filtrar(self, a, dy):
        """Valores de dy diferentes de a."""
        return [b for b in dy if b != a]

    def suporte(self, dy):
        """Aproximacao para LCV (nao usada: restricao nao informativa)."""
        return lambda a: len(dy) - 1

    def inversa(self):
        """Simetrica: a inversa e ela mesma."""
        return self


class CSP:
    """CSP generico: variaveis, dominios (listas) e restricoes binarias."""

    def __init__(self, dominios):
        """Cria o CSP a partir de {variavel: lista de valores}."""
        self.dominios = {v: list(d) for v, d in dominios.items()}
        self.variaveis = list(self.dominios)          # ordem estatica
        # viz[x] = [(y, restricao x->y)]
        self.viz = {v: [] for v in self.variaveis}
        # (x, y) -> restricao x->y
        self.arco = {}

    def adicionar(self, x, y, r):
        """Registra a restricao r (orientada x->y) e sua inversa (y->x)."""
        ri = r.inversa()
        self.viz[x].append((y, r))
        self.viz[y].append((x, ri))
        self.arco[(x, y)] = r
        self.arco[(y, x)] = ri

    def finalizar(self):
        """Ordena vizinhos (informativos primeiro: falham mais cedo) e prepara graus."""
        for v in self.variaveis:
            self.viz[v].sort(key=lambda t: not t[1].informativa)
        self.viz_inf = {v: [y for y, r in self.viz[v] if r.informativa]
                        for v in self.variaveis}
        return self


# ----------------------------------------------------------------------------
# Busca sistematica: backtracking + heuristicas + inferencia (FC / MAC)
# ----------------------------------------------------------------------------
@dataclass
class Config:
    """Combinacao de tecnicas. inferencia in {'nenhuma', 'fc', 'mac'}."""
    nome: str
    mrv: bool = False
    grau: bool = False
    lcv: bool = False
    inferencia: str = "nenhuma"
    ac3_previo: bool = False


@dataclass
class Resultado:
    """Saida de uma execucao: status, atribuicao e contadores de custo."""
    status: str            # OK | TIMEOUT (orcamento esgotado) | LIMITE_NOS | SEM_SOLUCAO
    atrib: dict = None
    nos: int = 0
    backtracks: int = 0
    tempo: float = 0.0     # segundos de CPU (process_time)
    trabalho: int = 0      # valores de dominio inspecionados (proxy deterministico de custo)
    extra: dict = field(default_factory=dict)


class Limite(Exception):
    """Levantada quando estoura o orcamento de trabalho, de nos ou de tempo."""

    def __init__(self, motivo):
        """Guarda o motivo do estouro de limite (TIMEOUT ou LIMITE_NOS)."""
        super().__init__(motivo)
        self.motivo = motivo


def ac3(csp, doms, fila_inicial, livre, checar):
    """AC-3. Poda `doms` (dict de listas; substitui listas, nao as altera).
    fila_inicial: arcos (x, y) a revisar; livre(z): True se z pode ser podada
    (no MAC, so variaveis nao atribuidas). Devolve False se algum dominio esvaziar."""
    fila = deque(fila_inicial)
    na_fila = set(fila)
    while fila:
        x, y = fila.popleft()
        na_fila.discard((x, y))
        r = csp.arco[(x, y)]
        dx = doms[x]
        # custo = valores inspecionados (restricoes "fracas" sao O(1) quando |D(Y)| > 1)
        checar(1 if (r.somente_unitario and len(doms[y]) != 1) else len(dx) + len(doms[y]))
        novo = r.revisar(dx, doms[y])
        if len(novo) < len(dx):
            if not novo:
                return False
            doms[x] = novo
            for z, _ in csp.viz[x]:
                if z == y or not livre(z):
                    continue
                if csp.arco[(z, x)].somente_unitario and len(novo) > 1:
                    continue          # revisao inutil: X ainda tem varios valores
                if (z, x) not in na_fila:
                    na_fila.add((z, x))
                    fila.append((z, x))
    return True


class Buscador:
    """Backtracking recursivo parametrizado por Config."""

    def __init__(self, csp, cfg, limite_trab=LIMITE_TRAB, limite_nos=LIMITE_NOS,
                 limite_tempo=LIMITE_TEMPO_MURO):
        """Guarda o CSP, a configuracao, os limites e zera os contadores."""
        self.csp, self.cfg = csp, cfg
        self.limite_trab, self.limite_nos = limite_trab, limite_nos
        self.limite_tempo = limite_tempo
        self.trab = 0
        self.nos = 0
        self.backtracks = 0
        self.t0 = 0.0

    # -- limite de recursos -------------------------------------------------
    def checar(self, custo=0):
        """Contabiliza `custo` (valores de dominio inspecionados) e levanta Limite se
        o orcamento de trabalho (deterministico), de nos ou de tempo (rede de
        seguranca) for excedido."""
        self.trab += custo
        if self.trab > self.limite_trab:
            raise Limite("TIMEOUT")
        if self.nos > self.limite_nos:
            raise Limite("LIMITE_NOS")
        if custo and time.process_time() - self.t0 > self.limite_tempo:
            raise Limite("TIMEOUT")

    # -- consistencia com atribuicao parcial (sem inferencia) -----------------
    def consistente(self, x, a, atrib):
        """True se X = a e compativel com todas as variaveis ja atribuidas."""
        for y, r in self.csp.viz[x]:
            if y in atrib and not r.ok(a, atrib[y]):
                return False
        return True

    def legais(self, x, atrib):
        """Valores de X consistentes com a atribuicao parcial (checagem sem inferencia)."""
        self.checar(3 * len(self.csp.dominios[x]))      # ~3 restricoes checadas por valor
        return [a for a in self.csp.dominios[x] if self.consistente(x, a, atrib)]

    # -- escolha de variavel ---------------------------------------------------
    def grau(self, v, atrib):
        """Grau dinamico: numero de vizinhos informativos ainda nao atribuidos."""
        return sum(1 for y in self.csp.viz_inf[v] if y not in atrib)

    def escolher_variavel(self, doms, atrib):
        """Escolhe a proxima variavel: ordem estatica, MRV e/ou grau (desempate por grau)."""
        cfg = self.cfg
        livres = [v for v in self.csp.variaveis if v not in atrib]
        if not (cfg.mrv or cfg.grau):
            return livres[0]                       # ordem estatica
        if cfg.mrv and cfg.inferencia == "nenhuma":
            tam = {v: len(self.legais(v, atrib))
                   for v in livres}  # valores legais reais
        else:
            tam = {v: len(doms[v]) for v in livres}
        if cfg.mrv:
            chave = (lambda v: (tam[v], -self.grau(v, atrib))) if cfg.grau \
                else (lambda v: tam[v])
        else:
            def chave(v):
                """Chave de ordenacao: maior grau primeiro (sem MRV)."""
                return -self.grau(v, atrib)
        # empate -> primeiro na ordem estatica
        return min(livres, key=chave)

    # -- ordenacao de valores --------------------------------------------------
    def ordenar_valores(self, x, valores, doms, atrib):
        """Ordena os valores de X: com LCV, os que menos eliminam nos vizinhos vem primeiro."""
        if not self.cfg.lcv:
            return valores
        sup = [(y, self.csp.arco[(x, y)].suporte(doms[y]), len(doms[y]))
               for y in self.csp.viz_inf[x] if y not in atrib]

        self.checar(sum(n for _, _, n in sup) + len(valores) * len(sup))

        def eliminados(a):                          # valores que a "mata" nos vizinhos
            """Quantos valores dos vizinhos livres a seria eliminado (menor = melhor)."""
            return sum(n - f(a) for _, f, n in sup)
        # estavel: mantem ordem em empates
        return sorted(valores, key=eliminados)

    # -- inferencias -----------------------------------------------------------
    def forward_checking(self, x, a, doms, atrib):
        """Forward checking: atribui X = a e filtra os vizinhos livres; None se algum dominio esvaziar."""
        novos = dict(doms)
        novos[x] = [a]
        for y, r in self.csp.viz[x]:
            if y in atrib:
                continue
            # valores de Y compativeis com X = a
            self.checar(len(doms[y]))
            dy = r.filtrar(a, doms[y])
            if len(dy) < len(doms[y]):
                if not dy:
                    return None
                novos[y] = dy
        return novos

    def mac(self, x, a, doms, atrib):
        """MAC: atribui X = a e roda AC-3 sobre os vizinhos livres; None se algum dominio esvaziar."""
        novos = dict(doms)
        novos[x] = [a]
        fila = [(y, x) for y, _ in self.csp.viz[x] if y not in atrib]
        if not ac3(self.csp, novos, fila, lambda z: z not in atrib, self.checar):
            return None
        return novos

    # -- nucleo recursivo --------------------------------------------------------
    def bt(self, atrib, doms):
        """Backtracking recursivo; devolve a atribuicao completa ou None."""
        if len(atrib) == len(self.csp.variaveis):
            return dict(atrib)
        x = self.escolher_variavel(doms, atrib)
        if self.cfg.inferencia == "nenhuma":
            valores = self.legais(x, atrib)
        else:
            valores = doms[x]
        for a in self.ordenar_valores(x, valores, doms, atrib):
            self.checar()
            self.nos += 1
            atrib[x] = a
            if self.cfg.inferencia == "fc":
                novos = self.forward_checking(x, a, doms, atrib)
            elif self.cfg.inferencia == "mac":
                novos = self.mac(x, a, doms, atrib)
            else:
                novos = doms
            if novos is not None:
                r = self.bt(atrib, novos)
                if r is not None:
                    return r
            del atrib[x]
            self.backtracks += 1
        return None

    def resolver(self):
        """Executa a busca (com AC-3 previo, se configurado) e devolve um Resultado."""
        self.t0 = time.process_time()
        doms = {v: list(d) for v, d in self.csp.dominios.items()}
        extra = {}
        try:
            if self.cfg.ac3_previo or self.cfg.inferencia == "mac":
                antes = sum(len(d) for d in doms.values())
                fila = list(self.csp.arco)
                ok = ac3(self.csp, doms, fila, lambda z: True, self.checar)
                depois = sum(len(d) for d in doms.values())
                extra["podados_ac3"] = 1 - depois / antes
                if not ok:
                    return Resultado("SEM_SOLUCAO", None, 0, 0,
                                     time.process_time() - self.t0, self.trab, extra)
            sol = self.bt({}, doms)
            status = "OK" if sol is not None else "SEM_SOLUCAO"
        except Limite as e:
            sol, status = None, e.motivo
        return Resultado(status, sol, self.nos, self.backtracks,
                         time.process_time() - self.t0, self.trab, extra)


# ----------------------------------------------------------------------------
# Busca local: Min-Conflicts
# ----------------------------------------------------------------------------
def min_conflitos(csp, rng, max_passos=PASSOS_MC, amostra=AMOSTRA_MC, ruido=RUIDO_MC):
    """Min-Conflicts: comeca com atribuicao completa aleatoria e, a cada passo,
    escolhe uma variavel em conflito e a muda para o valor (dentre ate `amostra`
    candidatos sorteados + o valor atual; como amostra >= maior dominio, sao todos) que minimiza o numero de
    restricoes violadas; com probabilidade `ruido` faz um passo aleatorio.
    O orcamento e o numero de passos (deterministico); esgota -> TIMEOUT."""
    t0 = time.process_time()
    atrib = {v: rng.choice(csp.dominios[v]) for v in csp.variaveis}

    def conflitos(v, a):
        """Numero de restricoes violadas se a variavel v assumir o valor a."""
        return sum(1 for y, r in csp.viz[v] if not r.ok(a, atrib[y]))

    melhor = len(csp.variaveis)
    for passo in range(max_passos):
        emconf = [v for v in csp.variaveis if conflitos(v, atrib[v]) > 0]
        melhor = min(melhor, len(emconf))
        if not emconf:
            return Resultado("OK", dict(atrib), passo, 0, time.process_time() - t0, 0,
                             {"melhor_conf": 0})
        v = rng.choice(emconf)
        dom = csp.dominios[v]
        if rng.random() < ruido:
            atrib[v] = rng.choice(dom)
            continue
        cands = dom if len(dom) <= amostra else rng.sample(dom, amostra)
        cands = list(cands) + [atrib[v]]
        pontos = [(conflitos(v, a), a) for a in cands]
        m = min(p for p, _ in pontos)
        atrib[v] = rng.choice([a for p, a in pontos if p == m])
    return Resultado("TIMEOUT", None, max_passos, 0, time.process_time() - t0, 0,
                     {"melhor_conf": melhor})


# ----------------------------------------------------------------------------
# Cruzadinha: dicionario, extracao de slots, construcao do CSP, verificacao
# ----------------------------------------------------------------------------
def carregar_dicionario(caminho=ARQ_DICIONARIO):
    """Le o dicionario (linhas iniciadas por # sao comentarios), normaliza,
    remove duplicatas e mantem apenas [a-z]{2,9}. Devolve lista ordenada."""
    palavras = set()
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if linha.strip().startswith("#"):
            continue
        for tok in linha.lower().split():
            if re.fullmatch(r"[a-z]{2,9}", tok):
                palavras.add(tok)
    return sorted(palavras)


def extrair_slots(grade):
    """Devolve lista de (nome, direcao, [(l, c), ...]) das corridas >= 2,
    ordenada por posicao inicial (horizontal antes de vertical)."""
    nl, nc = len(grade), len(grade[0])
    slots = []
    for r in range(nl):
        c = 0
        while c < nc:
            if grade[r][c] == "#":
                c += 1
                continue
            c0 = c
            while c < nc and grade[r][c] != "#":
                c += 1
            if c - c0 >= 2:
                slots.append(
                    (f"H{r}_{c0}", "H", [(r, k) for k in range(c0, c)]))
    for c in range(nc):
        r = 0
        while r < nl:
            if grade[r][c] == "#":
                r += 1
                continue
            r0 = r
            while r < nl and grade[r][c] != "#":
                r += 1
            if r - r0 >= 2:
                slots.append(
                    (f"V{r0}_{c}", "V", [(k, c) for k in range(r0, r)]))
    slots.sort(key=lambda s: (s[2][0], 0 if s[1] == "H" else 1))
    return slots


def dominios_por_comprimento(palavras, rng):
    """Dicionario agrupado por comprimento, com ordem embaralhada (semente fixa):
    a ordem inicial dos valores nao favorece nenhum algoritmo."""
    por_len = {}
    for p in palavras:
        por_len.setdefault(len(p), []).append(p)
    for lst in por_len.values():
        rng.shuffle(lst)
    return por_len


def construir_csp(grade, palavras, rng):
    """Transforma grade + dicionario em CSP (variavel = slot)."""
    slots = extrair_slots(grade)
    por_len = dominios_por_comprimento(palavras, rng)
    dominios = {nome: por_len.get(len(cel), []) for nome, _, cel in slots}
    csp = CSP(dominios)
    # cruzamentos: casa compartilhada entre slots
    posicao = {}
    for nome, _, cel in slots:
        for k, c in enumerate(cel):
            posicao.setdefault(c, []).append((nome, k))
    comp = {nome: len(cel) for nome, _, cel in slots}
    cruzam = set()
    for lst in posicao.values():
        for (n1, i), (n2, j) in ((lst[a], lst[b]) for a in range(len(lst))
                                 for b in range(a + 1, len(lst))):
            csp.adicionar(n1, n2, Cruzamento(i, j, comp[n1] == comp[n2]))
            cruzam.add(frozenset((n1, n2)))
    # diferenca entre slots de mesmo comprimento que NAO se cruzam
    nomes = [s[0] for s in slots]
    for a in range(len(nomes)):
        for b in range(a + 1, len(nomes)):
            if comp[nomes[a]] == comp[nomes[b]] and frozenset((nomes[a], nomes[b])) not in cruzam:
                csp.adicionar(nomes[a], nomes[b], Diferente())
    return csp.finalizar(), slots


def verificar_solucao(grade, solucao, slots, dicionario):
    """Verificador INDEPENDENTE do CSP: escreve as palavras na grade, confere
    conflitos de letras, relê as corridas por regex e valida dicionario e
    unicidade. Devolve True/False (nao levanta)."""
    letras = {}
    for nome, _, cel in slots:
        w = solucao.get(nome)
        if w is None or len(w) != len(cel):
            return False
        for c, ch in zip(cel, w):
            if letras.get(c, ch) != ch:          # cruzamento com letras diferentes
                return False
            letras[c] = ch
    linhas = [[letras.get((r, c), "#" if grade[r][c] == "#" else "?")
               for c in range(len(grade[0]))] for r in range(len(grade))]
    if any("?" in l for l in linhas):            # casa livre sem letra
        return False
    lidas = []
    textos = ["".join(l) for l in linhas]
    textos += ["".join(l[c] for l in linhas) for c in range(len(grade[0]))]
    for t in textos:
        # corridas horizontais e verticais
        lidas += re.findall(r"[^#]{2,}", t)
    dic = set(dicionario)
    return (all(w in dic for w in lidas)         # todas do dicionario
            and len(set(lidas)) == len(lidas)    # todas distintas
            and sorted(lidas) == sorted(solucao.values()))


def estatisticas_grade(grade, slots):
    """Densidade de bloqueios, no. de slots, histograma de comprimentos, grau medio."""
    nl, nc = len(grade), len(grade[0])
    bloq = sum(l.count("#") for l in grade)
    hist = Counter(len(c) for _, _, c in slots)
    pos = Counter(c for _, _, cel in slots for c in cel)
    cruz = sum(1 for v in pos.values() if v == 2)
    return {"bloq_pct": 100 * bloq / (nl * nc), "slots": len(slots),
            "cruzamentos": cruz, "hist": dict(sorted(hist.items())),
            "grau_medio": 2 * cruz / len(slots)}


# ----------------------------------------------------------------------------
# Problema de outra natureza para provar a generalidade: N rainhas
# ----------------------------------------------------------------------------
def csp_rainhas(n):
    """Variavel = coluna i; valor = linha da rainha. Restricoes: linha e diagonais."""
    csp = CSP({i: list(range(n)) for i in range(n)})
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            csp.adicionar(i, j, Predicado(
                lambda a, b, d=d: a != b and abs(a - b) != d))
    return csp.finalizar()


def verifica_rainhas(sol, n):
    """Confere de forma independente uma solucao das N rainhas (linhas e diagonais)."""
    return (len(sol) == n and len(set(sol.values())) == n and
            all(abs(sol[i] - sol[j]) != j - i for i in range(n) for j in range(i + 1, n)))


# ----------------------------------------------------------------------------
# Configuracoes comparadas
# ----------------------------------------------------------------------------
CONFIGS = [
    Config("BT"),
    Config("BT+MRV", mrv=True),
    Config("BT+MRV+Grau", mrv=True, grau=True),
    Config("BT+MRV+FC", mrv=True, inferencia="fc"),
    Config("BT+MRV+Grau+LCV+FC", mrv=True,
           grau=True, lcv=True, inferencia="fc"),
    Config("AC3+BT+MRV+FC", mrv=True, inferencia="fc", ac3_previo=True),
    Config("MAC+MRV+Grau", mrv=True, grau=True, inferencia="mac"),
    Config("MAC+MRV+Grau+LCV", mrv=True,
           grau=True, lcv=True, inferencia="mac"),
]
MAC_PADRAO = CONFIGS[6]


class Tee:
    """Duplica tudo que e impresso no console e em figs/05_saida.txt."""

    def __init__(self, *fluxos):
        """Guarda os fluxos de saida."""
        self.fluxos = fluxos

    def write(self, s):
        """Escreve em todos os fluxos."""
        for f in self.fluxos:
            f.write(s)

    def flush(self):
        """Descarrega todos os fluxos."""
        for f in self.fluxos:
            f.flush()


def tabela(cabecalho, linhas):
    """Imprime tabela de texto alinhada."""
    larg = [max(len(str(x)) for x in col) for col in zip(cabecalho, *linhas)]
    fmt = "  ".join("{:<%d}" % w for w in larg)
    print(fmt.format(*cabecalho))
    print(fmt.format(*["-" * w for w in larg]))
    for l in linhas:
        print(fmt.format(*[str(x) for x in l]))


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def desenhar_grade(ax, grade, solucao, slots, titulo):
    """Desenha a cruzadinha preenchida (letras) com numeracao classica."""
    nl, nc = len(grade), len(grade[0])
    letras = {}
    for nome, _, cel in slots:
        for c, ch in zip(cel, solucao[nome]):
            letras[c] = ch
    inicios = {}
    for _, _, cel in slots:
        inicios.setdefault(cel[0], None)
    numero = {c: k + 1 for k, c in enumerate(sorted(inicios))}
    for r in range(nl):
        for c in range(nc):
            if grade[r][c] == "#":
                ax.add_patch(plt.Rectangle(
                    (c, nl - 1 - r), 1, 1, color="#222222"))
            else:
                ax.add_patch(plt.Rectangle((c, nl - 1 - r), 1, 1, fc="white",
                                           ec="#444444", lw=1))
                ax.text(c + 0.5, nl - 1 - r + 0.45, letras[(r, c)].upper(),
                        ha="center", va="center", fontsize=max(8, 170 // nl),
                        fontweight="bold", color="#0b3d91")
                if (r, c) in numero:
                    ax.text(c + 0.06, nl - 1 - r + 0.94, str(numero[(r, c)]),
                            ha="left", va="top", fontsize=max(5, 80 // nl),
                            color="#555555")
    ax.set_xlim(0, nc)
    ax.set_ylim(0, nl)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titulo, fontsize=10)


def figura_grade(grade, solucao, slots, titulo, arq):
    """Salva a cruzadinha preenchida em arquivo PNG."""
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    desenhar_grade(ax, grade, solucao, slots, titulo)
    fig.tight_layout()
    fig.savefig(arq, dpi=140)
    plt.close(fig)


def figura_comparacao(resumo, tamanhos, arq):
    """Barras agrupadas por tamanho de grade, uma barra por configuracao, escala
    log: backtracks, trabalho (valores de dominio inspecionados) e tempo de CPU,
    media geometrica das grades do tamanho. Barra hachurada = ao menos uma grade
    nao resolvida (orcamento esgotado): o valor e um LIMITE INFERIOR. O rotulo
    k/n indica quantas grades foram resolvidas."""
    cores = plt.cm.tab10.colors
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    larg = 0.8 / len(CONFIGS)
    paineis = [("backtracks", "backtracks (media geometrica, log)", 0.5),
               ("trabalho", "trabalho: valores inspecionados (log); linha = orcamento", 1),
               ("tempo", "tempo de CPU em s (log)", 1e-3)]
    for ax, (chave, rot, piso) in zip(axs, paineis):
        for k, cfg in enumerate(CONFIGS):
            for i, t in enumerate(tamanhos):
                rs = resumo[(cfg.nome, t)]
                vals = [max(r[chave], piso) for r in rs]
                y = math.exp(sum(math.log(v) for v in vals) / len(vals))
                ok = sum(r["status"] == "OK" for r in rs)
                x = i - 0.4 + larg * (k + 0.5)
                ax.bar(x, y, larg * 0.92, color=cores[k],
                       hatch="///" if ok < len(rs) else None,
                       edgecolor="white" if ok == len(rs) else "#222222", linewidth=0.6,
                       label=cfg.nome if (i == 0 and chave == "backtracks") else None)
                ax.text(x, y * 1.08, f"{ok}/{len(rs)}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90)
        ax.set_yscale("log")
        ax.set_xticks(range(len(tamanhos)))
        ax.set_xticklabels([f"{t}x{t}" for t in tamanhos])
        ax.set_xlabel("tamanho da grade (2 grades por tamanho)")
        ax.set_ylabel(rot)
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(top=ax.get_ylim()[1] * 4)
    axs[1].axhline(LIMITE_TRAB, color="#888888", ls=":", lw=1)
    h, r = axs[0].get_legend_handles_labels()
    fig.legend(h, r, fontsize=8, loc="lower center", ncol=len(CONFIGS))  # fora dos graficos
    fig.suptitle("Cruzadinha como CSP: esforco por configuracao e tamanho "
                 "(hachurado = houve TIMEOUT, valor e limite inferior; k/n = grades resolvidas)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(arq, dpi=120)
    plt.close(fig)


def figura_dicionario_densidade(dic, dens, nome_inst, arq):
    """Dois paineis: (esq.) resultado do MAC versus fracao do dicionario;
    (dir.) resultado do MAC versus densidade de casas bloqueadas (grades 9x9)."""
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    cores = {"SAT": "#2e8b57", "UNSAT": "#c0392b", "TIMEOUT": "#999999"}

    def empilhar(ax, rotulos, sat, unsat, tmo, xlabel, titulo):
        """Desenha um grafico de barras empilhadas (SAT / UNSAT / TIMEOUT) em ax."""
        pos = list(range(len(rotulos)))
        ax.bar(pos, sat, 0.7, color=cores["SAT"], label="resolvida (SAT)")
        ax.bar(pos, unsat, 0.7, bottom=sat, color=cores["UNSAT"],
               label="insatisfativel (provado)")
        ax.bar(pos, tmo, 0.7, bottom=[a + b for a, b in zip(sat, unsat)],
               color=cores["TIMEOUT"], label="TIMEOUT")
        ax.set_xticks(pos)
        ax.set_xticklabels(rotulos, fontsize=8)
        ax.set_ylabel("% das execucoes")
        ax.set_xlabel(xlabel)
        ax.set_title(titulo, fontsize=10)
        ax.set_ylim(0, 100)

    fr, sat, unsat, tmo, npal = dic
    empilhar(axs[0], [f"{100 * f:.0f}%\n[{n}]" for f, n in zip(fr, npal)],
             [100 * x for x in sat], [100 * x for x in unsat], [100 * x for x in tmo],
             "fracao do dicionario [no. de palavras]",
             f"Dicionario: {nome_inst}, MAC+MRV+Grau")
    bl, sat, unsat, tmo = zip(*dens)
    empilhar(axs[1], [f"{b:.1f}%" for b in bl],
             [100 * x for x in sat], [100 * x for x in unsat], [100 * x for x in tmo],
             "casas bloqueadas (grades 9x9 aleatorias simetricas)",
             "Densidade: MAC+MRV+Grau")
    h, r = axs[1].get_legend_handles_labels()
    fig.legend(h, r, fontsize=8, loc="lower center", ncol=3)  # fora dos graficos
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(arq, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------------
def gerar_grade_aleatoria(n, n_bloq, rng, maxlen=6):
    """Grade n x n com simetria de rotacao de 180 graus e ~n_bloq casas bloqueadas.
    Rejeita (e sorteia de novo) grades com casa livre fora de qualquer slot ou
    com slot maior que `maxlen`. Usada apenas no experimento de densidade."""
    pares = [(r, c) for r in range(n) for c in range(n)
             if (r, c) <= (n - 1 - r, n - 1 - c)]
    while True:
        g = [["."] * n for _ in range(n)]
        rng.shuffle(pares)
        k = 0
        for r, c in pares:
            if k >= n_bloq:
                break
            g[r][c] = g[n - 1 - r][n - 1 - c] = "#"
            k += 1 if (r, c) == (n - 1 - r, n - 1 - c) else 2
        grade = ["".join(l) for l in g]
        slots = extrair_slots(grade)
        cobertas = {c for _, _, cel in slots for c in cel}
        if len(cobertas) == sum(l.count(".") for l in grade) and \
                max(len(cel) for _, _, cel in slots) <= maxlen:
            return grade


def experimento_densidade(palavras):
    """Densidade de bloqueios: para cada no. de casas bloqueadas sorteia grades 9x9
    simetricas e classifica o resultado do MAC como SAT / UNSAT (provado) / TIMEOUT."""
    print("=== Experimento: densidade de casas bloqueadas (9x9, MAC+MRV+Grau) ===")
    print(f"{N_GRADES_DENS} grades aleatorias simetricas por densidade; limite "
          f"{LIMITE_TRAB_DENS} unidades de trabalho; slots <= 6 letras.")
    linhas, dados = [], []
    for nb in [12, 16, 20, 24]:
        rng = random.Random(SEMENTE + 5000 + nb)
        cont = Counter()
        nos_sat, slots_med, doisl = [], [], []
        for _ in range(N_GRADES_DENS):
            grade = gerar_grade_aleatoria(9, nb, rng)
            csp, slots = construir_csp(grade, palavras, rng)
            res = Buscador(csp, MAC_PADRAO, limite_trab=LIMITE_TRAB_DENS).resolver()
            cont[res.status] += 1
            if res.status == "OK":
                assert verificar_solucao(grade, res.atrib, slots, palavras)
                nos_sat.append(res.nos)
            slots_med.append(len(slots))
            doisl.append(sum(1 for _, _, c in slots if len(c) == 2))
        bloq = 100 * nb / 81
        n = N_GRADES_DENS
        dados.append((bloq, cont["OK"] / n, cont["SEM_SOLUCAO"] / n,
                      (cont["TIMEOUT"] + cont["LIMITE_NOS"]) / n))
        med = sorted(nos_sat)[len(nos_sat) // 2] if nos_sat else "-"
        linhas.append([f"{bloq:.1f}%", f"{sum(slots_med) / n:.1f}", f"{sum(doisl) / n:.1f}",
                       cont["OK"], cont["SEM_SOLUCAO"], cont["TIMEOUT"] + cont["LIMITE_NOS"], med])
    tabela(["bloqueadas", "slots (media)", "slots de 2 letras", "SAT", "UNSAT", "TIMEOUT",
            "nos medianos (SAT)"], linhas)
    print()
    return dados


def teste_generalidade():
    """Roda BT, FC e MAC no problema das 8 rainhas (CSP nao-cruzadinha)."""
    print("=== Teste de generalidade do nucleo CSP: 8 rainhas ===")
    linhas = []
    for cfg in [CONFIGS[0], CONFIGS[1], CONFIGS[3], MAC_PADRAO]:
        csp = csp_rainhas(8)
        res = Buscador(csp, cfg).resolver()
        assert res.status == "OK" and verifica_rainhas(res.atrib, 8), cfg.nome
        linhas.append([cfg.nome, res.nos, res.backtracks, f"{res.tempo:.4f}",
                       str([res.atrib[i] for i in range(8)])])
    tabela(["config", "nos", "backtracks",
           "tempo(s)", "linhas por coluna"], linhas)
    print()


def experimento_principal(palavras):
    """Todas as configuracoes em todas as grades; devolve resumo e solucoes."""
    print("=== Experimento principal: configuracoes x grades ===")
    print(f"Dicionario: {len(palavras)} palavras; orcamento por execucao: {LIMITE_TRAB} "
          f"unidades de trabalho (valores de dominio inspecionados) ou {LIMITE_NOS} nos; "
          f"Min-Conflicts: {PASSOS_MC} passos.")
    print("TIMEOUT = orcamento esgotado (nos/backtracks sao entao LIMITES INFERIORES).")
    print()
    insts = {}
    linhas_inst = []
    for k, (nome, grade) in enumerate(TEMPLATES.items()):
        csp, slots = construir_csp(grade, palavras, random.Random(SEMENTE + k))
        est = estatisticas_grade(grade, slots)
        insts[nome] = (grade, csp, slots)
        linhas_inst.append([nome, f"{est['bloq_pct']:.1f}%", est["slots"], est["cruzamentos"],
                            f"{est['grau_medio']:.2f}",
                            " ".join(f"{l}:{n}" for l, n in est["hist"].items())])
    tabela(["grade", "bloqueadas", "slots", "cruzam.", "grau medio", "slots por comprimento"],
           linhas_inst)
    print()

    resumo = {}     # (config, tamanho) -> lista de dicts
    solucoes = {}   # nome_grade -> (config, solucao)
    linhas = []
    for nome, (grade, csp, slots) in insts.items():
        tam = len(grade)
        for cfg in CONFIGS:
            res = Buscador(csp, cfg).resolver()
            if res.status == "OK":
                assert verificar_solucao(
                    grade, res.atrib, slots, palavras), (nome, cfg.nome)
                solucoes.setdefault(nome, (cfg.nome, res.atrib))
            resumo.setdefault((cfg.nome, tam), []).append(
                {"grade": nome, "status": res.status, "nos": res.nos,
                 "backtracks": res.backtracks, "tempo": res.tempo, "trabalho": res.trabalho,
                 "extra": res.extra})
            pod = res.extra.get("podados_ac3")
            linhas.append([nome, cfg.nome, res.status, res.nos, res.backtracks,
                           res.trabalho, f"{res.tempo:.3f}", "" if pod is None else f"{100 * pod:.1f}%"])
    # Min-Conflicts
    for k, (nome, (grade, csp, slots)) in enumerate(insts.items()):
        res = min_conflitos(csp, random.Random(SEMENTE + 100 + k))
        if res.status == "OK":
            assert verificar_solucao(
                grade, res.atrib, slots, palavras), (nome, "MC")
            solucoes.setdefault(nome, ("Min-Conflicts", res.atrib))
        tam = len(grade)
        resumo.setdefault(("Min-Conflicts", tam), []).append(
            {"grade": nome, "status": res.status, "nos": res.nos, "backtracks": 0,
             "tempo": res.tempo, "trabalho": 0, "extra": res.extra})
        linhas.append([nome, "Min-Conflicts", res.status,
                       f"{res.nos} passos", f"melhor={res.extra['melhor_conf']} confl.",
                       "-", f"{res.tempo:.3f}", ""])
    tabela(["grade", "configuracao", "status", "nos", "backtracks", "trabalho", "CPU(s)",
            "AC-3 podou"],
           linhas)
    print()

    # Autoverificacao cruzada: nenhum metodo completo pode declarar SEM_SOLUCAO
    # numa grade que outro resolveu.
    for (cn, t), rs in resumo.items():
        for r in rs:
            if r["status"] == "SEM_SOLUCAO":
                assert r["grade"] not in solucoes, (
                    "SEM_SOLUCAO indevido", cn, r["grade"])
    return resumo, solucoes, insts


def tabela_por_tamanho(resumo, tamanhos):
    """Tabela agregada por configuracao x tamanho (soma de nos, tempo total, #OK)."""
    print("=== Resumo por tamanho (2 grades por tamanho) ===")
    linhas = []
    for cfg_nome in [c.nome for c in CONFIGS] + ["Min-Conflicts"]:
        l = [cfg_nome]
        for t in tamanhos:
            rs = resumo[(cfg_nome, t)]
            ok = sum(r["status"] == "OK" for r in rs)
            nos = sum(r["nos"] for r in rs)
            tmp = sum(r["tempo"] for r in rs)
            l.append(f"{ok}/{len(rs)} ok; {nos} nos; {tmp:.2f}s CPU")
        linhas.append(l)
    tabela(["configuracao"] + [f"{t}x{t}" for t in tamanhos], linhas)
    print()


def experimento_dicionario(palavras, insts):
    """Varia o tamanho do dicionario (subconjuntos aleatorios) numa grade fixa e
    classifica o resultado do MAC em SAT / UNSAT (provado) / TIMEOUT."""
    nome = GRADE_DICIONARIO
    grade = insts[nome][0]
    print(f"=== Experimento: tamanho do dicionario ({nome}, MAC+MRV+Grau) ===")
    print(f"{REPS_DIC} subconjuntos aleatorios (e ordens de valores) por fracao; "
          f"limite {LIMITE_TRAB_DIC} unidades de trabalho.")
    fracoes = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sat, unsat, tmo, n_pal, linhas = [], [], [], [], []
    for f in fracoes:
        cont, nos_sat, tempos = Counter(), [], []
        for rep in range(REPS_DIC):
            rng = random.Random(SEMENTE + 1000 + rep + int(f * 100))
            sub = rng.sample(palavras, round(f * len(palavras)))
            csp, slots = construir_csp(grade, sorted(sub), rng)
            res = Buscador(csp, MAC_PADRAO, limite_trab=LIMITE_TRAB_DIC).resolver()
            cont["TIMEOUT" if res.status in ("TIMEOUT", "LIMITE_NOS") else res.status] += 1
            tempos.append(res.tempo)
            if res.status == "OK":
                assert verificar_solucao(grade, res.atrib, slots, sub)
                nos_sat.append(res.nos)
        n = REPS_DIC
        sat.append(cont["OK"] / n)
        unsat.append(cont["SEM_SOLUCAO"] / n)
        tmo.append(cont["TIMEOUT"] / n)
        n_pal.append(round(f * len(palavras)))
        med = sorted(nos_sat)[len(nos_sat) // 2] if nos_sat else "-"
        linhas.append([f"{100 * f:.0f}%", n_pal[-1], cont["OK"], cont["SEM_SOLUCAO"],
                       cont["TIMEOUT"], med, f"{sum(tempos) / n:.3f}"])
    tabela(["dicionario", "palavras", "SAT", "UNSAT", "TIMEOUT", "nos medianos (SAT)",
            "tempo medio(s)"], linhas)
    print()
    return fracoes, sat, unsat, tmo, n_pal


def main():
    """Roda todos os experimentos, imprime tabelas e gera as figuras."""
    FIGS.mkdir(exist_ok=True)
    arq_saida = open(FIGS / "05_saida.txt", "w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = Tee(original, arq_saida)
    t_ini = time.process_time()
    t_muro = time.perf_counter()
    try:
        random.seed(SEMENTE)
        palavras = carregar_dicionario()
        por_len = Counter(len(p) for p in palavras)
        print("Projeto 5 - CSP: geracao de cruzadinha")
        print(f"Dicionario: {len(palavras)} palavras unicas; por comprimento: "
              + " ".join(f"{l}:{n}" for l, n in sorted(por_len.items())))
        print()

        teste_generalidade()
        resumo, solucoes, insts = experimento_principal(palavras)
        tamanhos = sorted({len(g) for g, _, _ in insts.values()})
        tabela_por_tamanho(resumo, tamanhos)
        assert solucoes, "nenhuma grade foi resolvida"

        # Figura (a): maior grade resolvida
        maior = max(solucoes, key=lambda n: (
            len(insts[n][0]), len(extrair_slots(insts[n][0]))))
        cfg_nome, sol = solucoes[maior]
        grade, _, slots = insts[maior]
        print(
            f"Maior grade resolvida: {maior} (primeiro solucionador: {cfg_nome})")
        print("Palavras: " + ", ".join(f"{n}={sol[n]}" for n, _, _ in slots))
        print()
        figura_grade(grade, sol, slots,
                     f"Cruzadinha {maior} gerada por {cfg_nome}", FIGS / "05_grade_resolvida.png")

        # Figura (b): comparacao
        figura_comparacao(resumo, tamanhos, FIGS / "05_comparacao_configs.png")

        # Figura (c): dicionario e densidade
        dic = experimento_dicionario(palavras, insts)
        dens = experimento_densidade(palavras)
        figura_dicionario_densidade(dic, dens, GRADE_DICIONARIO,
                                    FIGS / "05_dicionario_densidade.png")

        print(f"Tempo total: {time.process_time() - t_ini:.1f}s de CPU; "
              f"{time.perf_counter() - t_muro:.1f}s de relogio")
        print("Todas as autoverificacoes passaram.")
    finally:
        sys.stdout = original
        arq_saida.close()


if __name__ == "__main__":
    main()
