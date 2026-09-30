"""
Projeto 4 - Algoritmo Genetico: aproximacao de imagem com triangulos translucidos
==================================================================================

PROBLEMA
    Dada uma imagem-alvo (por padrao, uma paisagem sintetica gerada por codigo:
    ceu em gradiente, sol, duas cadeias de montanhas e um lago), encontrar uma
    lista de N triangulos translucidos cuja composicao (alpha blending, na ordem
    da lista = ordem-z, do fundo para a frente) se pareca o maximo possivel com o
    alvo. Nao ha gradiente util: a funcao de custo (MSE apos rasterizar) e
    nao-diferenciavel em relacao aos vertices, o que justifica uma busca
    evolutiva.

CROMOSSOMO
    Array numpy de forma (N, 10), valores em [0, 1]:
        [x1, y1, x2, y2, x3, y3, r, g, b, a]
    A posicao na lista define a ordem de desenho (z-order). O alpha real do
    triangulo e A_MIN + (A_MAX - A_MIN) * a.

FITNESS
    MSE entre a imagem renderizada (Pillow, RGB, 64x64, valores em [0, 1]) e o
    alvo. O GA MINIMIZA o MSE (equivale a maximizar -MSE). Tambem reportamos
    PSNR = -10 log10(MSE) para leitura mais intuitiva.

ALGORITMO (implementado a mao, sem bibliotecas de GA)
    - populacao de P individuos, inicializacao aleatoria;
    - selecao por torneio (k = 3);
    - crossover: uniforme por poligono (cada triangulo vem de um dos pais),
      ponto unico (prefixo de um pai + sufixo do outro) ou nenhum;
    - mutacao por poligono com probabilidade pm; operadores: deslocar vertice,
      transladar triangulo, jitter de cor, jitter de alpha, troca de ordem-z e
      reset do poligono; o desvio-padrao dos jitters decai linearmente
      (1.0 -> 0.25) ao longo do orcamento;
    - elitismo: os E melhores passam intactos para a proxima geracao.

EXPERIMENTOS (main)
    Todos com o MESMO orcamento de avaliacoes de fitness (BUDGET) para que a
    comparacao seja justa (populacoes maiores rodam menos geracoes).
    Ablacao de: taxa de mutacao, tamanho da populacao, tipo de crossover
    (nenhum / ponto unico / uniforme), numero de poligonos N e elitismo.
    Cada configuracao roda com N_SEEDS sementes fixas; reportamos media e
    desvio-padrao do MSE final.

USO
    python 04_algoritmo_genetico.py [caminho_de_imagem_opcional]
    Saidas: figs/04_evolucao.png, figs/04_ablacao.png, figs/04_galeria.png,
            figs/04_evolucao.gif e figs/04_saida.txt
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

# --------------------------------------------------------------------------
# Constantes globais
# --------------------------------------------------------------------------
AQUI = Path(__file__).resolve().parent
FIGS = AQUI / "figs"
S = 64                    # resolucao (S x S) da imagem
A_MIN, A_MAX = 0.05, 0.60  # faixa do alpha real dos triangulos
BUDGET = 40_000           # orcamento de avaliacoes de fitness por execucao
N_SEEDS = 3               # repeticoes (sementes 0..N_SEEDS-1) por configuracao
GENES = 10                # genes por triangulo


# --------------------------------------------------------------------------
# Saida: imprime no console e guarda para salvar em figs/04_saida.txt
# --------------------------------------------------------------------------
_LOG: list[str] = []


def say(*args) -> None:
    """print() que tambem acumula o texto para gravar em arquivo."""
    texto = " ".join(str(a) for a in args)
    _LOG.append(texto)
    print(texto, flush=True)


# --------------------------------------------------------------------------
# Imagem-alvo
# --------------------------------------------------------------------------
def alvo_sintetico(tam: int = S) -> np.ndarray:
    """Gera a paisagem sintetica (ceu, sol, montanhas, lago) em `tam` x `tam`.

    Desenhada em 4x de resolucao e reduzida (anti-aliasing). Retorna float32
    em [0, 1] com forma (tam, tam, 3).
    """
    k = 4
    W = tam * k
    img = Image.new("RGB", (W, W))
    px = img.load()
    horizonte = int(W * 0.62)
    # ceu: gradiente vertical azul -> laranja claro perto do horizonte
    for y in range(horizonte):
        t = y / horizonte
        cor = (int(40 + 215 * t ** 1.5), int(90 + 110 * t), int(190 - 20 * t))
        for x in range(W):
            px[x, y] = cor
    d = ImageDraw.Draw(img)
    # sol com um halo simples (circulos concentricos)
    cx, cy = int(W * 0.70), int(W * 0.30)
    for r, cor in [(int(W * 0.16), (255, 200, 120)), (int(W * 0.11), (255, 225, 150)),
                   (int(W * 0.07), (255, 245, 200))]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cor)
    # montanhas distantes (azul-acinzentado) e proximas (verde-escuro)
    d.polygon([(0, horizonte), (0, int(W * .45)), (int(W * .22), int(W * .32)),
               (int(W * .42), int(W * .50)), (int(W * .60), int(W * .38)),
               (W, int(W * .52)), (W, horizonte)], fill=(95, 105, 140))
    d.polygon([(0, horizonte), (0, int(W * .55)), (int(W * .18), int(W * .47)),
               (int(W * .38), int(W * .62)), (int(W * .55), int(W * .53)),
               (int(W * .80), int(W * .62)), (W, int(W * .56)), (W, horizonte)],
              fill=(30, 70, 60))
    # lago: gradiente escuro + reflexo do sol
    for y in range(horizonte, W):
        t = (y - horizonte) / (W - horizonte)
        cor = (int(60 - 30 * t), int(110 - 40 * t), int(150 - 40 * t))
        d.line([(0, y), (W, y)], fill=cor)
    d.polygon([(cx - int(W * .05), horizonte), (cx + int(W * .05), horizonte),
               (cx + int(W * .02), int(W * .90)), (cx - int(W * .02), int(W * .90))],
              fill=(230, 190, 130))
    img = img.resize((tam, tam), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def carregar_alvo(caminho: str | None) -> np.ndarray:
    """Alvo sintetico ou imagem do usuario (redimensionada para S x S)."""
    if caminho is None:
        return alvo_sintetico()
    img = Image.open(caminho).convert("RGB").resize((S, S), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------
# Renderizacao e fitness
# --------------------------------------------------------------------------
def renderizar(genoma: np.ndarray, tam: int = S) -> np.ndarray:
    """Rasteriza o cromossomo com Pillow. Retorna float32 (tam, tam, 3) em [0, 1].

    ImageDraw no modo "RGBA" sobre uma imagem RGB faz alpha blending em cada
    poligono, respeitando a ordem da lista (fundo preto).
    """
    img = Image.new("RGB", (tam, tam), (0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    xy = (genoma[:, :6] * tam).tolist()
    cor = (genoma[:, 6:9] * 255).astype(np.int32).tolist()
    alfa = ((A_MIN + (A_MAX - A_MIN) *
            genoma[:, 9]) * 255).astype(np.int32).tolist()
    for p, c, a in zip(xy, cor, alfa):
        d.polygon(p, fill=(c[0], c[1], c[2], a))
    return np.asarray(img, dtype=np.float32) / 255.0


def mse(genoma: np.ndarray, alvo: np.ndarray) -> float:
    """Fitness (a minimizar): erro quadratico medio contra o alvo."""
    return float(np.mean((renderizar(genoma, alvo.shape[0]) - alvo) ** 2))


def psnr(m: float) -> float:
    """PSNR em dB a partir do MSE (pixels em [0, 1]); protege contra log(0)."""
    return -10.0 * np.log10(max(m, 1e-12))


# --------------------------------------------------------------------------
# Operadores geneticos
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Hiperparametros de uma execucao do GA."""
    nome: str = "base"
    n_poly: int = 50          # numero de triangulos por cromossomo
    pop: int = 40             # tamanho da populacao
    pm: float = 0.05          # prob. de mutar cada poligono de um filho
    pc: float = 0.9           # prob. de aplicar crossover
    cruz: str = "uniforme"    # "uniforme" | "ponto" | "nenhum"
    elite: int = 2            # numero de elites
    torneio: int = 3          # tamanho do torneio


def poligono_aleatorio(rng: np.random.Generator) -> np.ndarray:
    """Triangulo pequeno-medio aleatorio: centro uniforme, vertices ao redor."""
    centro = rng.random(2)
    verts = np.clip(centro + rng.normal(0, 0.15, size=(3, 2)), 0, 1).ravel()
    cor = rng.random(3)
    alfa = rng.random(1)
    return np.concatenate([verts, cor, alfa])


def populacao_inicial(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Populacao aleatoria: array (pop, n_poly, 10) de triangulos aleatorios."""
    return np.stack([np.stack([poligono_aleatorio(rng) for _ in range(cfg.n_poly)])
                     for _ in range(cfg.pop)])


def torneio(fit: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """Sorteia k individuos e devolve o indice do de menor MSE."""
    cand = rng.integers(0, len(fit), size=k)
    return int(cand[np.argmin(fit[cand])])


def crossover(a: np.ndarray, b: np.ndarray, tipo: str,
              rng: np.random.Generator) -> np.ndarray:
    """Recombina dois pais. O poligono i ocupa sempre o slot z-i do filho."""
    if tipo == "uniforme":      # cada triangulo vem de A ou B com prob. 1/2
        mascara = rng.random(len(a)) < 0.5
        return np.where(mascara[:, None], a, b)
    if tipo == "ponto":         # prefixo de A + sufixo de B
        corte = int(rng.integers(1, len(a)))
        return np.concatenate([a[:corte], b[corte:]])
    return a.copy()


def mutar(g: np.ndarray, pm: float, escala: float,
          rng: np.random.Generator) -> np.ndarray:
    """Mutacao por poligono. `escala` multiplica o desvio dos jitters (decai)."""
    g = g.copy()
    n = len(g)
    for i in np.nonzero(rng.random(n) < pm)[0]:
        u = rng.random()
        if u < 0.25:                          # desloca um vertice
            v = int(rng.integers(0, 3))
            g[i, 2 * v:2 * v + 2] += rng.normal(0, 0.10 * escala, 2)
        elif u < 0.40:                        # translada o triangulo inteiro
            g[i, :6] += np.tile(rng.normal(0, 0.06 * escala, 2), 3)
        elif u < 0.65:                        # jitter de cor
            g[i, 6:9] += rng.normal(0, 0.15 * escala, 3)
        elif u < 0.80:                        # jitter de alpha
            g[i, 9] += rng.normal(0, 0.20 * escala)
        elif u < 0.92:                        # troca de ordem-z com outro poligono
            j = int(rng.integers(0, n))
            g[[i, j]] = g[[j, i]]
        else:                                 # reset: novo poligono aleatorio
            g[i] = poligono_aleatorio(rng)
    np.clip(g, 0.0, 1.0, out=g)
    return g


# --------------------------------------------------------------------------
# Laco principal do GA
# --------------------------------------------------------------------------
def rodar_ga(alvo: np.ndarray, cfg: Config, seed: int, budget: int = BUDGET,
             snap_cada: int = 0, snaps_extra: tuple = ()) -> dict:
    """Executa um GA ate esgotar `budget` avaliacoes de fitness.

    Retorna dict com historico (avaliacoes, melhor MSE por geracao), o melhor
    genoma final e, opcionalmente, snapshots do melhor genoma:
    em geracoes multiplas de `snap_cada` e nas listadas em `snaps_extra`.
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    pop = populacao_inicial(cfg, rng)
    fit = np.array([mse(g, alvo) for g in pop])
    avals = cfg.pop
    ger = 0
    h_aval, h_best = [avals], [fit.min()]
    snaps = {}

    def guardar():
        """Salva o melhor genoma atual se a geracao pedir snapshot."""
        if (snap_cada and ger % snap_cada == 0) or ger in snaps_extra:
            snaps[ger] = pop[int(np.argmin(fit))].copy()

    guardar()
    n_filhos = cfg.pop - cfg.elite
    while avals + n_filhos <= budget:
        ordem = np.argsort(fit)
        elites, fit_el = pop[ordem[:cfg.elite]], fit[ordem[:cfg.elite]]
        escala = 1.0 - 0.75 * avals / budget       # decaimento do passo
        filhos = np.empty((n_filhos,) + pop.shape[1:])
        for c in range(n_filhos):
            a = pop[torneio(fit, cfg.torneio, rng)]
            b = pop[torneio(fit, cfg.torneio, rng)]
            if cfg.cruz != "nenhum" and rng.random() < cfg.pc:
                f = crossover(a, b, cfg.cruz, rng)
            else:
                f = a
            filhos[c] = mutar(f, cfg.pm, escala, rng)
        fit_f = np.array([mse(g, alvo) for g in filhos])
        avals += n_filhos
        pop = np.concatenate([elites, filhos]) if cfg.elite else filhos
        fit = np.concatenate([fit_el, fit_f]) if cfg.elite else fit_f
        ger += 1
        h_aval.append(avals)
        h_best.append(fit.min())
        guardar()

    melhor = pop[int(np.argmin(fit))].copy()
    snaps[ger] = melhor
    return dict(cfg=cfg, seed=seed, aval=np.array(h_aval), best=np.array(h_best),
                final=float(fit.min()), genoma=melhor, ger=ger, snaps=snaps,
                tempo=time.perf_counter() - t0)


def _tarefa(args):
    """Wrapper de nivel superior (picklavel) para multiprocessing."""
    alvo, cfg, seed, kw = args
    return rodar_ga(alvo, cfg, seed, **kw)


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def fig_evolucao(alvo, base0, alvo_txt="alvo"):
    """(a) alvo + melhor imagem nas geracoes 0/100/1000/max (baseline, seed 0)."""
    gmax = base0["ger"]
    ger = [0, 100, 1000, gmax]
    fig, ax = plt.subplots(1, 5, figsize=(13, 3.5))
    ax[0].imshow(alvo, interpolation="nearest")
    ax[0].set_title(alvo_txt)
    for a, g in zip(ax[1:], ger):
        m = mse(base0["snaps"][g], alvo)
        a.imshow(renderizar(base0["snaps"][g]), interpolation="nearest")
        rotulo = f"geração {g}" + (" (máx.)" if g == gmax else "")
        a.set_title(
            f"{rotulo}\nMSE={m:.4f}  PSNR={psnr(m):.1f} dB", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(FIGS / "04_evolucao.png", dpi=130)
    plt.close(fig)


def curva_media(res_lista, grade):
    """Interpola (degrau) o melhor MSE de cada execucao em uma grade de avals."""
    m = []
    for r in res_lista:
        idx = np.searchsorted(r["aval"], grade, side="right") - 1
        m.append(r["best"][idx])
    m = np.array(m)
    return m.mean(0), m.std(0)


def fig_ablacao(resultados, grupos):
    """(b) curvas MSE x avaliacoes (media +- desvio) por grupo de ablacao."""
    grade = np.linspace(BUDGET * 0.02, BUDGET, 300)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    axs = axs.ravel()
    for ax, (titulo, nomes) in zip(axs, grupos.items()):
        for nome in nomes:
            mu, sd = curva_media(resultados[nome], grade)
            ax.plot(grade, mu, label=nome)
            ax.fill_between(grade, np.maximum(
                mu - sd, 1e-6), mu + sd, alpha=0.2)
        ax.set_yscale("log")
        ax.set_title(titulo)
        ax.set_xlabel("avaliações de fitness")
        ax.set_ylabel("melhor MSE (média ± dp)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    # ultimo painel: barras com o MSE final de todas as configuracoes
    ax = axs[-1]
    nomes = list(resultados)
    mus = [np.mean([r["final"] for r in resultados[n]]) for n in nomes]
    sds = [np.std([r["final"] for r in resultados[n]]) for n in nomes]
    ax.barh(range(len(nomes)), mus, xerr=sds, color="tab:blue", alpha=0.8)
    ax.set_yticks(range(len(nomes)))
    ax.set_yticklabels(nomes, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("MSE final (média ± dp)")
    ax.set_title("MSE final por configuração")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIGS / "04_ablacao.png", dpi=120)
    plt.close(fig)


def fig_galeria(alvo, resultados, nomes):
    """Imagens finais (seed 0) de configuracoes selecionadas, ao lado do alvo."""
    fig, axs = plt.subplots(2, 4, figsize=(11, 6.8))
    axs = axs.ravel()
    axs[0].imshow(alvo, interpolation="nearest")
    axs[0].set_title("alvo")
    for ax, n in zip(axs[1:], nomes):
        r = resultados[n][0]
        ax.imshow(renderizar(r["genoma"]), interpolation="nearest")
        ax.set_title(f"{n}\nMSE={r['final']:.4f}", fontsize=9)
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGS / "04_galeria.png", dpi=120)
    plt.close(fig)


def salvar_gif(base0):
    """GIF barato da evolucao do melhor individuo (baseline, seed 0)."""
    quadros = []
    for g in sorted(base0["snaps"]):
        im = Image.fromarray(
            (renderizar(base0["snaps"][g]) * 255).astype(np.uint8))
        quadros.append(im.resize((256, 256), Image.NEAREST))
    quadros[0].save(FIGS / "04_evolucao.gif", save_all=True,
                    append_images=quadros[1:], duration=80, loop=0)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    """Roda autoverificacoes, ablacao paralela, tabelas e gera figuras/saida."""
    FIGS.mkdir(exist_ok=True)
    t_inicio = time.perf_counter()
    caminho = sys.argv[1] if len(sys.argv) > 1 else None
    alvo = carregar_alvo(caminho)
    say("=" * 70)
    say("PROJETO 4 - Algoritmo Genetico: aproximacao de imagem com triangulos")
    say("=" * 70)
    say(f"alvo: {'sintetico (paisagem)' if caminho is None else caminho}, "
        f"{S}x{S}; orcamento = {BUDGET} avaliacoes/execucao; "
        f"sementes = {N_SEEDS}; nucleos = {cpu_count()}")

    # --- referencias triviais e autoverificacoes de infraestrutura ---------
    m_media = float(np.mean((alvo - alvo.mean((0, 1))) ** 2))
    m_preto = float(np.mean(alvo ** 2))
    say(f"MSE de referencia: imagem = cor media do alvo: {m_media:.5f} "
        f"(PSNR {psnr(m_media):.1f} dB); imagem preta: {m_preto:.5f}")
    rng = np.random.default_rng(123)
    g_teste = populacao_inicial(Config(n_poly=5, pop=1), rng)[0]
    assert mse(g_teste, alvo) == mse(
        g_teste, alvo), "render deve ser deterministico"
    im = renderizar(g_teste)
    assert im.shape == (S, S, 3) and 0.0 <= im.min() and im.max() <= 1.0
    assert mse(np.zeros((0, GENES)),
               alvo) == m_preto, "sem poligonos = fundo preto"
    # reprodutibilidade: mesma semente -> mesmo resultado
    pequeno = Config(n_poly=10, pop=8, elite=1)
    r1 = rodar_ga(alvo, pequeno, seed=7, budget=200)
    r2 = rodar_ga(alvo, pequeno, seed=7, budget=200)
    assert r1["final"] == r2["final"] and np.array_equal(
        r1["genoma"], r2["genoma"])
    # custo de uma avaliacao (para documentar o orcamento)
    g50 = populacao_inicial(Config(), rng)[0]
    t = time.perf_counter()
    for _ in range(300):
        mse(g50, alvo)
    ms_aval = (time.perf_counter() - t) / 300 * 1000
    say(f"custo de 1 avaliacao (N=50, render+MSE): {ms_aval:.3f} ms -> "
        f"{BUDGET} avals = {BUDGET * ms_aval / 1000:.1f} s por execucao (1 nucleo)")

    # --- definicao das configuracoes da ablacao ------------------------------
    base = Config(nome="base")
    cfgs: dict[str, Config] = {"base (pm=.05,P=40,unif,N=50,E=2)": base}
    grupos: dict[str, list[str]] = {}
    nome_base = "base (pm=.05,P=40,unif,N=50,E=2)"

    def add(grupo, **kw):
        """Registra uma variante da base (um fator alterado) e a agrupa para a figura."""
        nome = kw.pop("nome")
        cfgs[nome] = replace(base, nome=nome, **kw)
        grupos.setdefault(grupo, [nome_base]).append(nome)

    for pm in (0.01, 0.15, 0.40):
        add("Taxa de mutação pm", nome=f"pm={pm}", pm=pm)
    for p in (10, 160):
        add("Tamanho da população P", nome=f"P={p}", pop=p)
    add("Crossover", nome="sem crossover", cruz="nenhum", pc=0.0)
    add("Crossover", nome="ponto unico", cruz="ponto")
    for n in (10, 25, 100):
        add("Número de triângulos N", nome=f"N={n}", n_poly=n)
    add("Elitismo", nome="sem elitismo (E=0)", elite=0)
    # ordena para o grafico de barras ficar por grupo
    nomes_ordem = [nome_base] + [n for n in cfgs if n != nome_base]

    # --- execucao paralela (cada tarefa e independente e deterministica) ----
    tarefas = []
    for nome in nomes_ordem:
        for s in range(N_SEEDS):
            kw = {}
            if nome == nome_base and s == 0:   # execucao principal: guarda snapshots
                kw = dict(snap_cada=25, snaps_extra=(100, 1000))
            tarefas.append((alvo, cfgs[nome], s, kw))
    say(f"\nRodando {len(tarefas)} execucoes do GA em paralelo ({len(cfgs)} configuracoes "
        f"x {N_SEEDS} sementes)...")
    with Pool(min(cpu_count(), 10)) as pool:
        brutos = pool.map(_tarefa, tarefas, chunksize=1)
    resultados: dict[str, list] = {n: [] for n in nomes_ordem}
    for (_, cfg, _, _), r in zip(tarefas, brutos):
        resultados[[n for n in cfgs if cfgs[n] == cfg][0]].append(r)

    # --- autoverificacao: com elitismo o melhor fitness e monotonico -------
    for n, lista in resultados.items():
        for r in lista:
            if r["cfg"].elite >= 1:
                assert np.all(np.diff(r["best"]) <=
                              1e-12), f"nao monotonico: {n}"
    say("assert OK: com elitismo (E>=1) o melhor MSE e monotonicamente nao crescente "
        "em todas as execucoes")
    # sem elitismo pode piorar: contamos quantas geracoes pioraram
    piora = [int(np.sum(np.diff(r["best"]) > 1e-12))
             for r in resultados["sem elitismo (E=0)"]]
    say(
        f"sem elitismo: numero de geracoes em que o melhor piorou (por semente) = {piora}")

    # --- tabelas -------------------------------------------------------------
    base0 = resultados[nome_base][0]
    say("\n--- Execucao principal (base, semente 0) ---")
    say("geracao | avaliacoes | MSE      | PSNR (dB)")
    for g in (0, 100, 1000, base0["ger"]):
        idx = int(np.searchsorted(base0["aval"], base0["aval"][g]))
        m = mse(base0["snaps"][g], alvo)
        say(f"{g:7d} | {base0['aval'][idx]:10d} | {m:.5f}  | {psnr(m):.2f}")
    assert base0["final"] < base0["best"][0] and base0["final"] < m_media, \
        "GA deve superar o gen-0 e a imagem de cor media"

    say("\n--- Ablacao: MSE final apos o orcamento (media +- dp sobre "
        f"{N_SEEDS} sementes) ---")
    say(f"{'configuracao':38s} | {'MSE medio':>9s} | {'dp':>8s} | {'PSNR':>6s} | "
        f"{'melhor':>8s} | {'geracoes':>8s} | {'t/exec (s)':>10s}")
    for n in nomes_ordem:
        f = np.array([r["final"] for r in resultados[n]])
        say(f"{n:38s} | {f.mean():9.5f} | {f.std():8.5f} | {psnr(f.mean()):6.2f} | "
            f"{f.min():8.5f} | {resultados[n][0]['ger']:8d} | "
            f"{np.mean([r['tempo'] for r in resultados[n]]):10.1f}")
    say("\n--- MSE apos fracoes do orcamento (media sobre sementes) ---")
    grade = np.array([0.1, 0.25, 0.5, 1.0]) * BUDGET
    say(f"{'configuracao':38s} | " + " | ".join(f"{int(x):>8d}" for x in grade))
    for n in nomes_ordem:
        mu, _ = curva_media(resultados[n], grade)
        say(f"{n:38s} | " + " | ".join(f"{v:8.5f}" for v in mu))

    # --- figuras -------------------------------------------------------------
    fig_evolucao(alvo, base0, "alvo" if caminho is None else "alvo (usuário)")
    grupos_fig = dict(grupos)
    fig_ablacao(resultados, grupos_fig)
    galeria = [nome_base, "sem crossover",
               "ponto unico", "N=10", "N=100", "P=10", "P=160"]
    fig_galeria(alvo, resultados, galeria)
    salvar_gif(base0)
    say(f"\nFiguras salvas em {FIGS} (04_evolucao.png, 04_ablacao.png, "
        f"04_galeria.png, 04_evolucao.gif)")
    say(f"Tempo total: {time.perf_counter() - t_inicio:.1f} s")
    (FIGS / "04_saida.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    # evita erro de codificacao no console cp1252 do Windows
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    main()
