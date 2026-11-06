"""
Projeto 2 - Modelos de Markov Ocultos (HMM): corrigindo erros de digitacao.

PROBLEMA
    Um texto em portugues foi digitado num teclado QWERTY e cada tecla pode sair
    errada: com probabilidade p o digitador aperta uma tecla VIZINHA da correta
    (ex.: 'a' -> 's', 'q', 'w', 'z'). Dado so o texto digitado (ruidoso), recuperar o
    texto pretendido. Nao e um corretor de dicionario: o modelo so conhece
    estatisticas de letras.

MODELAGEM
    Estados ocultos : letra pretendida, 27 simbolos ('a'..'z' e espaco).
    Observacoes     : tecla efetivamente apertada (os mesmos 27 simbolos).
    Transicao A     : bigrama de letras, estimado do corpus (Dom Casmurro, dominio
                      publico, 80% iniciais para treino; os 20% finais viram teste).
    Emissao B       : canal de ruido do teclado (acerta com 1-p; senao uma vizinha
                      uniforme; o espaco erra para c,v,b,n,m e vice-versa).
    Limite          : so ha erros de SUBSTITUICAO (nao ha tecla a mais ou a menos),
                      o que mantem estados e observacoes alinhados no tempo.

ALGORITMOS IMPLEMENTADOS A MAO (numpy)
    - Forward e backward com ESCALONAMENTO (evita underflow em T = milhares).
    - Viterbi em log (decodificacao MAP da sequencia).
    - Decodificacao posterior (argmax de gamma_t(i), maximiza acertos por posicao).
    - BAUM-WELCH (EM): aprende B (com A fixa) ou A, B, pi juntos a partir SO do texto
      ruidoso, sem rotulos; a log-verossimilhanca cresce a cada iteracao.
    - HMM de SEGUNDA ORDEM (estado = par de letras, trigrama com backoff para o
      bigrama) e seu Viterbi: conceito alem do basico.

VERDADE DE REFERENCIA / AUTOVERIFICACAO
    - Forward == soma sobre todos os N^T caminhos e Viterbi == argmax de todos os
      caminhos (HMM pequeno, T=7, forca bruta); gamma soma 1; xi marginaliza em gamma.
    - Viterbi de 2a ordem == forca bruta (HMM pequeno) e == Viterbi de 1a ordem quando
      o trigrama e construido a partir do bigrama.
    - Log-verossimilhanca do Baum-Welch monotona nao decrescente.
    - CER do HMM < CER do texto cru e < CER sem contexto (media de 5 trechos/semente).

Execucao: python 02_hmm_digitacao.py  (gera figs/02_*.png e figs/02_saida.txt)
"""

import itertools
import re
import time
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fontes legiveis quando a figura e reduzida a largura de uma pagina A4
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
import numpy as np

PASTA = Path(__file__).resolve().parent
FIGS = PASTA / "figs"
FIGS.mkdir(exist_ok=True)

ALF = "abcdefghijklmnopqrstuvwxyz "
S = len(ALF)
IDX = {c: i for i, c in enumerate(ALF)}
_LOG = []


def _saida(*args):
    """Imprime a linha e a acumula em _LOG para gravar em figs/02_saida.txt."""
    linha = " ".join(str(a) for a in args)
    print(linha)
    _LOG.append(linha)


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def carrega_corpus():
    """Le corpus_pt.txt, remove acentos, deixa a-z e espaco e devolve o texto como array de indices."""
    txt = (PASTA / "corpus_pt.txt").read_text(encoding="utf-8", errors="ignore").lower()
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    txt = re.sub(r"[^a-z]+", " ", txt).strip()
    return np.array([IDX[c] for c in txt], dtype=np.int64)


def vizinhas_teclado():
    """Vizinhas fisicas no QWERTY (coordenadas com deslocamento de cada fileira)."""
    linhas, desloc = ["qwertyuiop", "asdfghjkl", "zxcvbnm"], [0.0, 0.5, 1.0]
    pos = {c: (j + desloc[i], i) for i, l in enumerate(linhas) for j, c in enumerate(l)}
    viz = {c: [d for d in pos if d != c and np.hypot(pos[c][0] - pos[d][0], pos[c][1] - pos[d][1]) < 1.2] for c in pos}
    for c in "cvbnm":
        viz[c].append(" ")
    viz[" "] = list("cvbnm")
    return viz


VIZ = vizinhas_teclado()


def matriz_emissao(p):
    """Matriz B (estado = letra pretendida, observacao = tecla): acerta com 1-p, senao tecla vizinha uniforme."""
    B = np.zeros((S, S))
    for c in ALF:
        i = IDX[c]
        B[i, i] = 1 - p
        for d in VIZ[c]:
            B[i, IDX[d]] = p / len(VIZ[c])
    return B


def corrompe(seq, p, rng):
    """Gera as teclas digitadas a partir do texto seq amostrando de B(p) com o rng dado."""
    B = matriz_emissao(p)
    cum = np.cumsum(B, axis=1)
    u = rng.random(len(seq))
    return np.minimum((u[:, None] > cum[seq]).sum(axis=1), S - 1)


def modelo_linguagem(treino, alfa=0.1, beta=5.0):
    """Estima do treino pi, bigrama A (suavizacao alfa) e trigrama A2 com backoff Dirichlet (beta) para o bigrama."""
    C = np.zeros((S, S))
    np.add.at(C, (treino[:-1], treino[1:]), 1)
    A = (C + alfa) / (C + alfa).sum(axis=1, keepdims=True)
    pi = np.bincount(treino, minlength=S) + alfa
    pi = pi / pi.sum()
    C3 = np.zeros((S, S, S))
    np.add.at(C3, (treino[:-2], treino[1:-1], treino[2:]), 1)
    A2 = (C3 + beta * A[None, :, :]) / (C3.sum(axis=2, keepdims=True) + beta)  # backoff Dirichlet p/ bigrama
    return pi, A, A2


# ----------------------------------------------------------------------------
# HMM de primeira ordem
# ----------------------------------------------------------------------------
def forward(pi, A, B, obs):
    """Forward escalonado: alpha_t normalizado, c_t = fator de escala.
    log P(obs) = sum log c_t."""
    T, N = len(obs), len(pi)
    alpha, c = np.zeros((T, N)), np.zeros(T)
    # Escalonamento: dividir alpha_t pela sua soma c_t a cada passo evita underflow em T grande;
    # a verossimilhanca vem de sum(log c_t) em vez do produto de numeros minusculos.
    a = pi * B[:, obs[0]]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c


def backward(A, B, obs, c):
    """Variaveis backward escalonadas pelos mesmos c_t do forward (evita underflow). Saida: beta (T x N)."""
    T, N = len(obs), A.shape[0]
    beta = np.ones((T, N))
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
    return beta


def viterbi(pi, A, B, obs):
    """Sequencia de estados mais provavel em log. Entrada: pi, A, B, obs. Saida: caminho (T)."""
    T, N = len(obs), len(pi)
    with np.errstate(divide="ignore"):
        lpi, lA, lB = np.log(pi), np.log(A), np.log(B)
    delta = lpi + lB[:, obs[0]]
    psi = np.zeros((T, N), dtype=np.int64)
    for t in range(1, T):
        sc = delta[:, None] + lA
        psi[t] = sc.argmax(axis=0)
        delta = sc.max(axis=0) + lB[:, obs[t]]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path, delta.max()


def posterior_decode(pi, A, B, obs):
    """Decodificacao posterior: em cada t, o estado de maior gamma_t = alpha*beta."""
    alpha, c = forward(pi, A, B, obs)
    gamma = alpha * backward(A, B, obs, c)
    return gamma.argmax(axis=1)


def baum_welch(pi, A, B, obs, iters, aprende_A=True, aprende_B=True, aprende_pi=True):
    """EM para HMM discreto (Rabiner). Retorna parametros e historico de log P(obs)."""
    pi, A, B = pi.copy(), A.copy(), B.copy()
    T, N, M = len(obs), len(pi), B.shape[1]
    hist = []
    onehot = np.zeros((T, M))
    onehot[np.arange(T), obs] = 1
    for _ in range(iters):
        alpha, c = forward(pi, A, B, obs)
        beta = backward(A, B, obs, c)
        hist.append(np.log(c).sum())
        gamma = alpha * beta
        if aprende_A:
            # xi_t(i,j) = alpha_t(i) A_ij B_j(o_t+1) beta_t+1(j) / c_t+1
            W = (B[:, obs[1:]].T * beta[1:]) / c[1:, None]
            xi = A * (alpha[:-1].T @ W)
            A = xi / xi.sum(axis=1, keepdims=True)
        if aprende_B:
            B = gamma.T @ onehot + 1e-12
            B = B / B.sum(axis=1, keepdims=True)
        if aprende_pi:
            pi = gamma[0]
    return pi, A, B, hist


# ----------------------------------------------------------------------------
# HMM de segunda ordem (estado = (letra anterior, letra atual))
# ----------------------------------------------------------------------------
def viterbi2(pi, A, A2, B, obs):
    """Viterbi de 2a ordem: o estado e o par (letra anterior, atual); usa o trigrama A2."""
    T = len(obs)
    with np.errstate(divide="ignore"):
        lpi, lA, lA2, lB = np.log(pi), np.log(A), np.log(A2), np.log(B)
    delta = lpi[:, None] + lB[:, obs[0]][:, None] + lA + lB[:, obs[1]][None, :]  # delta[a,b]
    psi = np.zeros((T, S, S), dtype=np.int64)
    for t in range(2, T):
        sc = delta[:, :, None] + lA2  # (a,b,c)
        psi[t] = sc.argmax(axis=0)    # melhor a para cada (b,c)
        delta = sc.max(axis=0) + lB[:, obs[t]][None, :]
    a, b = np.unravel_index(delta.argmax(), delta.shape)
    path = np.zeros(T, dtype=np.int64)
    path[-1], path[-2] = b, a
    for t in range(T - 1, 1, -1):
        path[t - 2] = psi[t, path[t - 1], path[t]]
    return path, delta.max()


# ----------------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------------
def cer(a, b):
    """Taxa de erro por caractere: fracao de posicoes em que a e b diferem."""
    return float(np.mean(a != b))


def acerto_palavras(verdade, dec):
    """Fracao de palavras (separadas por espaco) decodificadas inteiramente certas."""
    esp = np.where(verdade == IDX[" "])[0]
    lim = np.concatenate([[-1], esp, [len(verdade)]])
    ok, tot = 0, 0
    for i in range(len(lim) - 1):
        s, e = lim[i] + 1, lim[i + 1]
        if e > s:
            tot += 1
            ok += int(np.array_equal(verdade[s:e], dec[s:e]))
    return ok / tot


def texto(seq):
    """Converte indices em string."""
    return "".join(ALF[i] for i in seq)


# ----------------------------------------------------------------------------
def verifica_forca_bruta():
    """assert: forward = soma sobre todos os caminhos e Viterbi = melhor caminho num HMM pequeno (3 estados, T=7)."""
    rng = np.random.default_rng(7)
    N, M, T = 3, 4, 7
    pi = rng.dirichlet(np.ones(N))
    A = rng.dirichlet(np.ones(N), size=N)
    B = rng.dirichlet(np.ones(M), size=N)
    obs = rng.integers(M, size=T)
    melhor, soma = (-1, None), 0.0
    for caminho in itertools.product(range(N), repeat=T):
        p = pi[caminho[0]] * B[caminho[0], obs[0]]
        for t in range(1, T):
            p *= A[caminho[t - 1], caminho[t]] * B[caminho[t], obs[t]]
        soma += p
        if p > melhor[0]:
            melhor = (p, caminho)
    alpha, c = forward(pi, A, B, obs)
    assert abs(np.log(c).sum() - np.log(soma)) < 1e-10, "forward != soma dos caminhos"
    path, lp = viterbi(pi, A, B, obs)
    assert tuple(path) == melhor[1] and abs(lp - np.log(melhor[0])) < 1e-10, "viterbi != forca bruta"
    assert np.log(melhor[0]) <= np.log(soma) + 1e-12
    beta = backward(A, B, obs, c)
    gamma = alpha * beta
    assert np.allclose(gamma.sum(axis=1), 1.0)
    # xi marginaliza em gamma
    W = (B[:, obs[1:]].T * beta[1:]) / c[1:, None]
    xi_t = A[None] * alpha[:-1, :, None] * W[:, None, :]
    assert np.allclose(xi_t.sum(axis=2), gamma[:-1]) and np.allclose(xi_t.sum(axis=1), gamma[1:])
    return soma, melhor[0]


def verifica_segunda_ordem():
    """assert: Viterbi de 2a ordem = forca bruta e = Viterbi de 1a ordem quando o trigrama vem do bigrama."""
    rng = np.random.default_rng(11)
    N, T = 3, 5
    import sys
    mod = sys.modules[__name__]
    antigo = mod.S
    mod.S = N
    try:
        pi = rng.dirichlet(np.ones(N))
        A = rng.dirichlet(np.ones(N), size=N)
        A2 = rng.dirichlet(np.ones(N), size=(N, N))
        B = rng.dirichlet(np.ones(N), size=N)
        obs = rng.integers(N, size=T)
        melhor = (-1, None)
        for cam in itertools.product(range(N), repeat=T):
            p = pi[cam[0]] * B[cam[0], obs[0]] * A[cam[0], cam[1]] * B[cam[1], obs[1]]
            for t in range(2, T):
                p *= A2[cam[t - 2], cam[t - 1], cam[t]] * B[cam[t], obs[t]]
            if p > melhor[0]:
                melhor = (p, cam)
        path, lp = viterbi2(pi, A, A2, B, obs)
    finally:
        mod.S = antigo
    assert tuple(path) == melhor[1] and abs(lp - np.log(melhor[0])) < 1e-10, "viterbi2 != forca bruta"


# ----------------------------------------------------------------------------
def main():
    """Executa os experimentos do projeto 2, imprime as tabelas, faz os assert e grava figuras e saida."""
    t_ini = time.perf_counter()
    soma, pmax = verifica_forca_bruta()
    verifica_segunda_ordem()
    _saida("=== VERIFICACAO ===")
    _saida(f"HMM pequeno (3 estados, T=7): forward == soma de 3^7 caminhos ({soma:.6e}); "
           f"Viterbi == melhor caminho (p={pmax:.6e}); Viterbi 2a ordem == forca bruta.")

    corpus = carrega_corpus()
    corte = int(0.8 * len(corpus))
    treino, teste = corpus[:corte], corpus[corte:]
    pi, A, A2 = modelo_linguagem(treino)
    _saida(f"\ncorpus: {len(corpus)} simbolos (treino {len(treino)}, teste {len(teste)}); estados={S}")
    ent = -np.sum(pi * np.sum(A * np.log2(A), axis=1))
    _saida(f"entropia do bigrama ~ {ent:.2f} bits/letra (uniforme seria {np.log2(S):.2f})")

    # 2a ordem construida a partir do bigrama deve reproduzir a 1a ordem
    p_ = 0.2
    rng = np.random.default_rng(5)
    teste_seq = teste[:400]
    obs_ = corrompe(teste_seq, p_, rng)
    A2_fake = np.broadcast_to(A[None, :, :], (S, S, S)).copy()
    B = matriz_emissao(p_)
    v1, lp1 = viterbi(pi, A, B, obs_)
    v2, lp2 = viterbi2(pi, A, A2_fake, B, obs_)
    assert np.array_equal(v1, v2) and abs(lp1 - lp2) < 1e-8, "viterbi2 com trigrama=bigrama deve igualar viterbi"

    # ---- Varredura da taxa de erro de digitacao ----------------------------
    TAM, NSEM = 3000, 5
    ps = [0.05, 0.10, 0.20, 0.30, 0.40]
    metodos = ["cru", "sem contexto", "Viterbi bigrama", "posterior bigrama", "Viterbi trigrama"]
    res = {m: np.zeros((len(ps), NSEM)) for m in metodos}
    pal = {m: np.zeros((len(ps), NSEM)) for m in metodos}
    amostra = None
    for k, p in enumerate(ps):
        B = matriz_emissao(p)
        for s in range(NSEM):
            rng = np.random.default_rng(100 * k + s)
            ini = s * TAM
            ver = teste[ini:ini + TAM]
            obs = corrompe(ver, p, rng)
            dec = {
                "cru": obs,
                "sem contexto": (pi[:, None] * B).argmax(axis=0)[obs],
                "Viterbi bigrama": viterbi(pi, A, B, obs)[0],
                "posterior bigrama": posterior_decode(pi, A, B, obs),
                "Viterbi trigrama": viterbi2(pi, A, A2, B, obs)[0],
            }
            for m in metodos:
                res[m][k, s] = cer(dec[m], ver)
                pal[m][k, s] = acerto_palavras(ver, dec[m])
            if p == 0.2 and s == 0:
                amostra = (ver, obs, dec["Viterbi bigrama"], dec["Viterbi trigrama"])
    _saida(f"\n=== TAXA DE ERRO POR CARACTERE (CER, %), media +- desvio em {NSEM} trechos de {TAM} caracteres ===")
    _saida(f"{'p':>6}" + "".join(f"{m:>22}" for m in metodos))
    for k, p in enumerate(ps):
        _saida(f"{p:>6.2f}" + "".join(f"{100 * res[m][k].mean():>14.2f} +-{100 * res[m][k].std():>4.2f}" for m in metodos))
    _saida("\n=== PALAVRAS CORRETAS (%), mesmas execucoes ===")
    _saida(f"{'p':>6}" + "".join(f"{m:>22}" for m in metodos))
    for k, p in enumerate(ps):
        _saida(f"{p:>6.2f}" + "".join(f"{100 * pal[m][k].mean():>22.1f}" for m in metodos))
    for k, p in enumerate(ps):
        if p >= 0.10:
            assert res["Viterbi bigrama"][k].mean() < res["cru"][k].mean(), p
            assert res["Viterbi bigrama"][k].mean() < res["sem contexto"][k].mean(), p
    # comparacao pareada bi x tri
    dif = res["Viterbi bigrama"] - res["Viterbi trigrama"]
    _saida(f"\ntrigrama - bigrama (CER, pontos percentuais, pareado): " +
           ", ".join(f"p={p}: {100 * dif[k].mean():+.2f}" for k, p in enumerate(ps)))

    # ---- Erros residuais (p = 0.2) -----------------------------------------
    ver, obs, d_bi, d_tri = amostra
    conf = {}
    for t_, d_ in zip(ver, d_bi):
        if t_ != d_:
            conf[(ALF[t_], ALF[d_])] = conf.get((ALF[t_], ALF[d_]), 0) + 1
    top = sorted(conf.items(), key=lambda kv: -kv[1])[:6]
    _saida("\nconfusoes mais frequentes do Viterbi bigrama (p=0.2, verdade->decodificado; '_'=espaco): " +
           ", ".join(f"{a.replace(' ', '_')}->{b.replace(' ', '_')}:{n}" for (a, b), n in top))
    _saida("amostra (p=0.2):")
    _saida("  verdade : " + texto(ver[:90]))
    _saida("  digitado: " + texto(obs[:90]))
    _saida("  Viterbi : " + texto(d_bi[:90]))
    _saida("  trigrama: " + texto(d_tri[:90]))

    # ---- Baum-Welch ---------------------------------------------------------
    p_real = 0.30
    rng = np.random.default_rng(900)
    ini_em = 5 * TAM
    seq_all = teste[ini_em:ini_em + 12000]
    obs_all = corrompe(seq_all, p_real, rng)
    B_real = matriz_emissao(p_real)
    B0 = np.full((S, S), 0.3 / (S - 1))
    np.fill_diagonal(B0, 0.7)
    A_ini = 0.5 * A + 0.5 / S

    def kl_linhas(Bt, Be):
        """KL media entre as linhas de duas matrizes de emissao (verdadeira Bt, estimada Be)."""
        m = Bt > 0
        return float(np.mean([np.sum(Bt[i][m[i]] * np.log(Bt[i][m[i]] / np.maximum(Be[i][m[i]], 1e-12))) for i in range(S)]))

    # (a) EM livre: B inteira (27x27 = 729 parametros) com A fixa
    _saida(f"\n=== BAUM-WELCH: digitador de taxa de erro DESCONHECIDA (p real = {p_real}) ===")
    _saida("(a) EM livre de B (729 parametros), A do corpus fixa, so texto digitado, sem rotulos:")
    _saida(f"{'T (chars)':>10}{'log P final':>13}{'log P (B real)':>16}{'KL(B real||B)':>15}")
    livre = {}
    for T_em in (2000, 6000, 12000):
        o = obs_all[:T_em]
        _, _, B_x, h = baum_welch(pi, A, B0, o, 40, aprende_A=False, aprende_B=True, aprende_pi=False)
        assert np.all(np.diff(h) > -1e-6), "log-verossimilhanca nao monotona"
        ll_real = float(np.log(forward(pi, A, B_real, o)[1]).sum())
        livre[T_em] = (B_x, h, ll_real)
        _saida(f"{T_em:>10}{h[-1]:>13.1f}{ll_real:>16.1f}{kl_linhas(B_real, B_x):>15.4f}")
    _saida(f"  (B inicial: KL = {kl_linhas(B_real, B0):.4f}). log P final > log P com B real, e KL cresce: o EM livre SUPERAJUSTA.")
    B_livre, h_B = livre[6000][0], livre[6000][1]
    ll_verdade = livre[6000][2]

    # (b) EM com emissao parametrizada: so a taxa p e desconhecida; B(p) tem a forma do teclado.
    #     M-step: p = fracao esperada de posicoes com tecla errada = 1 - mean_t gamma_t(o_t).
    def em_taxa(o, p0=0.10, iters=25):
        """Baum-Welch com B restrita ao teclado: so a taxa p e aprendida (M-step: p = 1 - fracao esperada de acertos). Devolve (p, historico log P, trajetoria de p)."""
        p, hist, tr = p0, [], [p0]
        for _ in range(iters):
            Bp = matriz_emissao(p)
            alpha, c = forward(pi, A, Bp, o)
            gamma = alpha * backward(A, Bp, o, c)
            hist.append(np.log(c).sum())
            p = 1 - float(gamma[np.arange(len(o)), o].mean())
            tr.append(p)
        return p, hist, tr

    _saida("(b) EM com emissao parametrizada pelo teclado (1 parametro, p), A fixa:")
    _saida(f"{'T (chars)':>10}{'p estimado':>12}{'|erro|':>9}")
    for T_em in (500, 2000, 6000, 12000):
        p_hat, h_p, tr_p = em_taxa(obs_all[:T_em])
        assert np.all(np.diff(h_p) > -1e-6)
        _saida(f"{T_em:>10}{p_hat:>12.4f}{abs(p_hat - p_real):>9.4f}")
        if T_em == 6000:
            p_hat6, h_p6 = p_hat, h_p
    assert abs(p_hat6 - p_real) < 0.03

    # (c) EM completo (A, B, pi) a partir so do texto ruidoso
    obs_em = obs_all[:6000]
    pi_f, A_f, B_f, h_full = baum_welch(pi, A_ini, B0, obs_em, 60)
    assert np.all(np.diff(h_full) > -1e-6)
    _saida(f"(c) EM de A, B, pi (6000 chars): log P {h_full[0]:.1f} -> {h_full[-1]:.1f} (monotono); "
           f"|A - A_corpus|_1 por linha: {np.abs(A_f - A).sum(axis=1).mean():.3f} (A inicial: {np.abs(A_ini - A).sum(axis=1).mean():.3f})")

    avali = {}
    for nome, (pi_x, A_x, B_x) in {
        "oracle (p=0.30 conhecido)": (pi, A, B_real),
        "p suposto 0.05 (errado)": (pi, A, matriz_emissao(0.05)),
        "p suposto 0.60 (errado)": (pi, A, matriz_emissao(0.60)),
        "EM de p (teclado, 1 param.)": (pi, A, matriz_emissao(p_hat6)),
        "EM livre de B (729 param.)": (pi, A, B_livre),
        "EM de A, B, pi": (pi_f, A_f, B_f),
    }.items():
        c_ = []
        for s in range(NSEM):
            r_ = np.random.default_rng(777 + s)
            ver_ = teste[s * TAM:(s + 1) * TAM]
            obs_s = corrompe(ver_, p_real, r_)
            c_.append(cer(viterbi(pi_x, A_x, B_x, obs_s)[0], ver_))
        avali[nome] = np.array(c_)
    _saida(f"\nCER do Viterbi bigrama com cada modelo de emissao (p real {p_real}; {NSEM} trechos de {TAM}):")
    for nome, v in avali.items():
        _saida(f"  {nome:<30}{100 * v.mean():>7.2f} +- {100 * v.std():.2f}")
    _saida(f"  {'(texto cru)':<30}{100 * res['cru'][ps.index(0.30)].mean():>7.2f}")
    assert avali["EM de p (teclado, 1 param.)"].mean() < avali["oracle (p=0.30 conhecido)"].mean() + 0.01
    assert avali["EM de p (teclado, 1 param.)"].mean() < avali["p suposto 0.05 (errado)"].mean()
    assert avali["EM de p (teclado, 1 param.)"].mean() < avali["EM livre de B (729 param.)"].mean()

    # ---- Figuras -------------------------------------------------------------
    cores = {"cru": "#888", "sem contexto": "#c84", "Viterbi bigrama": "#26a",
             "posterior bigrama": "#7ad", "Viterbi trigrama": "#2a6"}
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for m in metodos:
        axs[0].errorbar(ps, 100 * res[m].mean(axis=1), yerr=100 * res[m].std(axis=1), marker="o", capsize=3,
                        color=cores[m], label=m, ls="--" if m == "posterior bigrama" else "-")
        axs[1].errorbar(ps, 100 * pal[m].mean(axis=1), yerr=100 * pal[m].std(axis=1), marker="o", capsize=3,
                        color=cores[m], label=m, ls="--" if m == "posterior bigrama" else "-")
    axs[0].set_xlabel("taxa de erro de digitacao p")
    axs[0].set_ylabel("CER (% de caracteres errados)")
    axs[0].set_title("Erro por caractere apos a decodificacao")
    axs[1].set_xlabel("taxa de erro de digitacao p")
    axs[1].set_ylabel("palavras totalmente corretas (%)")
    axs[1].set_title("Palavras corretas")
    axs[0].legend(fontsize=9)
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "02_cer_vs_ruido.png", dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(18, 6.4))
    ax = axs[0]
    ax.plot(range(1, len(h_B) + 1), h_B, "o-", ms=3, color="#26a", label="EM livre de B (40 iter.)")
    ax.plot(range(1, len(h_p6) + 1), h_p6, "^-", ms=3, color="#c84", label="EM de p (teclado, 25 iter.)")
    ax.plot(range(1, len(h_full) + 1), h_full, "s-", ms=3, color="#2a6", label="EM de A, B, pi (60 iter.)")
    ax.axhline(ll_verdade, color="k", ls=":", label="parametros verdadeiros (A do corpus, B real)")
    ax.set_xlabel("iteracao de Baum-Welch")
    ax.set_ylabel("log P(observacoes)")
    ax.set_title("Log-verossimilhanca monotona crescente")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    rotulos = list(ALF.replace(" ", "_"))
    for ax, M_, tit in ((axs[1], B_real, "B real (teclado, p = 0.30)"), (axs[2], B_livre, "B aprendida pelo EM livre (T=6000): superajuste"),
                        (axs[3], A, "A: bigrama de letras do corpus")):
        im = ax.imshow(M_, cmap="viridis")
        ax.set_xticks(range(S), rotulos, fontsize=8)
        ax.set_yticks(range(S), rotulos, fontsize=8)
        ax.set_title(tit)
        ax.set_xlabel("tecla digitada" if M_ is not A else "letra seguinte")
        ax.set_ylabel("letra pretendida" if M_ is not A else "letra atual")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGS / "02_baum_welch.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4.6))
    L, linhas_n = 70, 3
    nomes = ["pretendido", "digitado", "Viterbi bigrama", "Viterbi trigrama"]
    for r in range(linhas_n):
        trechos = [ver[r * L:(r + 1) * L], obs[r * L:(r + 1) * L], d_bi[r * L:(r + 1) * L], d_tri[r * L:(r + 1) * L]]
        for q, tr in enumerate(trechos):
            y = -(r * 5.0 + q)
            if r == 0:
                pass
            ax.text(-2.5, y, nomes[q], fontsize=9, ha="right", va="center")
            for j, ch in enumerate(tr):
                err_ = q > 0 and ch != trechos[0][j]
                ax.text(j, y, ALF[ch].replace(" ", "·"), family="monospace", fontsize=9, ha="center", va="center",
                        color="#c00" if err_ else "black", fontweight="bold" if err_ else "normal")
    ax.set_xlim(-14, L)
    ax.set_ylim(-(linhas_n * 5.0), 1)
    ax.axis("off")
    ax.set_title("Exemplo de decodificacao (p = 0.20); vermelho = caractere diferente do pretendido; · = espaco")
    fig.tight_layout()
    fig.savefig(FIGS / "02_exemplo.png", dpi=220)
    plt.close(fig)

    _saida(f"\nTodos os assert passaram. (tempo total ~ {time.perf_counter() - t_ini:.0f} s)")
    (FIGS / "02_saida.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
