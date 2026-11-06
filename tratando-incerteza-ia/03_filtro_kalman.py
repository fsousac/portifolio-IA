"""
Projeto 3 - Filtro de Kalman: rastreando um drone que passa por um tunel sem sinal.

PROBLEMA
    Um drone voa numa trajetoria em "8" (Lissajous) a ~10 m/s. Um receptor de posicao
    (tipo GPS, desvio 20 m) mede (x, y) uma vez por segundo, mas (i) perde uma fracao
    das leituras ao acaso e (ii) fica 40 s sem sinal ao passar por um tunel. Estimar a
    posicao e a velocidade verdadeiras, e dizer o quao incerta e a estimativa.

MODELAGEM
    Estado x = [px, py, vx, vy]; modelo de velocidade constante com aceleracao
    aleatoria (discrete white-noise acceleration): x_{k+1} = F x_k + G a_k,
    a_k ~ N(0, q I2), Q = q G G^T.  Medida z_k = H x_k + v_k, v_k ~ N(0, s^2 I2).
    A trajetoria real e curva (aceleracao ate 0.6 m/s^2), logo o modelo e uma
    APROXIMACAO: o filtro precisa de um q que absorva a manobra.

ALGORITMOS IMPLEMENTADOS A MAO (numpy)
    - Filtro de Kalman (predict/update, forma de Joseph), com falta de medida = so predict.
    - Suavizador RTS (Rauch-Tung-Striebel): usa tambem as medidas FUTURAS.
    - Filtro de particulas (bootstrap, reamostragem sistematica) com o mesmo modelo.
    - Linhas de base: medida crua (mantem o ultimo valor) e media movel de 5 passos.
    - Testes de CONSISTENCIA: NEES (erro vs covariancia estimada) e NIS (inovacao) com
      limites qui-quadrado, e sensibilidade a Q e R (conceito alem do basico).

VERDADE DE REFERENCIA / AUTOVERIFICACAO
    - Estimativa do KF e do RTS == solucao MAP em lote (minimos quadrados ponderados
      sobre a trajetoria inteira), a unica estimativa otima de um sistema linear
      gaussiano. Esta e uma verificacao independente do algoritmo recursivo.
    - Ganho estacionario do KF == formulas fechadas do filtro alfa-beta (Kalata).
    - P simetrica e semidefinida positiva em todos os passos; na falta de medida P cresce.
    - NEES medio do KF bem ajustado dentro dos limites 95%; descalibrado fora deles.
    - Pareado em 50 sementes: RMSE(RTS) < RMSE(KF) < RMSE(media movel) e < RMSE(crua).
    - Filtro de particulas se aproxima do KF quando N cresce.

Execucao: python 03_filtro_kalman.py  (gera figs/03_*.png e figs/03_saida.txt)
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fontes legiveis quando a figura e reduzida a largura de uma pagina A4
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
import numpy as np
from matplotlib.patches import Ellipse

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
# Modelo
# ----------------------------------------------------------------------------
DT = 1.0
F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], float)
G = np.array([[DT ** 2 / 2, 0], [0, DT ** 2 / 2], [DT, 0], [0, DT]], float)
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)


def Q_de(q):
    """Covariancia do ruido de processo Q = q G G' (aceleracao branca de intensidade q)."""
    return q * G @ G.T


def R_de(s):
    """Covariancia do ruido de medida R = s^2 I (2x2)."""
    return s ** 2 * np.eye(2)


# ----------------------------------------------------------------------------
# Filtro de Kalman e suavizador RTS
# ----------------------------------------------------------------------------
def kalman(z, mask, q, s, m0, P0, joseph=True, dtype=np.float64):
    """z: (T,2); mask: (T,) bool (True = medida disponivel). m0,P0 = priori ANTES do
    passo 0. Retorna dict com predicoes (xp,Pp), filtradas (xf,Pf) e inovacoes."""
    Fm, Hm, Qm, Rm = (a.astype(dtype) for a in (F, H, Q_de(q), R_de(s)))
    T = len(z)
    xp = np.zeros((T, 4), dtype)
    Pp = np.zeros((T, 4, 4), dtype)
    xf, Pf = np.zeros_like(xp), np.zeros_like(Pp)
    nis = np.full(T, np.nan)
    x, P = m0.astype(dtype), P0.astype(dtype)
    I4 = np.eye(4, dtype=dtype)
    for k in range(T):
        if k > 0:
            x = Fm @ x
            P = Fm @ P @ Fm.T + Qm
        xp[k], Pp[k] = x, P
        if mask[k]:
            nu = z[k].astype(dtype) - Hm @ x   # inovacao; sem medida (mask False) so ha predicao e P cresce
            Sk = Hm @ P @ Hm.T + Rm
            K = P @ Hm.T @ np.linalg.inv(Sk)
            x = x + K @ nu
            # Forma de Joseph: P = (I-KH) P (I-KH)' + K R K' mantem P simetrica e semidefinida
            # positiva mesmo com erro de arredondamento; (I-KH)P pode perde-la (ver ablacao float32).
            if joseph:
                A_ = I4 - K @ Hm
                P = A_ @ P @ A_.T + K @ Rm @ K.T
            else:
                P = (I4 - K @ Hm) @ P
            nis[k] = float(nu @ np.linalg.solve(Sk, nu))
        xf[k], Pf[k] = x, P
    return dict(xp=xp, Pp=Pp, xf=xf, Pf=Pf, nis=nis)


def rts(res):
    """Suavizador de Rauch-Tung-Striebel (passagem para tras)."""
    xf, Pf, xp, Pp = res["xf"], res["Pf"], res["xp"], res["Pp"]
    T = len(xf)
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(T - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps


# ----------------------------------------------------------------------------
# Filtro de particulas (bootstrap) com o mesmo modelo
# ----------------------------------------------------------------------------
def particulas(z, mask, q, s, m0, P0, N, rng):
    """Filtro de particulas bootstrap (reamostragem sistematica quando ESS < N/2). Entrada: medidas z, mascara, q, s, priori (m0,P0), N, rng. Saida: media filtrada (T x 4)."""
    T = len(z)
    # Inicializacao: priori de 4 dimensoes com desvio de 500 m nao deixa quase nenhuma
    # particula perto da primeira medida (desvio 20 m) e os pesos degeneram. Por isso
    # as particulas nascem da posteriori gaussiana exata apos a 1a medida (passo 0 ja
    # incorporado). Isso da ao filtro de particulas a mesma informacao inicial do KF.
    S0 = H @ P0 @ H.T + R_de(s)
    K0 = P0 @ H.T @ np.linalg.inv(S0)
    m1, P1 = m0 + K0 @ (z[0] - H @ m0), (np.eye(4) - K0 @ H) @ P0
    X = rng.multivariate_normal(m1, (P1 + P1.T) / 2, size=N)
    w = np.full(N, 1.0 / N)
    est = np.zeros((T, 4))
    sq = np.sqrt(q)
    for k in range(T):
        if k > 0:
            X = X @ F.T + (rng.standard_normal((N, 2)) * sq) @ G.T
        if mask[k] and k > 0:
            d2 = ((z[k] - X[:, :2]) ** 2).sum(axis=1)
            w = w * np.exp(-0.5 * (d2 - d2.min()) / s ** 2)
            w = w / w.sum()
        est[k] = w @ X
        if 1.0 / np.sum(w ** 2) < N / 2:  # ESS baixo -> reamostragem sistematica
            pos = (rng.random() + np.arange(N)) / N
            idx = np.minimum(np.searchsorted(np.cumsum(w), pos), N - 1)
            X, w = X[idx], np.full(N, 1.0 / N)
    return est


# ----------------------------------------------------------------------------
# Linhas de base
# ----------------------------------------------------------------------------
def crua(z, mask):
    """Linha de base: usa a ultima medida disponivel como posicao."""
    out = np.zeros((len(z), 2))
    ult = z[np.argmax(mask)]
    for k in range(len(z)):
        if mask[k]:
            ult = z[k]
        out[k] = ult
    return out


def media_movel(z, mask, w=5):
    """Linha de base: media das medidas disponiveis nos ultimos w passos (retem o ultimo valor se nao houver)."""
    out = np.zeros((len(z), 2))
    ult = z[np.argmax(mask)]
    for k in range(len(z)):
        idx = [j for j in range(max(0, k - w + 1), k + 1) if mask[j]]
        if idx:
            ult = z[idx].mean(axis=0)
        out[k] = ult
    return out


# ----------------------------------------------------------------------------
# Cenarios
# ----------------------------------------------------------------------------
T_B = 300
TUNEL = (120, 160)
SIGMA = 20.0
M0_B = np.array([300.0, 150.0, 0.0, 0.0])
P0_B = np.diag([500.0 ** 2, 500.0 ** 2, 15.0 ** 2, 15.0 ** 2])


def trajetoria_oito():
    """Trajetoria verdadeira em 8 (Lissajous): estados [px, py, vx, vy] em cada passo."""
    t = np.arange(T_B) * DT
    px, py = 300 + 400 * np.sin(0.025 * t + 0.5), 150 + 250 * np.sin(0.05 * t)
    vx, vy = 400 * 0.025 * np.cos(0.025 * t + 0.5), 250 * 0.05 * np.cos(0.05 * t)
    return np.stack([px, py, vx, vy], axis=1)


def medidas_B(verd, taxa_perda, rng):
    """Medidas ruidosas da trajetoria, com perda aleatoria e o tunel (sem sinal); devolve (z, mascara)."""
    z = verd[:, :2] + SIGMA * rng.standard_normal((len(verd), 2))
    mask = rng.random(len(verd)) >= taxa_perda
    mask[TUNEL[0]:TUNEL[1]] = False
    mask[0] = True
    return z, mask


def rmse(est_pos, verd, sel=None):
    """Raiz do erro quadratico medio de posicao; sel opcional restringe os passos."""
    e = ((est_pos - verd[:, :2]) ** 2).sum(axis=1)
    if sel is not None:
        e = e[sel]
    return float(np.sqrt(e.mean()))


def simula_modelo(rng, T, q, s, m0, P0, perda):
    """Dados GERADOS pelo proprio modelo (para testes de consistencia)."""
    x = m0 + np.linalg.cholesky(P0) @ rng.standard_normal(4)
    X = np.zeros((T, 4))
    for k in range(T):
        if k > 0:
            x = F @ x + G @ (np.sqrt(q) * rng.standard_normal(2))
        X[k] = x
    z = X[:, :2] + s * rng.standard_normal((T, 2))
    return X, z, rng.random(T) >= perda


# ----------------------------------------------------------------------------
# Verificacoes independentes
# ----------------------------------------------------------------------------
def map_em_lote(z, mask, q, s, m0, P0):
    """Minimos quadrados ponderados sobre a trajetoria inteira: incognitas
    u = [x0, a_0..a_{T-2}]; retorna a media a posteriori de todos os estados."""
    T = len(z)
    nu = 4 + 2 * (T - 1)
    Phi = np.zeros((T, 4, nu))
    Phi[0, :, :4] = np.eye(4)
    for k in range(1, T):
        Phi[k] = F @ Phi[k - 1]
        Phi[k, :, 4 + 2 * (k - 1):4 + 2 * k] += G
    L = np.linalg.cholesky(np.linalg.inv(P0))
    A = [np.hstack([L.T, np.zeros((4, nu - 4))])]
    b = [L.T @ m0]
    A.append(np.hstack([np.zeros((nu - 4, 4)), np.eye(nu - 4) / np.sqrt(q)]))
    b.append(np.zeros(nu - 4))
    for k in range(T):
        if mask[k]:
            A.append(H @ Phi[k] / s)
            b.append(z[k] / s)
    A, b = np.vstack(A), np.concatenate(b)
    u = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.einsum("kij,j->ki", Phi, u)


def verifica_kalman():
    """assert: KF e RTS = solucao MAP em lote, ganho estacionario = alfa-beta de Kalata, P simetrica PSD."""
    rng = np.random.default_rng(3)
    q, s = 0.3, 5.0
    m0, P0 = np.zeros(4), np.diag([30.0, 30.0, 4.0, 4.0]) ** 2
    X, z, mask = simula_modelo(rng, 16, q, s, m0, P0, 0.3)
    mask[0] = True
    r = kalman(z, mask, q, s, m0, P0)
    xs, _ = rts(r)
    X_map = map_em_lote(z, mask, q, s, m0, P0)
    assert np.allclose(xs, X_map, atol=1e-7), "RTS != MAP em lote"
    assert np.allclose(r["xf"][-1], X_map[-1], atol=1e-7), "KF final != MAP em lote"
    # filtrada em k = KF com so os dados ate k
    kmid = 9
    X_mid = map_em_lote(z[:kmid + 1], mask[:kmid + 1], q, s, m0, P0)
    assert np.allclose(r["xf"][kmid], X_mid[-1], atol=1e-7)
    return float(np.abs(xs - X_map).max())


def verifica_alfa_beta():
    """Ganho estacionario (so posicao, 1 eixo) vs. formulas de Kalata:
    lambda = sqrt(q) T^2 / s;  r = (4 + lam - sqrt(8 lam + lam^2)) / 4;
    alfa = 1 - r^2;  beta = 2 (2 - alfa) - 4 sqrt(1 - alfa)."""
    out = []
    F1 = np.array([[1, DT], [0, 1.0]])
    G1 = np.array([[DT ** 2 / 2], [DT]])
    H1 = np.array([[1.0, 0.0]])
    for q, s in ((0.01, 10.0), (0.3, 5.0), (4.0, 20.0)):
        P = np.diag([100.0, 10.0])
        for _ in range(5000):
            Pp = F1 @ P @ F1.T + q * G1 @ G1.T
            K = Pp @ H1.T / (H1 @ Pp @ H1.T + s ** 2)
            P = (np.eye(2) - K @ H1) @ Pp
        lam = np.sqrt(q) * DT ** 2 / s
        r = (4 + lam - np.sqrt(8 * lam + lam ** 2)) / 4
        al = 1 - r ** 2
        be = 2 * (2 - al) - 4 * np.sqrt(1 - al)
        assert abs(K[0, 0] - al) < 1e-6 and abs(K[1, 0] * DT - be) < 1e-6, (q, s, K.ravel(), al, be)
        out.append((q, s, float(K[0, 0]), al, float(K[1, 0]), be))
    return out


def chi2_lim(nu, z):
    """Quantil aproximado de qui-quadrado (Wilson-Hilferty)."""
    return nu * (1 - 2 / (9 * nu) + z * np.sqrt(2 / (9 * nu))) ** 3


def elipse(ax, centro, P2, cor, alpha=0.25):
    """Desenha a elipse de confianca de 95% (qui-quadrado 2 g.l. = 5.991) de uma covariancia 2x2."""
    w, V = np.linalg.eigh(P2)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(centro, 2 * np.sqrt(5.991 * w[1]), 2 * np.sqrt(5.991 * w[0]), angle=ang,
                         fill=True, fc=cor, ec=cor, alpha=alpha, lw=0.8))


# ----------------------------------------------------------------------------
def main():
    """Executa os experimentos do projeto 3, imprime as tabelas, faz os assert e grava figuras e saida."""
    t_ini = time.perf_counter()
    dmax = verifica_kalman()
    ab = verifica_alfa_beta()
    _saida("=== VERIFICACAO ===")
    _saida(f"KF filtrado e RTS == solucao MAP em lote (T=16, 30% de perda): maior diferenca {dmax:.2e}")
    _saida("ganho estacionario do KF vs. filtro alfa-beta (Kalata):")
    for q, s, a1, a2, b1, b2 in ab:
        _saida(f"  q={q:<5} s={s:<5} alfa KF={a1:.6f} formula={a2:.6f} | beta KF={b1:.6f} formula={b2:.6f}")

    verd = trajetoria_oito()
    # q nominal = 0.5 (desvio de aceleracao 0.7 m/s^2): maior que a aceleracao real (~0.3 m/s^2 RMS)
    # porque o modelo de velocidade constante precisa absorver as curvas; ver ablacao e grade abaixo.
    Q0 = 0.5

    # ---- Um voo exemplo ----------------------------------------------------
    rng = np.random.default_rng(42)
    z, mask = medidas_B(verd, 0.2, rng)
    r = kalman(z, mask, Q0, SIGMA, M0_B, P0_B)
    xs, Ps = rts(r)
    for k in range(T_B):
        Pk = r["Pf"][k]
        assert np.allclose(Pk, Pk.T, atol=1e-9) and np.linalg.eigvalsh((Pk + Pk.T) / 2).min() > -1e-9
    traco = np.array([np.trace(P[:2, :2]) for P in r["Pf"]])
    assert traco[TUNEL[1] - 1] > 10 * traco[TUNEL[0] - 1], "P deveria crescer sem medida"
    sig_pos = np.sqrt(traco / 2)
    _saida(f"\n=== VOO EXEMPLO (semente 42): {int((~mask).sum())} de {T_B} leituras ausentes, tunel = passos {TUNEL[0]}-{TUNEL[1] - 1} ===")
    _saida(f"incerteza de posicao do KF (desvio por eixo, m): antes do tunel {sig_pos[TUNEL[0] - 1]:.1f}, "
           f"fim do tunel {sig_pos[TUNEL[1] - 1]:.1f}, 5 passos depois {sig_pos[TUNEL[1] + 4]:.1f}; "
           f"ruido da medida {SIGMA:.0f}")
    erro_t = np.hypot(*(r["xf"][TUNEL[1] - 1, :2] - verd[TUNEL[1] - 1, :2]))
    _saida(f"erro real do KF no fim do tunel: {erro_t:.1f} m")

    # ---- Comparacao pareada -------------------------------------------------
    NS, N_PART = 50, 2000
    nomes = ["crua (hold)", "media movel 5", "Kalman (filtro)", "Kalman + RTS", f"particulas N={N_PART}"]
    tot = np.zeros((len(nomes), NS))
    tun = np.zeros((len(nomes), NS))
    sel_t = np.zeros(T_B, bool)
    sel_t[TUNEL[0]:TUNEL[1]] = True
    for s in range(NS):
        rng = np.random.default_rng(10_000 + s)
        z, mask = medidas_B(verd, 0.2, rng)
        rr = kalman(z, mask, Q0, SIGMA, M0_B, P0_B)
        ests = [crua(z, mask), media_movel(z, mask), rr["xf"][:, :2], rts(rr)[0][:, :2],
                particulas(z, mask, Q0, SIGMA, M0_B, P0_B, N_PART, rng)[:, :2]]
        for i, e in enumerate(ests):
            tot[i, s] = rmse(e, verd)
            tun[i, s] = rmse(e, verd, sel_t)
    _saida(f"\n=== RMSE DE POSICAO (m) em {NS} voos, 20% de perda aleatoria + tunel; mesmas medidas para todos ===")
    _saida(f"{'metodo':<20}{'RMSE total':>18}{'RMSE no tunel':>20}{'melhor que KF (voos)':>24}")
    for i, n in enumerate(nomes):
        _saida(f"{n:<20}{tot[i].mean():>10.2f} +-{tot[i].std():>5.2f}{tun[i].mean():>12.2f} +-{tun[i].std():>6.2f}"
               f"{int((tot[i] < tot[2]).sum()):>16}/{NS}")
    assert tot[3].mean() < tot[2].mean() < tot[1].mean() and tot[2].mean() < tot[0].mean()
    assert np.all(tot[2] < tot[1]) and np.all(tot[3] <= tot[2])
    ganho = 100 * (1 - tot[2].mean() / tot[0].mean())
    _saida(f"ganho do KF sobre a medida crua: {ganho:.1f}% do RMSE; RTS sobre KF: {100 * (1 - tot[3].mean() / tot[2].mean()):.1f}%")

    # ---- Perda de dados ------------------------------------------------------
    taxas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    NS2 = 30
    curv = {n: np.zeros((len(taxas), NS2)) for n in nomes[:4]}
    for k, d in enumerate(taxas):
        for s in range(NS2):
            rng = np.random.default_rng(20_000 + 100 * k + s)
            z, mask = medidas_B(verd, d, rng)
            rr = kalman(z, mask, Q0, SIGMA, M0_B, P0_B)
            for n, e in zip(nomes[:4], [crua(z, mask), media_movel(z, mask), rr["xf"][:, :2], rts(rr)[0][:, :2]]):
                curv[n][k, s] = rmse(e, verd)
    _saida(f"\n=== RMSE (m) x taxa de perda aleatoria de leituras (alem do tunel), {NS2} voos por taxa ===")
    _saida(f"{'perda':>7}" + "".join(f"{n:>18}" for n in nomes[:4]))
    for k, d in enumerate(taxas):
        _saida(f"{d:>7.1f}" + "".join(f"{curv[n][k].mean():>18.2f}" for n in nomes[:4]))
    assert np.all(curv[nomes[2]].mean(axis=1) < curv[nomes[1]].mean(axis=1))

    # ---- Particulas: convergencia para o KF ----------------------------------
    Nps = [20, 100, 500, 2500, 12500]
    dist = np.zeros((len(Nps), 10))
    for s in range(10):
        rng = np.random.default_rng(30_000 + s)
        z, mask = medidas_B(verd, 0.2, rng)
        rr = kalman(z, mask, Q0, SIGMA, M0_B, P0_B)
        for i, Np in enumerate(Nps):
            pf = particulas(z, mask, Q0, SIGMA, M0_B, P0_B, Np, np.random.default_rng(31_000 + 100 * s + i))
            dist[i, s] = np.sqrt(np.mean(((pf[:, :2] - rr["xf"][:, :2]) ** 2).sum(axis=1)))
    _saida("\n=== FILTRO DE PARTICULAS: distancia RMS a media do KF (m), 10 voos ===")
    _saida(f"{'N':>8}{'dist. a media do KF':>22}")
    for i, Np in enumerate(Nps):
        _saida(f"{Np:>8}{dist[i].mean():>14.3f} +-{dist[i].std():.3f}")
    assert dist[-1].mean() < dist[0].mean() / 5

    # ---- Ablacao: q fisico (0.1) x q com folga (0.5) no KF e nas particulas -----
    _saida("\n=== ABLACAO: ruido de processo q, KF vs. particulas (N=500), 10 voos, RMSE total (m) ===")
    _saida(f"{'q':>6}{'KF':>10}{'particulas':>13}")
    abq = {}
    for qq in (0.1, 0.5):
        a_, b_ = [], []
        for s in range(10):
            rng = np.random.default_rng(60_000 + s)
            z, mask = medidas_B(verd, 0.2, rng)
            a_.append(rmse(kalman(z, mask, qq, SIGMA, M0_B, P0_B)["xf"][:, :2], verd))
            b_.append(rmse(particulas(z, mask, qq, SIGMA, M0_B, P0_B, 500, rng)[:, :2], verd))
        abq[qq] = (np.mean(a_), np.mean(b_))
        _saida(f"{qq:>6}{abq[qq][0]:>10.1f}{abq[qq][1]:>13.1f}")
    assert abq[0.1][1] > 3 * abq[0.1][0] and abq[0.5][1] < 2 * abq[0.5][0]

    # ---- Consistencia (NEES / NIS) em dados gerados pelo modelo --------------
    q_m, s_m, T_m, M_m, perda_m = 0.5, 10.0, 100, 300, 0.2
    m0m, P0m = np.zeros(4), np.diag([25.0, 25.0, 5.0, 5.0]) ** 2
    variantes = {"bem ajustado (q, s verdadeiros)": (q_m, s_m), "q 100x menor (otimista)": (q_m / 100, s_m),
                 "q 100x maior (pessimista)": (q_m * 100, s_m), "s 3x maior (R pessimista)": (q_m, 3 * s_m)}
    nees = {n: np.zeros((M_m, T_m)) for n in variantes}
    nis = {n: np.full((M_m, T_m), np.nan) for n in variantes}
    for m in range(M_m):
        rng = np.random.default_rng(40_000 + m)
        X, z, mask = simula_modelo(rng, T_m, q_m, s_m, m0m, P0m, perda_m)
        for n, (qf, sf) in variantes.items():
            rr = kalman(z, mask, qf, sf, m0m, P0m)
            e = X - rr["xf"]
            nees[n][m] = np.einsum("ki,kij,kj->k", e, np.linalg.inv(rr["Pf"]), e)
            nis[n][m] = rr["nis"]
    lo, hi = chi2_lim(4 * M_m, -1.96) / M_m, chi2_lim(4 * M_m, 1.96) / M_m
    _saida(f"\n=== CONSISTENCIA em dados do proprio modelo: {M_m} execucoes de {T_m} passos, 20% de perda ===")
    _saida(f"NEES medio esperado = 4 (limites 95% da media sobre {M_m} execucoes: [{lo:.2f}, {hi:.2f}]); NIS esperado = 2")
    _saida(f"{'filtro':<34}{'NEES medio':>12}{'% passos nos limites':>22}{'NIS medio':>11}")
    resumo = {}
    for n in variantes:
        medio = nees[n].mean(axis=0)
        dentro = 100 * np.mean((medio >= lo) & (medio <= hi))
        resumo[n] = (nees[n].mean(), dentro, np.nanmean(nis[n]))
        _saida(f"{n:<34}{nees[n].mean():>12.2f}{dentro:>22.0f}{np.nanmean(nis[n]):>11.2f}")
    n_ok, n_obs = variantes and "bem ajustado (q, s verdadeiros)", int(np.isfinite(nis["bem ajustado (q, s verdadeiros)"]).sum())
    lo_i, hi_i = chi2_lim(2 * n_obs, -1.96) / n_obs, chi2_lim(2 * n_obs, 1.96) / n_obs
    assert lo <= nees[n_ok].mean() <= hi, "NEES do filtro ajustado fora dos limites"
    assert lo_i <= np.nanmean(nis[n_ok]) <= hi_i, "NIS do filtro ajustado fora dos limites"
    assert nees["q 100x menor (otimista)"].mean() > 2 * hi, "filtro otimista deveria ter NEES alto"
    assert nees["q 100x maior (pessimista)"].mean() < lo, "filtro pessimista deveria ter NEES baixo"

    # ---- Sensibilidade a Q e R no cenario do drone ---------------------------
    escalas = [0.01, 0.1, 1, 10, 100]
    grade = np.zeros((len(escalas), len(escalas)))
    NS3 = 20
    dados = []
    for s in range(NS3):
        z, mask = medidas_B(verd, 0.2, np.random.default_rng(50_000 + s))
        dados.append((z, mask))
    for i, eq in enumerate(escalas):
        for j, er in enumerate(escalas):
            v = [rmse(kalman(z, mask, Q0 * eq, SIGMA * np.sqrt(er), M0_B, P0_B)["xf"][:, :2], verd) for z, mask in dados]
            grade[i, j] = np.mean(v)
    _saida(f"\n=== SENSIBILIDADE: RMSE (m) do KF no drone, q_nominal={Q0} e s_nominal={SIGMA:.0f}; linhas = fator em q, colunas = fator em s^2 ===")
    _saida(f"{'q x \\ s^2 x':>13}" + "".join(f"{e:>9g}" for e in escalas))
    for i, eq in enumerate(escalas):
        _saida(f"{eq:>13g}" + "".join(f"{grade[i, j]:>9.2f}" for j in range(len(escalas))))
    i_b, j_b = np.unravel_index(grade.argmin(), grade.shape)
    _saida(f"melhor da grade: q x {escalas[i_b]:g}, s^2 x {escalas[j_b]:g} -> {grade[i_b, j_b]:.2f} m; "
           f"nominal (1,1): {grade[2, 2]:.2f} m; pior: {grade.max():.2f} m")
    assert grade[2, 2] < 1.1 * grade.min() and grade.max() > 1.5 * grade.min()

    # ---- Ablacao: forma de Joseph (float32) ----------------------------------
    rng = np.random.default_rng(77)
    zz, mm = medidas_B(verd, 0.0, rng)
    mm[:] = True
    s_pequeno, q_pequeno = 0.05, 1e-6
    zz = verd[:, :2] + s_pequeno * rng.standard_normal((T_B, 2))
    _saida("\n=== ABLACAO: forma de Joseph x (I-KH)P em float32, medida quase exata (s=0.05, q=1e-6, 300 passos) ===")
    abl = {}
    for nome, jos in (("Joseph", True), ("(I-KH)P", False)):
        rr = kalman(zz, mm, q_pequeno, s_pequeno, M0_B, P0_B, joseph=jos, dtype=np.float32)
        assim = max(float(np.abs(P - P.T).max() / max(np.abs(P).max(), 1e-30)) for P in rr["Pf"])
        mineig = min(float(np.linalg.eigvalsh((P.astype(float) + P.astype(float).T) / 2).min()) for P in rr["Pf"])
        abl[nome] = (assim, mineig)
        _saida(f"  {nome:<9} assimetria relativa maxima {assim:.2e}   menor autovalor de P {mineig:.2e}")

    assert abl["(I-KH)P"][0] > 100 * abl["Joseph"][0], "forma simples deveria perder simetria"

    # ---- Figuras ---------------------------------------------------------------
    rng = np.random.default_rng(42)
    z, mask = medidas_B(verd, 0.2, rng)
    r = kalman(z, mask, Q0, SIGMA, M0_B, P0_B)
    xs, Ps = rts(r)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    ax.plot(verd[:, 0], verd[:, 1], "k-", lw=1.6, label="trajetoria real")
    ax.plot(verd[TUNEL[0]:TUNEL[1], 0], verd[TUNEL[0]:TUNEL[1], 1], "-", color="#c33", lw=4, alpha=0.35, label="tunel (sem sinal)")
    ax.plot(z[mask, 0], z[mask, 1], ".", color="#999", ms=4, label="medidas")
    ax.plot(r["xf"][8:, 0], r["xf"][8:, 1], "-", color="#26a", lw=1.3, label="Kalman (filtro, a partir de 8 s)")
    ax.plot(xs[8:, 0], xs[8:, 1], "--", color="#2a6", lw=1.3, label="Kalman + RTS")
    for k in range(10, T_B, 10):
        elipse(ax, r["xf"][k, :2], r["Pf"][k][:2, :2], "#26a", 0.25 if not (TUNEL[0] <= k < TUNEL[1]) else 0.18)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Voo exemplo: elipses = regiao de 95% da posicao estimada (a cada 10 s)")
    ax.legend(fontsize=9, loc="lower right")
    ax = axs[1]
    t = np.arange(T_B)
    ax.fill_between([TUNEL[0], TUNEL[1]], 0, 1, color="#c33", alpha=0.12, transform=ax.get_xaxis_transform())
    ax.plot(t, np.hypot(*(r["xf"][:, :2] - verd[:, :2]).T), color="#26a", label="erro real do KF")
    ax.plot(t, np.hypot(*(xs[:, :2] - verd[:, :2]).T), "--", color="#2a6", label="erro real do RTS")
    ax.plot(t, 2.45 * sig_pos, color="#26a", ls=":", label="raio 95% estimado pelo KF (2.45 sigma)")
    ax.set_xlabel("passo (s)")
    ax.set_ylabel("erro de posicao (m)")
    ax.set_title("Erro e incerteza ao longo do tempo")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "03_trajetoria.png", dpi=220)
    plt.close(fig)

    cores = ["#999", "#c84", "#26a", "#2a6", "#a4c"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 5.2))
    ax = axs[0]
    x = np.arange(len(nomes))
    ax.bar(x - 0.2, tot.mean(axis=1), 0.4, yerr=tot.std(axis=1), color=cores, label="total", capsize=3)
    ax.bar(x + 0.2, tun.mean(axis=1), 0.4, yerr=tun.std(axis=1), color=cores, alpha=0.45, hatch="//", label="so no tunel", capsize=3)
    ax.set_xticks(x, [n.replace(" ", "\n", 1) for n in nomes], fontsize=9)
    ax.set_ylabel("RMSE de posicao (m)")
    ax.set_title(f"{NS} voos: cheio = total, hachurado = tunel")
    ax.grid(alpha=0.3, axis="y")
    ax = axs[1]
    for n, c in zip(nomes[:4], cores):
        m = curv[n].mean(axis=1)
        ax.errorbar(taxas, m, yerr=curv[n].std(axis=1) / np.sqrt(NS2), marker="o", color=c, label=n, capsize=3)
    ax.set_xlabel("taxa de perda aleatoria de leituras")
    ax.set_ylabel("RMSE de posicao (m)")
    ax.set_title("Degradacao com a perda de dados")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax = axs[2]
    ax.loglog(Nps, dist.mean(axis=1), "o-", color="#a4c", label="particulas vs. KF")
    ax.loglog(Nps, dist[0].mean() * np.sqrt(Nps[0] / np.array(Nps)), "k:", label=r"$\propto 1/\sqrt{N}$")
    ax.set_xlabel("numero de particulas N")
    ax.set_ylabel("distancia RMS a media do KF (m)")
    ax.set_title("Filtro de particulas converge ao KF")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "03_comparacao.png", dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 5.2))
    ax = axs[0]
    cs = ["#2a6", "#d62", "#26a", "#a4c"]
    for (n, v), c in zip(nees.items(), cs):
        ax.semilogy(np.arange(T_m), v.mean(axis=0), color=c, label=n)
    ax.axhspan(lo, hi, color="k", alpha=0.15, label="limites 95%")
    ax.axhline(4, color="k", lw=0.8, ls=":")
    ax.set_xlabel("passo")
    ax.set_ylabel(f"NEES medio ({M_m} execucoes)")
    ax.set_title("Consistencia: NEES (esperado 4)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax = axs[1]
    for (n, v), c in zip(nis.items(), cs):
        ax.semilogy(np.arange(T_m), np.nanmean(v, axis=0), color=c, label=n, lw=0.9)
    ax.axhline(2, color="k", lw=0.8, ls=":")
    ax.set_xlabel("passo")
    ax.set_ylabel("NIS medio")
    ax.set_title("Consistencia: NIS (esperado 2)")
    ax.grid(alpha=0.3, which="both")
    ax = axs[2]
    im = ax.imshow(grade, origin="lower", cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(escalas)), [f"{e:g}" for e in escalas])
    ax.set_yticks(range(len(escalas)), [f"{e:g}" for e in escalas])
    for i in range(len(escalas)):
        for j in range(len(escalas)):
            ax.text(j, i, f"{grade[i, j]:.0f}", ha="center", va="center", fontsize=9, color="w" if grade[i, j] > grade.mean() else "k")
    ax.set_xlabel("fator em s^2 (ruido de medida assumido)")
    ax.set_ylabel("fator em q (ruido de processo assumido)")
    ax.set_title("RMSE (m) do KF no drone vs. Q e R assumidos")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGS / "03_consistencia_sensibilidade.png", dpi=220)
    plt.close(fig)

    _saida(f"\nTodos os assert passaram. (tempo total ~ {time.perf_counter() - t_ini:.0f} s)")
    (FIGS / "03_saida.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
