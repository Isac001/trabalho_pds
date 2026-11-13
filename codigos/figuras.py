import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import signal

AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(AQUI, "figuras_relatorio")
os.makedirs(SAIDA, exist_ok=True)

# --- fonte (usa Carlito se existir; senão a padrão do matplotlib) --------------
for f in ["/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
          "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = "Carlito"

# Especificação e Coeficientes
OMEGA_REJ, OMEGA_PASS, RIPPLE_DB, ATEN_DB = 0.4, 0.6, 0.1, 60.0
W = np.linspace(0, np.pi, 8192)
B_ART = np.array([0.00233783, 0.00496587, 0.00273583, 0.000523340, 0.00488782,
                  0.00812861, -0.00131833, -0.00696899, 0.00839354, 0.0152882,
                  -0.0133790, -0.0130128, 0.0611511, -0.0522489, 0.0323092])
A_ART = np.array([1.0, 3.20033, 6.57387, 10.0924, 12.4517, 12.7867, 11.1194,
                  8.24336, 5.20661, 2.78062, 1.23592, 0.444013, 0.122931,
                  0.0238848, 0.00233213])
TAU_NOMINAL = 14.0      # suposição declarada: atraso de um FIR de 29 coeficientes

N_ART, N_FIR, N_BUT, N_C2, N_ELI = ("IIR do artigo", "FIR (Parks-McClellan)",
                                    "Butterworth", "Chebyshev II", "Elíptico")
ESTILO = {
    N_ART: dict(color="#2a78d6", ls="-",  lw=2.2, zorder=5),
    N_FIR: dict(color="#eb6834", ls="-",  lw=1.5, zorder=4),
    N_BUT: dict(color="#1baf7a", ls="--", lw=1.5, zorder=3),
    N_C2:  dict(color="#eda100", ls="-.", lw=1.5, zorder=3),
    N_ELI: dict(color="#e87ba4", ls=":",  lw=2.0, zorder=3),
}
CINZA = "#52514e"
EIXO_X = "Frequência normalizada (×π rad/amostra)"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#9a9990", "axes.labelcolor": CINZA, "xtick.color": CINZA,
    "ytick.color": CINZA, "text.color": "#0b0b0b", "axes.grid": True,
    "grid.color": "#e6e5de", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.frameon": False, "legend.fontsize": 7.5,
})


def fir_minimo():
    d_p, d_r = 10 ** (RIPPLE_DB / 20) - 1, 10 ** (-ATEN_DB / 20)
    for n in range(11, 401, 2):
        h = signal.remez(n, [0, OMEGA_REJ, OMEGA_PASS, 1.0], [0, 1],
                         weight=[1 / d_r, 1 / d_p], fs=2)
        w, H = signal.freqz(h, worN=W)
        m = 20 * np.log10(np.maximum(np.abs(H), 1e-12)); wn = w / np.pi
        if np.max(np.abs(m[wn >= OMEGA_PASS])) <= RIPPLE_DB + 0.005 and \
           -np.max(m[wn <= OMEGA_REJ]) >= ATEN_DB - 0.005:
            return h
    raise RuntimeError("FIR não encontrado")


def classicos():
    n, wn = signal.buttord(OMEGA_PASS, OMEGA_REJ, RIPPLE_DB, ATEN_DB)
    but = signal.butter(n, wn, btype="highpass")
    n, wn = signal.cheb2ord(OMEGA_PASS, OMEGA_REJ, RIPPLE_DB, ATEN_DB)
    c2 = signal.cheby2(n, ATEN_DB, wn, btype="highpass")
    n, wn = signal.ellipord(OMEGA_PASS, OMEGA_REJ, RIPPLE_DB, ATEN_DB)
    eli = signal.ellip(n, RIPPLE_DB, ATEN_DB, wn, btype="highpass")
    return but, c2, eli


FILTROS = {N_ART: (B_ART, A_ART), N_FIR: (fir_minimo(), np.array([1.0]))}
FILTROS[N_BUT], FILTROS[N_C2], FILTROS[N_ELI] = classicos()

# ============================ FIGURA 1: magnitude =============================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1.25, 1]})
for nome, (b, a) in FILTROS.items():
    w, h = signal.freqz(b, a, worN=W)
    m = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    a1.plot(w / np.pi, m, label=nome, **ESTILO[nome])
    a2.plot(w / np.pi, m, **ESTILO[nome])
a1.axvspan(OMEGA_REJ, OMEGA_PASS, color="#e6e5de", alpha=0.8, lw=0)
a1.text(0.5, -92, "transição", ha="center", color=CINZA, fontsize=7.5)
a1.hlines(-ATEN_DB, 0, OMEGA_REJ, colors=CINZA, linestyles="--", lw=0.9)
a1.text(0.02, -56, "limite: −60 dB", color=CINZA, fontsize=7.5)
a1.set(xlim=(0, 1), ylim=(-100, 5), xlabel=EIXO_X, ylabel="Ganho (dB)")
a1.set_title("(a) Visão geral")
a1.legend(loc="lower right", ncol=1)
a2.hlines([RIPPLE_DB, -RIPPLE_DB], OMEGA_PASS, 1.0, colors=CINZA, linestyles="--", lw=0.9)
a2.text(0.62, RIPPLE_DB + 0.008, "+0,1 dB", color=CINZA, fontsize=7.5)
a2.text(0.62, -RIPPLE_DB - 0.02, "−0,1 dB", color=CINZA, fontsize=7.5)
a2.set(xlim=(OMEGA_PASS, 1.0), ylim=(-0.16, 0.16), xlabel=EIXO_X, ylabel="Ganho (dB)")
a2.set_title("(b) Zoom na faixa de passagem")
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(SAIDA, "fig_relatorio_1_magnitude.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================ FIGURA 2: atraso de grupo =======================
w_pass = np.linspace(OMEGA_PASS * np.pi, np.pi, 4000)
w_zoom = np.linspace(0.65 * np.pi, 0.95 * np.pi, 3000)

fig = plt.figure(figsize=(7.0, 2.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.42)
ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

# (a) atraso de grupo de todos os filtros na faixa de passagem
for nome, (b, a) in FILTROS.items():
    _, gd = signal.group_delay((b, a), w=w_pass)
    ax_a.plot(w_pass / np.pi, gd, label=nome, **ESTILO[nome])
ax_a.set(xlim=(OMEGA_PASS, 1.0), ylim=(0, 22), xlabel=EIXO_X, ylabel="Atraso de grupo (amostras)")
ax_a.set_title("(a) Replicação: todos os filtros")
ax_a.legend(loc="upper right", fontsize=6.8, handlelength=2.2)

# (b) replicação do desvio de atraso de grupo do IIR do artigo (0,65π–0,95π) com
#     envelope de sensibilidade ao arredondamento a 6 algarismos significativos
rng = np.random.default_rng(42)
passo = lambda c: 10.0 ** (np.floor(np.log10(np.abs(c))) - 5)
curvas = []
for _ in range(500):
    b_ = B_ART + rng.uniform(-.5, .5, B_ART.size) * passo(B_ART)
    a_ = A_ART + rng.uniform(-.5, .5, A_ART.size) * passo(A_ART); a_[0] = 1.0
    curvas.append(signal.group_delay((b_, a_), w=w_zoom)[1] - TAU_NOMINAL)
curvas = np.array(curvas)
_, gd_nom = signal.group_delay((B_ART, A_ART), w=w_zoom)
x = w_zoom / np.pi
ax_b.fill_between(x, curvas.min(0), curvas.max(0), color="#2a78d6", alpha=0.18, lw=0,
                  label="500 arredondamentos")
ax_b.plot(x, gd_nom - TAU_NOMINAL, label="coef. publicados", **ESTILO[N_ART])
ax_b.axhline(0, color=ESTILO[N_FIR]["color"], lw=1.2, label="FIR (fase linear)")
ax_b.set(xlim=(0.65, 0.95), ylim=(-0.015, 0.026), xlabel=EIXO_X,
         ylabel="Atraso de grupo − 14 (amostras)")
ax_b.set_title("(b) Replicação: Fig. 9(b)")
ax_b.legend(loc="upper right", fontsize=6.3, handlelength=1.8, borderaxespad=0.1)

png = os.path.join(AQUI, "fig9b_artigo.png")
if os.path.exists(png):
    ax_c.grid(False)
    ax_c.imshow(plt.imread(png)); ax_c.axis("off")
    ax_c.set_title("(c) Artigo: Fig. 9(b)")
else:
    dev_mc = (curvas.max(1) - curvas.min(1)) / 2          # (máx - mín)/2 de cada sorteio
    dev_nom = (gd_nom.max() - gd_nom.min()) / 2           # coeficientes publicados
    n, _, _ = ax_c.hist(dev_mc, bins=18, color="#2a78d6", alpha=0.25, lw=0,
                        label="500 arredondamentos")
    ax_c.axvline(dev_nom, ymax=0.66, color=ESTILO[N_ART]["color"], lw=2.0,
                 label=f"replicação: {dev_nom:.4f}".replace(".", ","))
    ax_c.axvline(0.0042, ymax=0.66, color="#6250d6", lw=2.0, ls="--", label="artigo (Tab. 7): 0,0042")
    ax_c.set_ylim(0, n.max() * 2.3)
    ax_c.set(xlabel="Desvio de atraso (amostras)", ylabel="Nº de sorteios")
    ax_c.set_title("(c) Desvio: artigo × replicação")
    ax_c.legend(loc="upper right", fontsize=6.3, handlelength=1.8, borderaxespad=0.1)
fig.savefig(os.path.join(SAIDA, "fig_relatorio_2_atraso_grupo.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# resumo numérico auxiliar (para conferir com o texto do relatório)
for nome, (b, a) in FILTROS.items():
    _, gd = signal.group_delay((b, a), w=w_pass)
    print(f"{nome:<24} GD min={gd.min():7.3f}  max={gd.max():7.3f}")
print("figuras salvas em", SAIDA)