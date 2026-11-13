import numpy as np                
import matplotlib.pyplot as plt   
from scipy import signal          

# ==============================================================================
# 1. ESPECIFICAÇÃO DO FILTRO (a mesma para TODOS os filtros comparados)
# ==============================================================================

OMEGA_REJ = 0.4        # a faixa de rejeição (bloqueio) vai de 0 até 0,4 (x π)
OMEGA_PASS = 0.6       # a faixa de passagem vai de 0,6 até 1,0 (x π)
                       # entre 0,4 e 0,6 fica a "faixa de transição"
RIPPLE_MAX_DB = 0.1    # variação máxima (ondulação) permitida na passagem, em dB
ATEN_MIN_DB = 60.0     # atenuação mínima exigida na faixa de rejeição, em dB

# Grade de frequências onde todas as curvas serão calculadas (de 0 a π).
N_PONTOS = 8192
W_GRADE = np.linspace(0, np.pi, N_PONTOS)

# ==============================================================================
# 2. COEFICIENTES DO IIR DO ARTIGO (solução final da Seção III-E)
# ==============================================================================
# Função de transferência do IIR:
#
#            b0 + b1 z^-1 + ... + b14 z^-14        (numerador   -> zeros)
#   H(z) = ---------------------------------
#            1  + a1 z^-1 + ... + a14 z^-14        (denominador -> polos)
#

# Valores com 6 algarismos significativos, copiados do artigo.
B_ARTIGO = np.array([
    0.00233783, 0.00496587, 0.00273583, 0.000523340, 0.00488782,
    0.00812861, -0.00131833, -0.00696899, 0.00839354, 0.0152882,
    -0.0133790, -0.0130128, 0.0611511, -0.0522489, 0.0323092,
])
A_ARTIGO = np.array([
    1.0, 3.20033, 6.57387, 10.0924, 12.4517,
    12.7867, 11.1194, 8.24336, 5.20661, 2.78062,
    1.23592, 0.444013, 0.122931, 0.0238848, 0.00233213,
])

# Nomes das curvas (usados nos gráficos e na tabela do terminal).
N_ARTIGO = "IIR do artigo"
N_FIR = "FIR (Parks-McClellan)"
N_BUTTER = "Butterworth"
N_CHEBY2 = "Chebyshev II"
N_ELIP = "Elíptico"


# ==============================================================================
# 3. FUNÇÃO QUE "MEDE" UM FILTRO (resposta em frequência + métricas)
# ==============================================================================
def mede_filtro(b, a):
    """
    Recebe os coeficientes (b = numerador, a = denominador) e devolve um
    dicionário com a resposta em frequência e as métricas principais.
    Para um filtro FIR, basta usar a = [1.0] (não há realimentação).
    """
    # Resposta em frequência H(e^jw): freqz avalia H(z) sobre o círculo unitário.
    w, h = signal.freqz(b, a, worN=W_GRADE)
    wn = w / np.pi                                    # frequência normalizada (x π)

    # Magnitude em decibéis (o 1e-12 evita log de zero nos "buracos" da resposta).
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

    # Máscaras booleanas: quais pontos da grade pertencem a cada faixa.
    faixa_pass = wn >= OMEGA_PASS
    faixa_rej = wn <= OMEGA_REJ

    # Ondulação na passagem: o quanto o ganho se afasta de 0 dB (ganho = 1).
    ripple_db = np.max(np.abs(mag_db[faixa_pass]))

    # Atenuação mínima na rejeição: o pior caso (maior ganho) nessa faixa.
    aten_db = -np.max(mag_db[faixa_rej])

    # Atraso de grupo = -dφ/dω (derivada da fase). Só nos interessa na passagem.
    w_pass = w[faixa_pass]
    _, gd = signal.group_delay((b, a), w=w_pass)

    # Referência constante: o ponto médio entre o maior e o menor atraso.
    gd_ref = (gd.max() + gd.min()) / 2
    gd_dev = (gd.max() - gd.min()) / 2

    # Estabilidade: todos os polos precisam estar DENTRO do círculo unitário.
    polos = np.roots(a)
    raio_max = np.max(np.abs(polos)) if polos.size > 0 else 0.0

    return {
        "b": b, "a": a,
        "wn": wn, "mag_db": mag_db,
        "w_pass": wn[faixa_pass], "gd": gd, "gd_ref": gd_ref, "gd_dev": gd_dev,
        "ripple_db": ripple_db, "aten_db": aten_db,
        "raio_max": raio_max,
        "ordem": max(len(a), len(b)) - 1,
        "n_coef": len(b) + len(a) - 1,
    }


def atende_especificacao(f):
    
    # Diz se o filtro medido cumpre a ondulação e a atenuação exigidas.
    # A pequena folga (0,005 dB) evita reprovar por erro de arredondamento:
    
    folga = 0.005
    return (f["ripple_db"] <= RIPPLE_MAX_DB + folga and
            f["aten_db"] >= ATEN_MIN_DB - folga)


# ==============================================================================
# 4. PROJETO DOS FILTROS DE COMPARAÇÃO
# ==============================================================================
filtros = {}   # dicionário: nome do filtro -> resultados medidos

# ------------------------------------------------------------------------------
# 4.1  IIR do artigo (coeficientes publicados)
# ------------------------------------------------------------------------------
filtros[N_ARTIGO] = mede_filtro(B_ARTIGO, A_ARTIGO)

# ------------------------------------------------------------------------------
# 4.2  FIR de fase linear (Parks-McClellan / equiripple)
# ------------------------------------------------------------------------------


# Tolerâncias em escala linear (não em dB):
DELTA_PASS = 10 ** (RIPPLE_MAX_DB / 20) - 1    # ondulação máx. na passagem
DELTA_REJ = 10 ** (-ATEN_MIN_DB / 20)          # ganho máx. na rejeição (60 dB -> 0,001)

fir_escolhido = None
for numtaps in range(11, 401, 2):

    # remez = algoritmo de Parks-McClellan.
    #  - bandas:   [0, 0.4]  rejeição  |  [0.6, 1.0] passagem   (fs=2 -> Nyquist = 1)
    #  - desejado: 0 na rejeição, 1 na passagem
    #  - pesos:    inversamente proporcionais à tolerância de cada faixa
    h_fir = signal.remez(
        numtaps,
        [0, OMEGA_REJ, OMEGA_PASS, 1.0],
        [0, 1],
        weight=[1 / DELTA_REJ, 1 / DELTA_PASS],
        fs=2,
    )
    medido = mede_filtro(h_fir, np.array([1.0]))
    if atende_especificacao(medido):      # achamos o menor FIR que cumpre!
        fir_escolhido = medido
        break

filtros[N_FIR] = fir_escolhido

# ------------------------------------------------------------------------------
# 4.3  IIR clássicos projetados para a MESMA especificação
# ------------------------------------------------------------------------------


# Butterworth: resposta suave (sem ondulação), mas exige ordem alta.
n_but, wn_but = signal.buttord(OMEGA_PASS, OMEGA_REJ, RIPPLE_MAX_DB, ATEN_MIN_DB)
b_but, a_but = signal.butter(n_but, wn_but, btype="highpass")
filtros[N_BUTTER] = mede_filtro(b_but, a_but)

# Chebyshev tipo II: sem ondulação na passagem, ondulação na rejeição.
n_c2, wn_c2 = signal.cheb2ord(OMEGA_PASS, OMEGA_REJ, RIPPLE_MAX_DB, ATEN_MIN_DB)
b_c2, a_c2 = signal.cheby2(n_c2, ATEN_MIN_DB, wn_c2, btype="highpass")
filtros[N_CHEBY2] = mede_filtro(b_c2, a_c2)

# Elíptico (Cauer): ordem mínima possível, ondulação nas duas faixas.
# É o "campeão" de economia, mas o atraso de grupo é o mais irregular.
n_el, wn_el = signal.ellipord(OMEGA_PASS, OMEGA_REJ, RIPPLE_MAX_DB, ATEN_MIN_DB)
b_el, a_el = signal.ellip(n_el, RIPPLE_MAX_DB, ATEN_MIN_DB, wn_el, btype="highpass")
filtros[N_ELIP] = mede_filtro(b_el, a_el)

# ==============================================================================
# 5. TABELA-RESUMO NO TERMINAL
# ==============================================================================
print("=" * 100)
print("ESPECIFICAÇÃO (passa-alta): rejeição até %.1fπ | passagem a partir de %.1fπ | "
      "ondulação <= %.2f dB | atenuação >= %.0f dB"
      % (OMEGA_REJ, OMEGA_PASS, RIPPLE_MAX_DB, ATEN_MIN_DB))
print("=" * 100)
print(f"{'Filtro':<24}{'Ordem':>6}{'Nº coef.':>9}{'Ripple(dB)':>12}{'Aten.(dB)':>11}"
      f"{'Atraso médio':>14}{'Desvio GD':>11}{'Raio máx. polo':>16}{'Cumpre?':>9}")
print("-" * 100)
for nome, f in filtros.items():
    print(f"{nome:<24}{f['ordem']:>6}{f['n_coef']:>9}{f['ripple_db']:>12.3f}"
          f"{f['aten_db']:>11.1f}{f['gd_ref']:>14.3f}{f['gd_dev']:>11.4f}"
          f"{f['raio_max']:>16.3f}{('sim' if atende_especificacao(f) else 'NÃO'):>9}")
print("-" * 100)
print("Desvio GD = desvio máximo do atraso de grupo (em amostras) na faixa de passagem.")
print("Quanto menor, mais linear é a fase (FIR de fase linear tem desvio = 0).")

f_art = filtros[N_ARTIGO]
print("\nConferência com o artigo (filtro IIR3, Seção III-E):")
print("  ondulação  : artigo 0.063 dB | aqui %.3f dB" % f_art["ripple_db"])
print("  atenuação  : artigo 61.9 dB  | aqui %.1f dB" % f_art["aten_db"])
print("  (pequenas diferenças são esperadas: os coeficientes publicados estão "
      "arredondados em 6 algarismos)")

w_zoom = np.linspace(0.65 * np.pi, 0.95 * np.pi, 4000)
_, gd_zoom = signal.group_delay((B_ARTIGO, A_ARTIGO), w=w_zoom)
print("  desvio GD  : artigo 0.0042   | aqui %.4f (faixa 0,6π-1,0π) e %.4f (faixa 0,65π-0,95π)"
      % (f_art["gd_dev"], (gd_zoom.max() - gd_zoom.min()) / 2))

# ==============================================================================
# 6. ESTILO DOS GRÁFICOS
# ==============================================================================
def estilo_grafico():
    """Define em um só lugar o visual dos gráficos (fundo, grade, fontes)."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "text.color": "#0b0b0b",
        "axes.labelcolor": "#52514e",
        "axes.edgecolor": "#c3c2b7",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,       
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "font.size": 10,
    })


estilo_grafico()

ESTILO_SERIE = {
    N_ARTIGO: dict(color="#2a78d6", ls="-",  lw=3.0, zorder=5),
    N_FIR:    dict(color="#eb6834", ls="-",  lw=2.0, zorder=4),
    N_BUTTER: dict(color="#1baf7a", ls="--", lw=2.0, zorder=3),
    N_CHEBY2: dict(color="#eda100", ls="-.", lw=2.0, zorder=3),
    N_ELIP:   dict(color="#e87ba4", ls=":",  lw=2.6, zorder=3),
}
COR_TEXTO_SEC = "#52514e"
EIXO_X = r"Frequência normalizada ($\times\pi$ rad/amostra)"


def virgula(valor, casas=1):
    """Formata um número com vírgula decimal (padrão brasileiro) para os títulos."""
    return f"{valor:.{casas}f}".replace(".", ",")


def salva(fig, nome_arquivo):
    """Salva a figura em PNG (boa resolução para colar nos slides)."""
    fig.savefig(nome_arquivo, dpi=200, bbox_inches="tight")
    print("Figura salva:", nome_arquivo)


# ==============================================================================
# 7. FIGURA 1 - RESPOSTA EM MAGNITUDE (o que o filtro deixa passar ou barra)
# ==============================================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for nome, f in filtros.items():
    ax1.plot(f["wn"], f["mag_db"], label=nome, **ESTILO_SERIE[nome])
    ax2.plot(f["wn"], f["mag_db"], **ESTILO_SERIE[nome])

# --- Painel (a): visão geral -------------------------------------------------
ax1.axvspan(OMEGA_REJ, OMEGA_PASS, color="#e1e0d9", alpha=0.7, lw=0)   # transição
ax1.text((OMEGA_REJ + OMEGA_PASS) / 2, 1.5, "transição", ha="center", va="center",
         color=COR_TEXTO_SEC, fontsize=9)
ax1.hlines(-ATEN_MIN_DB, 0, OMEGA_REJ, colors=COR_TEXTO_SEC, linestyles="--", lw=1)
ax1.text(0.02, -57, "limite da especificação: -60 dB", color=COR_TEXTO_SEC, fontsize=9)
ax1.set_xlim(0, 1)
ax1.set_ylim(-100, 5)
ax1.set_xlabel(EIXO_X)
ax1.set_ylabel("Ganho (dB)")
ax1.set_title("(a) Resposta em magnitude: todos cumprem a mesma especificação")
ax1.legend(loc="lower right")

# --- Painel (b): zoom na faixa de passagem ------------------------------------
ax2.hlines([RIPPLE_MAX_DB, -RIPPLE_MAX_DB], OMEGA_PASS, 1.0, colors=COR_TEXTO_SEC,
           linestyles="--", lw=1)
ax2.text(0.62, RIPPLE_MAX_DB + 0.006, "limite: +0,1 dB", color=COR_TEXTO_SEC, fontsize=9)
ax2.text(0.62, -RIPPLE_MAX_DB - 0.017, "limite: -0,1 dB", color=COR_TEXTO_SEC, fontsize=9)
ax2.set_xlim(OMEGA_PASS, 1.0)
ax2.set_ylim(-0.16, 0.16)
ax2.set_xlabel(EIXO_X)
ax2.set_ylabel("Ganho (dB)")
ax2.set_title("(b) Zoom na faixa de passagem (ondulação)")

fig1.suptitle("Figura 1 - Mesma especificação passa-alta, filtros diferentes",
              fontsize=13, fontweight="bold")
fig1.tight_layout()
salva(fig1, "fig1_magnitude.png")

# ==============================================================================
# 8. FIGURA 2 - ATRASO DE GRUPO (aqui está a tese do artigo)
# ==============================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for nome, f in filtros.items():
    ax1.plot(f["w_pass"], f["gd"], label=nome, **ESTILO_SERIE[nome])

topo = min(40, 1.15 * max(f["gd"].max() for f in filtros.values()))
ax1.set_ylim(0, topo)
ax1.set_xlim(OMEGA_PASS, 1.0)
ax1.text(0.61, 14.5, "FIR e IIR do artigo coincidem (≈14)", color=COR_TEXTO_SEC, fontsize=9)
ax1.set_xlabel(EIXO_X)
ax1.set_ylabel("Atraso de grupo (amostras)")
ax1.set_title("(a) Atraso de grupo na faixa de passagem")
ax1.legend(loc="upper right")

f_art = filtros[N_ARTIGO]
f_fir = filtros[N_FIR]
ax2.plot(f_fir["w_pass"], f_fir["gd"] - f_fir["gd_ref"], label=N_FIR,
         **ESTILO_SERIE[N_FIR])
ax2.plot(f_art["w_pass"], f_art["gd"] - f_art["gd_ref"], label=N_ARTIGO,
         **ESTILO_SERIE[N_ARTIGO])
ax2.set_xlim(OMEGA_PASS, 1.0)
ax2.set_ylim(-0.02, 0.02)
ax2.set_xlabel(EIXO_X)
ax2.set_ylabel("Desvio do atraso de grupo (amostras)")
ax2.set_title("(b) Zoom: o IIR do artigo varia só ±%s amostra" % virgula(f_art["gd_dev"], 3))
ax2.legend(loc="upper right")

fig2.suptitle("Figura 2 - Linearidade da fase: o FIR é perfeito, o IIR do artigo "
              "chega perto, os IIR clássicos não",
              fontsize=13, fontweight="bold")
fig2.tight_layout()
salva(fig2, "fig2_atraso_de_grupo.png")

# ==============================================================================
# 9. FIGURA 3 - POLOS E ZEROS (estabilidade do IIR)
# ==============================================================================
fig3, eixos = plt.subplots(1, 2, figsize=(11, 5.5))
theta = np.linspace(0, 2 * np.pi, 400)

for ax, nome in zip(eixos, [N_ARTIGO, N_ELIP]):
    f = filtros[nome]
    zeros = np.roots(f["b"])
    polos = np.roots(f["a"])
    cor = ESTILO_SERIE[nome]["color"]

    ax.plot(np.cos(theta), np.sin(theta), color="#c3c2b7", lw=1.5)   
    ax.scatter(zeros.real, zeros.imag, s=60, facecolors="none",
               edgecolors=cor, linewidths=1.8, label="zeros", zorder=4)
    ax.scatter(polos.real, polos.imag, s=70, marker="x", color=cor,
               linewidths=2.2, label="polos", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Parte real")
    ax.set_ylabel("Parte imaginária")
    ax.set_title("%s\nraio máximo dos polos = %s" % (nome, virgula(f["raio_max"], 3)))
    ax.legend(loc="upper right")

fig3.suptitle("Figura 3 - Plano Z: polos dentro do círculo unitário = filtro estável",
              fontsize=13, fontweight="bold")
fig3.tight_layout()
salva(fig3, "fig3_polos_zeros.png")

# ==============================================================================
# 10. FIGURA 4 - POR QUE O FIR FICA CARO: ORDEM x LARGURA DA FAIXA DE TRANSIÇÃO
# ==============================================================================

def menor_fir_passa_baixa(w_pass, w_rej, ripple_db, aten_db):
    """Menor número de coeficientes de um FIR passa-baixa (Parks-McClellan)."""
    d_pass = 10 ** (ripple_db / 20) - 1     
    d_rej = 10 ** (-aten_db / 20)
    n_estimado, _ = signal.kaiserord(aten_db, w_rej - w_pass)
    for numtaps in range(max(5, int(0.5 * n_estimado)), 600):
        h = signal.remez(numtaps, [0, w_pass, w_rej, 1.0], [1, 0],
                         weight=[1 / d_pass, 1 / d_rej], fs=2)
        w, H = signal.freqz(h, worN=4096)
        wn, mag = w / np.pi, np.abs(H)
        cumpre_pass = np.max(np.abs(20 * np.log10(mag[wn <= w_pass]))) <= ripple_db
        cumpre_rej = np.max(mag[wn >= w_rej]) <= d_rej
        if cumpre_pass and cumpre_rej:
            return numtaps
    return None


W_PASS_PB = 0.042                              
larguras = np.array([0.03, 0.05, 0.07, 0.098, 0.15, 0.2, 0.3])  
taps_fir = [menor_fir_passa_baixa(W_PASS_PB, W_PASS_PB + L, 0.2, 60) for L in larguras]

fig4, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(larguras, taps_fir, marker="o", label="FIR (Parks-McClellan, calculado aqui)",
        **{k: v for k, v in ESTILO_SERIE[N_FIR].items() if k != "ls"})

L9 = 0.14 - W_PASS_PB
ax.scatter([L9], [51], s=110, marker="s", color="#6250d6", zorder=6,
           label="FIR [6] do artigo: 51 coeficientes")
ax.scatter([L9], [13], s=140, marker="D", color=ESTILO_SERIE[N_ARTIGO]["color"],
           zorder=6, label="IIR do artigo: ordem 13")
ax.set_xlabel(r"Largura da faixa de transição ($\times\pi$ rad/amostra)")
ax.set_ylabel("Nº de coeficientes do FIR")
ax.set_title("Figura 4 - FIR: transição estreita = muitos coeficientes\n"
             "(passa-baixa, ondulação 0,2 dB, atenuação 60 dB)")
ax.legend(loc="upper right")
fig4.tight_layout()
salva(fig4, "fig4_ordem_fir_vs_transicao.png")

# ==============================================================================
# 11. FIGURA 5 - CUSTO DE HARDWARE: IIR DO ARTIGO x FIR (dados da Tabela 3)
# ==============================================================================
rotulos_t3 = ["FIR [6]", "MFIR [34]", "IIR do artigo"]
ordem_t3 = [51, 57, 13]
area_t3 = [2219, 2535, 1432]          # nº de LUTs
potencia_t3 = [0.185, 0.201, 0.112]   # mW
cores_t3 = [ESTILO_SERIE[N_FIR]["color"], "#6250d6", ESTILO_SERIE[N_ARTIGO]["color"]]

fig5, eixos = plt.subplots(1, 3, figsize=(14, 5))
paineis = [
    (ordem_t3, "Ordem do filtro", "Nº de coeficientes (ordem)"),
    (area_t3, "Área na FPGA", "Nº de LUTs"),
    (potencia_t3, "Potência na FPGA", "mW"),
]
for ax, (valores, titulo, ylabel) in zip(eixos, paineis):
    barras = ax.bar(rotulos_t3, valores, color=cores_t3, width=0.6)
    ax.bar_label(barras, fmt="%g", padding=3, fontsize=10)    
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(valores) * 1.15)
    ax.grid(axis="x", visible=False)

red_area = np.mean([1 - area_t3[2] / area_t3[0], 1 - area_t3[2] / area_t3[1]]) * 100
red_pot = np.mean([1 - potencia_t3[2] / potencia_t3[0],
                   1 - potencia_t3[2] / potencia_t3[1]]) * 100
fig5.suptitle("Figura 5 - IIR do artigo x FIR (passa-baixa, Tabela 3): "
              "menos %s%% de área e menos %s%% de potência (média)"
              % (virgula(red_area), virgula(red_pot)), fontsize=13, fontweight="bold")
fig5.tight_layout()
salva(fig5, "fig5_hardware_fir.png")

# ==============================================================================
# 12. FIGURA 6 - SOMADORES (FAs) vs OUTROS IIR QUASE LINEARES (Tabela 5)
# ==============================================================================
bench = ["IIR1", "IIR2", "IIR3", "IIR4", "IIR5", "IIR6", "IIR7", "IIR8"]
fas_17 = np.array([1028, 1702, 1547, 1918, 1953, 3082, 2023, 1350])
fas_18 = np.array([1762, 1926, 1796, 2541, 2328, 3275, 2350, 1542])
fas_prop = np.array([749, 1287, 1171, 1290, 1654, 2626, 1897, 1370])
x = np.arange(len(bench))      
larg = 0.27                    
fig6, ax = plt.subplots(figsize=(13, 5.5))
ax.bar(x - larg, fas_17, larg, label="Lai & Lin [17]", color="#6250d6")
ax.bar(x, fas_18, larg, label="Jiang & Kwan [18]", color="#008300")
ax.bar(x + larg, fas_prop, larg, label="IIR do artigo", color=ESTILO_SERIE[N_ARTIGO]["color"])
ax.set_xticks(x)
ax.set_xticklabels(bench)
ax.set_xlabel("Filtros de referência (benchmark de Nongpiur et al.)")
ax.set_ylabel("Nº de somadores completos (FAs)")
ax.set_title("Figura 6 - Custo de hardware (Tabela 5): o IIR do artigo usa menos "
             "somadores na maioria dos casos")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left")
fig6.tight_layout()
salva(fig6, "fig6_somadores.png")
red17 = np.mean(1 - fas_prop / fas_17) * 100
red18 = np.mean(1 - fas_prop / fas_18) * 100
print("\nRedução média de somadores recalculada com a Tabela 5:")
print("  vs [17]: %.1f%%   | vs [18]: %.1f%%" % (red17, red18))
print("  (o texto do artigo cita 27,4% e 40,9% - confira ao apresentar)")

# ==============================================================================
# 13. MOSTRA TODAS AS FIGURAS NA TELA
# ==============================================================================
plt.show()