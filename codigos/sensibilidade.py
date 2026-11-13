import json
import numpy as np
from scipy import signal

# Especificação do script principal
OMEGA_REJ, OMEGA_PASS = 0.4, 0.6
B = np.array([0.00233783, 0.00496587, 0.00273583, 0.000523340, 0.00488782,
              0.00812861, -0.00131833, -0.00696899, 0.00839354, 0.0152882,
              -0.0133790, -0.0130128, 0.0611511, -0.0522489, 0.0323092])
A = np.array([1.0, 3.20033, 6.57387, 10.0924, 12.4517, 12.7867, 11.1194,
              8.24336, 5.20661, 2.78062, 1.23592, 0.444013, 0.122931,
              0.0238848, 0.00233213])
W = np.linspace(0, np.pi, 8192)


def metricas(b, a):
    """Ripple, atenuação, raio máximo dos polos e atraso de grupo em várias faixas."""
    w, h = signal.freqz(b, a, worN=W)
    wn = w / np.pi
    mag = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    out = {
        "ripple_db": float(np.max(np.abs(mag[wn >= OMEGA_PASS]))),
        "aten_db": float(-np.max(mag[wn <= OMEGA_REJ])),
        "raio_polo": float(np.max(np.abs(np.roots(a)))),
    }
    for (lo, hi) in [(0.60, 1.00), (0.62, 0.98), (0.65, 0.95), (0.70, 0.90)]:
        wz = np.linspace(lo * np.pi, hi * np.pi, 4000)
        _, gd = signal.group_delay((b, a), w=wz)
        out[f"gd_dev_{lo:.2f}_{hi:.2f}"] = float((gd.max() - gd.min()) / 2)
        out[f"gd_med_{lo:.2f}_{hi:.2f}"] = float((gd.max() + gd.min()) / 2)
        # desvio em relação a um atraso inteiro nominal de 14 amostras
        out[f"gd_dev14_{lo:.2f}_{hi:.2f}"] = float(np.max(np.abs(gd - 14.0)))
    return out


# ------------------------------------------------------------------------------
# A) Sensibilidade à faixa de medição (coeficientes publicados, sem perturbar)
# ------------------------------------------------------------------------------
nominal = metricas(B, A)

# ------------------------------------------------------------------------------
# B) Monte Carlo: incerteza de arredondamento a 6 algarismos significativos
# ------------------------------------------------------------------------------
rng = np.random.default_rng(42)          # semente fixa -> resultado reprodutível
N_MC = 500


def passo_6_sig(c):
    """Tamanho da unidade do 6º algarismo significativo de c (passo do arredondamento)."""
    return 10.0 ** (np.floor(np.log10(np.abs(c))) - 5)


pb, pa = passo_6_sig(B), passo_6_sig(A)
mc = []
for _ in range(N_MC):
    b_ = B + rng.uniform(-0.5, 0.5, B.size) * pb
    a_ = A + rng.uniform(-0.5, 0.5, A.size) * pa
    a_[0] = 1.0                           # a0 = 1 é exato por definição
    mc.append(metricas(b_, a_))

chaves = ["ripple_db", "aten_db", "raio_polo",
          "gd_dev_0.60_1.00", "gd_dev_0.65_0.95", "gd_dev14_0.65_0.95"]
resumo_mc = {k: {"min": float(np.min([m[k] for m in mc])),
                 "media": float(np.mean([m[k] for m in mc])),
                 "max": float(np.max([m[k] for m in mc]))} for k in chaves}

# ------------------------------------------------------------------------------
# C) Conferências aritméticas e FIR do "filtro 9"
# ------------------------------------------------------------------------------
area = [2219, 2535, 1432]               # Tabela 3: FIR [6], MFIR [34], IIR do artigo (LUTs)
pot = [0.185, 0.201, 0.112]             # Tabela 3 (mW)
red_area = np.mean([1 - area[2] / area[0], 1 - area[2] / area[1]]) * 100
red_pot = np.mean([1 - pot[2] / pot[0], 1 - pot[2] / pot[1]]) * 100

fas_17 = np.array([1028, 1702, 1547, 1918, 1953, 3082, 2023, 1350])
fas_18 = np.array([1762, 1926, 1796, 2541, 2328, 3275, 2350, 1542])
fas_p = np.array([749, 1287, 1171, 1290, 1654, 2626, 1897, 1370])
red17_por_filtro = (1 - fas_p / fas_17) * 100
red18_por_filtro = (1 - fas_p / fas_18) * 100


def menor_fir_passa_baixa(w_pass, w_rej, ripple_db, aten_db):
    d_pass = 10 ** (ripple_db / 20) - 1
    d_rej = 10 ** (-aten_db / 20)
    n_est, _ = signal.kaiserord(aten_db, w_rej - w_pass)
    for numtaps in range(max(5, int(0.5 * n_est)), 600):
        h = signal.remez(numtaps, [0, w_pass, w_rej, 1.0], [1, 0],
                         weight=[1 / d_pass, 1 / d_rej], fs=2)
        w, H = signal.freqz(h, worN=4096)
        wn, mag = w / np.pi, np.abs(H)
        if (np.max(np.abs(20 * np.log10(mag[wn <= w_pass]))) <= ripple_db and
                np.max(mag[wn >= w_rej]) <= d_rej):
            return numtaps
    return None


taps9 = menor_fir_passa_baixa(0.042, 0.14, 0.2, 60)   # transição do filtro 9 (0,098π)

res = {
    "nominal": nominal,
    "monte_carlo": {"n": N_MC, "resumo": resumo_mc},
    "hardware": {"red_area_media_pct": float(red_area), "red_pot_media_pct": float(red_pot),
                 "red17_por_filtro_pct": red17_por_filtro.round(1).tolist(),
                 "red18_por_filtro_pct": red18_por_filtro.round(1).tolist(),
                 "red17_media_pct": float(red17_por_filtro.mean()),
                 "red18_media_pct": float(red18_por_filtro.mean())},
    "fir_filtro9_taps": taps9,
}
with open("sensibilidade.json", "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

print("A) Faixa de medição do desvio de GD (coeficientes publicados):")
for lo, hi in [(0.60, 1.00), (0.62, 0.98), (0.65, 0.95), (0.70, 0.90)]:
    k = f"{lo:.2f}_{hi:.2f}"
    print(f"   {lo:.2f}π–{hi:.2f}π: (max-min)/2 = {nominal['gd_dev_'+k]:.4f}"
          f" | max|GD-14| = {nominal['gd_dev14_'+k]:.4f} | ponto médio = {nominal['gd_med_'+k]:.4f}")
print(f"\nB) Monte Carlo ({N_MC} sorteios de arredondamento a 6 alg. significativos):")
for k, v in resumo_mc.items():
    print(f"   {k:<20} min={v['min']:.4f}  média={v['media']:.4f}  max={v['max']:.4f}")
print(f"\nC) Hardware (Tabela 3): redução média de área = {red_area:.1f}% | de potência = {red_pot:.1f}%")
print(f"   Tabela 5: redução média vs [17] = {red17_por_filtro.mean():.1f}% | vs [18] = {red18_por_filtro.mean():.1f}%")
print("   por filtro vs [17]:", red17_por_filtro.round(1).tolist())
print("   por filtro vs [18]:", red18_por_filtro.round(1).tolist())
print(f"   FIR equiripple do filtro 9 (transição 0,098π): {taps9} coeficientes (artigo: 51)")