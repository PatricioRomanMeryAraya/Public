# -*- coding: utf-8 -*-
"""
META: clasificación alcista/bajista/lateral (2015-06-01 a 2025-07-31)
- Hurst (DFA) en ventana móvil de 252 días
- Drift (pendiente OLS del log-precio) y Z = (a_t / sigma_r)*sqrt(252)
- Reglas:
    H>=0.5 y Z>=+0.5  -> Alcista
    H>=0.5 y Z<=-0.5  -> Bajista
    H<=0.5 o |Z|<0.5  -> Lateral

Requisitos:
    pip install yfinance pandas numpy matplotlib
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ========================= Descarga de datos =============================== #
def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    """
    Descarga precios ajustados (equivalentes a Adj Close) con yfinance.
    Maneja columnas simples o MultiIndex. Si falla, prueba con 'FB'.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("Falta yfinance. Instala con: pip install yfinance")

    def _try_download(tkr: str) -> pd.DataFrame:
        # auto_adjust=True -> 'Close' ya viene ajustado (como Adj Close)
        return yf.download(
            tkr, start=start, end=end,
            progress=False, auto_adjust=True, group_by="column"
        )

    df = _try_download(ticker)
    if df is None or df.empty:
        alt = "FB"  # histórico
        df = _try_download(alt)
        if df is None or df.empty:
            raise ValueError(f"Descarga vacía para {ticker} y fallback {alt}.")

    # Extraer Serie de cierre ajustado
    if isinstance(df.columns, pd.MultiIndex):
        cols = df.columns
        px = None
        if ("Adj Close", ticker) in cols:
            px = df[("Adj Close", ticker)].copy()
        elif ("Close", ticker) in cols:
            px = df[("Close", ticker)].copy()
        elif ("Adj Close", "FB") in cols:
            px = df[("Adj Close", "FB")].copy()
        elif ("Close", "FB") in cols:
            px = df[("Close", "FB")].copy()
        else:
            raise KeyError("No encuentro 'Close' ni 'Adj Close' en MultiIndex.")
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
    else:
        if "Adj Close" in df.columns:
            px = df["Adj Close"].copy()
        elif "Close" in df.columns:
            px = df["Close"].copy()
        else:
            raise KeyError("No encuentro columnas 'Close' ni 'Adj Close'.")

    px = px.astype(float).dropna()
    px.name = "AdjClose"
    if px.empty:
        raise ValueError("Serie de precios vacía después de limpiar NaN.")
    return px

# ========================= Hurst por DFA(1) ================================ #
def hurst_dfa(x: np.ndarray, min_s: int = 10, max_s: int | None = None, num_scales: int = 12) -> float:
    """
    Estimador DFA(1). x: array de log-precios (o retornos).
    Devuelve H (pendiente del ajuste log-log de F(s) ~ s^H).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4 * min_s:
        return np.nan

    if max_s is None:
        max_s = max(20, n // 4)

    x = x - x.mean()
    y = np.cumsum(x)  # perfil integrado

    scales = np.unique(
        np.floor(np.logspace(np.log10(min_s), np.log10(max_s), num=num_scales)).astype(int)
    )

    Fs, Ss = [], []
    for s in scales:
        if s < 5:
            continue
        m = n // s
        if m < 2:
            continue
        rms = []
        for k in range(m):
            seg = y[k * s : (k + 1) * s]
            t = np.arange(s, dtype=float)
            A = np.vstack([t, np.ones(s)]).T
            a, b = np.linalg.lstsq(A, seg, rcond=None)[0]
            resid = seg - (a * t + b)
            rms.append(np.sqrt(np.mean(resid**2)))
        if rms:
            Fs.append(float(np.mean(rms)))
            Ss.append(int(s))

    if len(Fs) < 2:
        return np.nan

    H = np.polyfit(np.log(Ss), np.log(Fs), 1)[0]
    return float(H)

# ==================== Rolling: H, a_t, sigma_r, Z, etiqueta ================ #
def classify_trend(prices: pd.Series,
                   window: int = 252,
                   z_thresh: float = 0.5,
                   hurst_thresh: float = 0.5) -> pd.DataFrame:
    """
    prices: Serie de precios ajustados (índice datetime).
    Retorna DataFrame con:
        AdjClose, logP, H, a_day, sigma_r, Z, tendencia
    """
    px = prices.dropna().astype(float).copy()
    lp = np.log(px.values)
    idx = px.index

    H_arr   = np.full_like(lp, np.nan, dtype=float)
    a_day   = np.full_like(lp, np.nan, dtype=float)
    sigma_r = np.full_like(lp, np.nan, dtype=float)
    Z_arr   = np.full_like(lp, np.nan, dtype=float)
    label   = np.full(lp.shape, "", dtype=object)

    for i in range(window - 1, len(lp)):
        seg = lp[i - window + 1 : i + 1]

        # Hurst (DFA)
        H = hurst_dfa(seg, min_s=10, max_s=max(20, window // 4), num_scales=10)

        # Pendiente OLS diaria del log-precio en la ventana
        t = np.arange(window, dtype=float)
        A = np.vstack([t, np.ones(window)]).T
        a, b = np.linalg.lstsq(A, seg, rcond=None)[0]   # a: por día

        # Volatilidad de retornos log diarios
        r = np.diff(seg)
        sig = r.std(ddof=1)

        Z = (a / sig) * math.sqrt(252.0) if (np.isfinite(sig) and sig > 0) else np.nan

        H_arr[i]   = H
        a_day[i]   = a
        sigma_r[i] = sig
        Z_arr[i]   = Z

        # Clasificación (tus reglas exactas)
        if np.isfinite(H) and np.isfinite(Z):
            if (H >= hurst_thresh) and (Z >= +z_thresh):
                lab = "alcista"
            elif (H >= hurst_thresh) and (Z <= -z_thresh):
                lab = "bajista"
            else:
                lab = "lateral"
        else:
            lab = "lateral"
        label[i] = lab

    out = pd.DataFrame(
        {
            "AdjClose": px.values,
            "logP": np.log(px.values),
            "H": H_arr,
            "a_day": a_day,
            "sigma_r": sigma_r,
            "Z": Z_arr,
            "tendencia": label,
        },
        index=idx,
    )
    return out

# ============================ Utilidades runs =============================== #
def _iter_runs(labels: np.ndarray, index: pd.Index):
    """
    Itera bloques contiguos con la misma etiqueta.
    Devuelve (label, start_time, end_time).
    """
    current = None
    start_i = None
    for i, lab in enumerate(labels):
        if lab != current:
            if current is not None:
                yield current, index[start_i], index[i - 1]
            current = lab
            start_i = i
    if current is not None:
        yield current, index[start_i], index[len(labels) - 1]

# ============================== Visualización =============================== #
def plot_results(prices: pd.Series, res: pd.DataFrame,
                 png_path: str = "META_trend_plot.png",
                 pdf_path: str = "META_trend_plot.pdf"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Colores de alto contraste por régimen
    COLORS = {
        "alcista": "#1B9E77",   # verde azulado intenso
        "bajista": "#D95F02",   # naranja/rojo intenso
        "lateral": "#636363",   # gris oscuro
    }

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.2, 1.2, 0.4]})

    # ---- Panel 1: Precio + sombreado por régimen (alto contraste) ---- #
    (line_price,) = axes[0].plot(prices.index, prices.values, label="Precio ajustado", linewidth=1.2)
    axes[0].set_title("META - Precio ajustado (con régimen)")
    axes[0].grid(True, alpha=0.25)

    # Fondo sombreado con mayor contraste
    labels = res["tendencia"].values
    for lab, t0, t1 in _iter_runs(labels, res.index):
        if lab in COLORS and lab != "":
            axes[0].axvspan(t0, t1, color=COLORS[lab], alpha=0.28)

    # Leyenda: línea + parches de régimen
    legend_handles = [line_price] + [
        Patch(facecolor=COLORS["alcista"], label="Alcista"),
        Patch(facecolor=COLORS["bajista"], label="Bajista"),
        Patch(facecolor=COLORS["lateral"], label="Lateral"),
    ]
    axes[0].legend(handles=legend_handles, ncol=4, loc="upper left", frameon=True)

    # ---- Panel 2: Hurst ---- #
    axes[1].plot(res.index, res["H"], label="H (DFA)", linewidth=1.0)
    axes[1].axhline(0.5, linestyle="--", linewidth=1.0)
    axes[1].set_title("Coeficiente de Hurst (DFA)")
    axes[1].grid(True, alpha=0.25)

    # ---- Panel 3: Z (fuerza del drift) ---- #
    axes[2].plot(res.index, res["Z"], label="Z", linewidth=1.0)
    axes[2].axhline(+0.5, linestyle="--", linewidth=1.0)
    axes[2].axhline(-0.5, linestyle="--", linewidth=1.0)
    axes[2].set_title("Fuerza del drift (Z)")
    axes[2].grid(True, alpha=0.25)

    # ---- Panel 4: Barra de régimen (timeline categórico) ---- #
    axes[3].set_title("Régimen")
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    axes[3].grid(False)
    for lab, t0, t1 in _iter_runs(labels, res.index):
        if lab in COLORS and lab != "":
            axes[3].axvspan(t0, t1, color=COLORS[lab], alpha=1.0)
    # Etiquetas x más respiradas
    fig.autofmt_xdate()

    plt.tight_layout()

    # Guardar y mostrar
    try:
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"\nGráficos guardados en:\n - {png_path}\n - {pdf_path}")
    except Exception as e:
        print(f"Advertencia: no pude guardar las figuras: {e}")

    try:
        plt.show(block=True)
    except Exception as e:
        print(f"Sin backend gráfico para ventana: {e}. Revisa los PNG/PDF.")

# ================================ Main ====================================== #
if __name__ == "__main__":
    START = "2015-06-01"
    # yfinance usa 'end' semi-exclusivo; para incluir 2025-07-31, usa 2025-08-01
    END   = "2025-08-01"

    px = fetch_prices("META", START, END)
    res = classify_trend(px, window=252, z_thresh=0.5, hurst_thresh=0.5)

    # Resumen final (último día con métricas completas)
    last_complete = res.dropna(subset=["H", "a_day", "sigma_r", "Z"])
    if not last_complete.empty:
        last_row = last_complete.iloc[-1]
        print("\n--- META: clasificación en la última fecha ---")
        print(f"Fecha:       {last_row.name.date()}")
        print(f"H:           {last_row['H']:.3f}")
        print(f"a_day:       {last_row['a_day']:.6f} (log por día)")
        print(f"sigma_r:     {last_row['sigma_r']:.6f} (log-retorno diario)")
        print(f"Z:           {last_row['Z']:.3f}")
        print(f"Tendencia:   {last_row['tendencia']}\n")
    else:
        print("Aún no hay 252 datos completos para calcular métricas.")

    # Conteo de días por etiqueta
    counts = res["tendencia"].value_counts(dropna=False)
    print("--- Conteo de días por régimen ---")
    print(counts)

    # Guardar CSV
    res.to_csv("META_trend_hurst_20150601_20250731.csv", index=True)
    print("\nResultados guardados en: META_trend_hurst_20150601_20250731.csv")

    # Visualización con leyenda clara y alto contraste
    plot_results(px, res)
