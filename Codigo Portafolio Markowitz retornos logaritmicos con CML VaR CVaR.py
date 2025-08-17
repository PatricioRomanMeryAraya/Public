# -*- coding: utf-8 -*-
"""
Autor: Patricio Román Mery Araya. Se autoriza su uso bajo los términos y condiciones de la 
licencia Creative Commons Atribución-No Comercial 4.0 Internacional (CC BY-NC 4.0): 
https://creativecommons.org/licenses/by-nc/4.0/

Disclaimer: El código es de uso exclusivo para fines académicos.
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# ====================== CONFIGURACIÓN ======================
# Universo de activos
TICKERS = ['ORCL', 'QCOM', 'META', 'CRM', 'INTC']  # ajusta si quieres otros

# Fechas para descargar datos
START_DATE = '2015-06-01'
END_DATE   = '2025-07-31'

# Parámetros de trading
TRADING_DAYS = 252

# Parámetros de riesgo y robustez
RISK_FREE_SYMBOL       = '^FVX'   # ^IRX=13w T-Bill, ^FVX=5Y, ^TNX=10Y
REQUIRE_RISK_FREE_DATA = True     # Si True, lanza error si no hay dato de tasa libre
VAR_HORIZON_DAYS       = 1        # Horizonte para VaR/CVaR (en días)
COV_REG_EPS            = 1e-8     # Regularización numérica (añade eps*I a la covarianza anual)

# Ruta de salida
OUTPUT_FILE = 'markowitz_portfolios_with_CML.xlsx'

# ====================== FUNCIONES AUXILIARES ======================

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos logarítmicos diarios."""
    return np.log(prices / prices.shift(1)).dropna()

def obtener_tasa_libre(symbol: str, start_date: str, end_date: str, require: bool = True) -> float:
    """
    Descarga la tasa libre (rendimiento, %) desde 'symbol' y devuelve el último valor en DECIMAL.
    Si require=True y no hay datos, lanza RuntimeError.
    """
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty or 'Close' not in data.columns or data['Close'].dropna().empty:
        if require:
            raise RuntimeError(f"No se pudo obtener la tasa libre desde {symbol}.")
        else:
            print("Advertencia: sin datos de tasa libre; usando 4% (no recomendado).")
            return 0.04
    tasa_pct = float(data['Close'].dropna().iloc[-1])
    return tasa_pct / 100.0  # decimal

def portfolio_variance_annual(w: np.ndarray, cov_matrix_annual: np.ndarray) -> float:
    """Varianza anual del portafolio."""
    return float(w.T @ cov_matrix_annual @ w)

def weight_sum_constraint(w: np.ndarray) -> float:
    """Restricción: suma de pesos = 1."""
    return float(np.sum(w) - 1.0)

# ====================== FUNCIONES DE RIESGO ======================

def _horizon_stats_from_daily(pr: np.ndarray, horizon_days: int):
    """
    A partir de retornos diarios del portafolio (pr), obtiene:
    mu_h, sig_h (paramétricos) y también pr_h (serie hist. de retornos a H días, en logs).
    Para logs: retorno en H días ~ suma de retornos diarios.
    """
    if pr.ndim != 1:
        pr = pr.ravel()

    mu_d  = pr.mean()
    sig_d = pr.std(ddof=1)

    mu_h  = mu_d * horizon_days
    sig_h = sig_d * np.sqrt(horizon_days)

    if horizon_days <= 0:
        raise ValueError("horizon_days debe ser >= 1")
    if horizon_days == 1:
        pr_h = pr.copy()
    else:
        pr_h = pd.Series(pr).rolling(window=horizon_days).sum().dropna().values

    return mu_h, sig_h, pr_h, mu_d, sig_d

def var_cvar_param_normal(returns: np.ndarray, weights: np.ndarray, confidence_levels=None, horizon_days: int = 1) -> pd.DataFrame:
    """VaR y CVaR paramétricos asumiendo Normalidad. Retorna %."""
    if confidence_levels is None:
        confidence_levels = np.concatenate([np.arange(0.01, 0.95, 0.01), [0.95], np.arange(0.96, 1.00, 0.01)])

    pr = returns @ weights
    mu_h, sig_h, _, _, _ = _horizon_stats_from_daily(pr, horizon_days)

    out = []
    for conf in confidence_levels:
        alpha   = 1.0 - conf
        z_alpha = norm.ppf(alpha)  # negativo (p.ej., -1.64485 en 95%)
        var  = -(mu_h + z_alpha * sig_h)
        cvar = -(mu_h - sig_h * (norm.pdf(z_alpha) / alpha)) if alpha > 0 else var
        out.append({"Confidence Level": round(conf, 4), "VaR": var * 100.0, "CVaR": cvar * 100.0})
    return pd.DataFrame(out)

def var_cvar_historico(returns: np.ndarray, weights: np.ndarray, confidence_levels=None, horizon_days: int = 1) -> pd.DataFrame:
    """VaR y CVaR histórico usando la distribución empírica de retornos a H días (logs sumados)."""
    if confidence_levels is None:
        confidence_levels = np.concatenate([np.arange(0.01, 0.95, 0.01), [0.95], np.arange(0.96, 1.00, 0.01)])

    pr = returns @ weights
    _, _, pr_h, _, _ = _horizon_stats_from_daily(pr, horizon_days)

    if pr_h.size < 50:
        raise RuntimeError("Muy pocos datos para calcular VaR/CVaR histórico al horizonte especificado.")

    out = []
    for conf in confidence_levels:
        alpha   = 1.0 - conf
        q_alpha = np.quantile(pr_h, alpha)  # cuantil cola inferior
        var  = -q_alpha
        tail = pr_h[pr_h <= q_alpha]
        cvar = -tail.mean() if tail.size > 0 else var
        out.append({"Confidence Level": round(conf, 4), "VaR": var * 100.0, "CVaR": cvar * 100.0})
    return pd.DataFrame(out)

def var_cvar_cornish_fisher(returns: np.ndarray, weights: np.ndarray, confidence_levels=None, horizon_days: int = 1) -> pd.DataFrame:
    """
    VaR y CVaR con ajuste Cornish–Fisher:
    z_cf = z + (1/6)(z^2-1)s + (1/24)(z^3-3z)k - (1/36)(2z^3-5z)s^2
    s=asimetría, k=exceso de curtosis; para horizonte h: s_h = s/sqrt(h), k_h = k/h.
    """
    if confidence_levels is None:
        confidence_levels = np.concatenate([np.arange(0.01, 0.95, 0.01), [0.95], np.arange(0.96, 1.00, 0.01)])

    pr = returns @ weights
    mu_h, sig_h, _, mu_d, sig_d = _horizon_stats_from_daily(pr, horizon_days)

    s_d        = float(skew(pr, bias=False)) if sig_d > 0 else 0.0
    k_excess_d = float(kurtosis(pr, bias=False, fisher=True)) if sig_d > 0 else 0.0

    s_h        = s_d / np.sqrt(horizon_days) if horizon_days > 0 else s_d
    k_excess_h = k_excess_d / horizon_days   if horizon_days > 0 else k_excess_d

    out = []
    for conf in confidence_levels:
        alpha = 1.0 - conf
        z  = norm.ppf(alpha)
        z2 = z*z
        z3 = z2*z
        z_cf = (z
                + (1.0/6.0)*(z2 - 1.0)*s_h
                + (1.0/24.0)*(z3 - 3.0*z)*k_excess_h
                - (1.0/36.0)*(2.0*z3 - 5.0*z)*(s_h**2))
        var  = -(mu_h + z_cf * sig_h)
        cvar = -(mu_h - sig_h * (norm.pdf(z_cf) / alpha)) if alpha > 0 else var
        out.append({"Confidence Level": round(conf, 4), "VaR": var * 100.0, "CVaR": cvar * 100.0})
    return pd.DataFrame(out)

# Función para calcular VaR/CVaR al 95% para un portafolio específico
def calculate_var_cvar_95(returns: np.ndarray, weights: np.ndarray, horizon_days: int = 1) -> dict:
    """
    Calcula VaR y CVaR al 95% de confianza para un portafolio usando los tres métodos.
    Retorna un diccionario con los resultados.
    """
    result = {}
    confidence_level = 0.95
    
    # Normal
    try:
        df_normal = var_cvar_param_normal(returns, weights, [confidence_level], horizon_days)
        result['VaR_Normal_95'] = df_normal.iloc[0]['VaR']
        result['CVaR_Normal_95'] = df_normal.iloc[0]['CVaR']
    except:
        result['VaR_Normal_95'] = np.nan
        result['CVaR_Normal_95'] = np.nan
    
    # Histórico
    try:
        df_hist = var_cvar_historico(returns, weights, [confidence_level], horizon_days)
        result['VaR_Historico_95'] = df_hist.iloc[0]['VaR']
        result['CVaR_Historico_95'] = df_hist.iloc[0]['CVaR']
    except:
        result['VaR_Historico_95'] = np.nan
        result['CVaR_Historico_95'] = np.nan
    
    # Cornish-Fisher
    try:
        df_cf = var_cvar_cornish_fisher(returns, weights, [confidence_level], horizon_days)
        result['VaR_CornishFisher_95'] = df_cf.iloc[0]['VaR']
        result['CVaR_CornishFisher_95'] = df_cf.iloc[0]['CVaR']
    except:
        result['VaR_CornishFisher_95'] = np.nan
        result['CVaR_CornishFisher_95'] = np.nan
    
    return result

# ====================== PROCESO PRINCIPAL ======================

def main():
    print("=" * 80)
    print("ANÁLISIS DE PORTAFOLIO - MARKOWITZ + CML + VaR/CVaR (Normal, Histórico, Cornish–Fisher)")
    print("=" * 80)

    # 1) Datos de precios
    print("\n1. Descargando datos históricos...")
    try:
        data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Adj Close']
        data.index = data.index.tz_localize(None)
        data = data.dropna()
        if data.empty:
            raise RuntimeError("No se obtuvieron precios válidos.")
        print(f"   [OK] {len(data)} días de trading | {data.index[0].date()} → {data.index[-1].date()}")
    except Exception as e:
        print(f"   [ERROR] Descargando datos: {e}")
        return

    # 2) Retornos y matrices
    print("\n2. Calculando retornos logarítmicos...")
    log_returns = calculate_returns(data)
    mean_returns = log_returns.mean().values              # diarios (log)
    cov_matrix   = log_returns.cov().values              # diaria

    # Anualizar
    mean_returns_annual = mean_returns * TRADING_DAYS
    cov_matrix_annual   = cov_matrix   * TRADING_DAYS
    cov_matrix_annual   = cov_matrix_annual + np.eye(len(TICKERS)) * COV_REG_EPS  # regularización

    print(f"   [OK] Retorno promedio anual (promedio de activos): {mean_returns_annual.mean()*100:.2f}%")
    print(f"   [OK] Volatilidad anual (promedio de activos): {np.sqrt(np.diag(cov_matrix_annual)).mean()*100:.2f}%")

    # Configuración de optimización
    n_assets       = len(TICKERS)
    bounds         = [(0.0, 1.0) for _ in range(n_assets)]
    initial_guess  = np.ones(n_assets) / n_assets

    def neg_return_annual(w):
        return -float(mean_returns_annual @ w)

    # 3) Portafolio de mínima varianza
    print("\n3. Optimizando portafolio de mínima varianza...")
    min_var_result = minimize(
        lambda w: portfolio_variance_annual(w, cov_matrix_annual),
        x0=initial_guess, bounds=bounds,
        constraints={"type": "eq", "fun": weight_sum_constraint},
        method='SLSQP'
    )
    if not min_var_result.success:
        print("   [ERROR] Optimización de mínima varianza falló")
        return
    min_var_weights = min_var_result.x
    min_var_risk    = np.sqrt(portfolio_variance_annual(min_var_weights, cov_matrix_annual)) * 100.0
    min_var_return  = float(mean_returns_annual @ min_var_weights) * 100.0
    print(f"   [OK] Riesgo: {min_var_risk:.2f}%, Retorno: {min_var_return:.2f}%")

    # 4) Portafolio de máximo retorno
    print("\n4. Optimizando portafolio de máximo retorno...")
    max_ret_result = minimize(
        neg_return_annual,
        x0=initial_guess, bounds=bounds,
        constraints={"type": "eq", "fun": weight_sum_constraint},
        method='SLSQP'
    )
    if not max_ret_result.success:
        print("   [ERROR] Optimización de máximo retorno falló")
        return
    max_ret_weights = max_ret_result.x
    max_ret_risk    = np.sqrt(portfolio_variance_annual(max_ret_weights, cov_matrix_annual)) * 100.0
    max_ret_return  = float(mean_returns_annual @ max_ret_weights) * 100.0
    print(f"   [OK] Riesgo: {max_ret_risk:.2f}%, Retorno: {max_ret_return:.2f}%")

    # 5) Frontera eficiente (versión basada en medias de activos, 200 puntos)
    print("\n5. Calculando frontera eficiente (versión de referencia)...")
    low_return   = mean_returns_annual.min() * 0.5
    high_return  = mean_returns_annual.max() * 1.5
    target_returns = np.linspace(low_return, high_return, 200)

    frontier_risks, frontier_returns, frontier_weights = [], [], []
    for tr in target_returns:
        constraints = (
            {"type": "eq", "fun": weight_sum_constraint},
            {"type": "eq", "fun": (lambda w, tr=tr: float(mean_returns_annual @ w) - tr)}
        )
        frontier_result = minimize(
            lambda w: portfolio_variance_annual(w, cov_matrix_annual),
            x0=initial_guess, bounds=bounds, constraints=constraints, method='SLSQP'
        )
        if frontier_result.success:
            w_opt = frontier_result.x
            std   = np.sqrt(portfolio_variance_annual(w_opt, cov_matrix_annual)) * 100.0
            frontier_risks.append(std)
            frontier_returns.append(tr * 100.0)
            frontier_weights.append(w_opt)

    if len(frontier_risks) == 0:
        # Si nada fue factible, usa al menos los extremos ya calculados
        frontier_risks   = [min_var_risk, max_ret_risk]
        frontier_returns = [min_var_return, max_ret_return]
        frontier_weights = [min_var_weights, max_ret_weights]

    # DataFrame de la frontera eficiente completa (para exportar)
    df_efficient_frontier = pd.DataFrame(frontier_weights, columns=TICKERS)
    df_efficient_frontier['Riesgo Anual (%)'] = frontier_risks
    df_efficient_frontier['Retorno Esperado Anual Logarítmico (%)'] = frontier_returns
    cols = ['Riesgo Anual (%)', 'Retorno Esperado Anual Logarítmico (%)'] + TICKERS
    df_efficient_frontier = df_efficient_frontier[cols]
    print(f"   [ok] {len(frontier_risks)} puntos factibles en la frontera")

    # 6) Tasa libre y portafolio tangente
    print("\n6. Calculando portafolio tangente y CML...")
    try:
        risk_free_rate = obtener_tasa_libre(RISK_FREE_SYMBOL, START_DATE, END_DATE, require=REQUIRE_RISK_FREE_DATA)
    except Exception as e:
        print(f"   [ERROR] Tasa libre: {e}")
        return
    print(f"   Tasa libre de riesgo anual (5Y): {risk_free_rate*100:.2f}%")

    def negative_sharpe_ratio(w):
        port_ret = float(mean_returns_annual @ w)
        port_vol = np.sqrt(portfolio_variance_annual(w, cov_matrix_annual))
        if port_vol == 0:
            return np.inf
        return - (port_ret - risk_free_rate) / port_vol

    tangente_result = minimize(
        negative_sharpe_ratio, x0=initial_guess,
        bounds=bounds, constraints={"type": "eq", "fun": weight_sum_constraint},
        method='SLSQP'
    )
    if not tangente_result.success:
        print("   [x] Optimización del portafolio tangente falló")
        return
    tangente_weights = tangente_result.x
    tangente_return  = float(mean_returns_annual @ tangente_weights) * 100.0
    tangente_risk    = np.sqrt(portfolio_variance_annual(tangente_weights, cov_matrix_annual)) * 100.0
    sharpe_ratio     = (tangente_return/100.0 - risk_free_rate) / (tangente_risk/100.0)
    print(f"   [ok] Tangente → Riesgo: {tangente_risk:.2f}%, Retorno: {tangente_return:.2f}% | Sharpe: {sharpe_ratio:.3f}")

    # CML (sobre el panel principal)
    cml_risks   = np.linspace(0, max(frontier_risks) * 1.2, 100)
    cml_returns = risk_free_rate * 100.0 + sharpe_ratio * cml_risks

    # 7) VaR & CVaR (3 métodos)
    print("\n7. Calculando VaR y CVaR (Normal, Histórico, Cornish–Fisher)...")
    portfolios = {
        'Min Varianza': min_var_weights,
        'Max Retorno':  max_ret_weights,
        'Tangente':     tangente_weights
    }

    var_cvar_results = {"Normal": {}, "Historico": {}, "CornishFisher": {}}

    for name, w in portfolios.items():
        try:
            var_cvar_results["Normal"][name] = var_cvar_param_normal(
                log_returns.values, w, horizon_days=VAR_HORIZON_DAYS
            )
        except Exception as e:
            var_cvar_results["Normal"][name] = None
            print(f"   [Aviso] Normal {name}: {e}")

        try:
            var_cvar_results["Historico"][name] = var_cvar_historico(
                log_returns.values, w, horizon_days=VAR_HORIZON_DAYS
            )
        except Exception as e:
            var_cvar_results["Historico"][name] = None
            print(f"   [Aviso] Historico {name}: {e}")

        try:
            var_cvar_results["CornishFisher"][name] = var_cvar_cornish_fisher(
                log_returns.values, w, horizon_days=VAR_HORIZON_DAYS
            )
        except Exception as e:
            var_cvar_results["CornishFisher"][name] = None
            print(f"   [Aviso] CornishFisher {name}: {e}")

    # Impresión VaR/CVaR al 95% para cada método y portafolio
    def extract_95(df: pd.DataFrame):
        idx = (df['Confidence Level'] - 0.95).abs().idxmin()
        return float(df.loc[idx, 'VaR']), float(df.loc[idx, 'CVaR'])

    for method in ["Normal", "Historico", "CornishFisher"]:
        print(f"\n   --- {method} ---")
        for name in portfolios:
            try:
                dfm = var_cvar_results[method][name]
                if dfm is None:
                    raise RuntimeError("resultados no disponibles")
                v95, c95 = extract_95(dfm)
                print(f"   {name}: VaR(95%)={v95:.2f}%, CVaR(95%)={c95:.2f}%")
            except Exception as e:
                print(f"   {name}: (no disponible) -> {e}")

    # 7.5) NUEVO: Calcular VaR y CVaR para TODOS los portafolios de la frontera eficiente
    print("\n7.5. Calculando VaR y CVaR para todos los portafolios de la frontera eficiente...")
    frontier_var_cvar = []
    
    for i, w in enumerate(frontier_weights):
        if i % 20 == 0:  # Mostrar progreso cada 20 portafolios
            print(f"   Procesando portafolio {i+1}/{len(frontier_weights)}...")
        
        # Calcular VaR y CVaR al 95% para este portafolio
        risk_metrics = calculate_var_cvar_95(log_returns.values, w, VAR_HORIZON_DAYS)
        
        # Agregar información del portafolio
        portfolio_info = {
            'Portafolio_ID': i + 1,
            'Riesgo Anual (%)': frontier_risks[i],
            'Retorno Esperado Anual (%)': frontier_returns[i]
        }
        
        # Agregar los pesos
        for j, ticker in enumerate(TICKERS):
            portfolio_info[f'Peso_{ticker}'] = w[j]
        
        # Combinar toda la información
        portfolio_info.update(risk_metrics)
        frontier_var_cvar.append(portfolio_info)
    
    # Crear DataFrame con todos los resultados
    df_frontier_var_cvar = pd.DataFrame(frontier_var_cvar)
    
    # Reordenar columnas para mejor visualización
    cols_order = ['Portafolio_ID', 'Riesgo Anual (%)', 'Retorno Esperado Anual (%)']
    cols_order += [f'Peso_{ticker}' for ticker in TICKERS]
    cols_order += ['VaR_Normal_95', 'CVaR_Normal_95', 
                   'VaR_Historico_95', 'CVaR_Historico_95',
                   'VaR_CornishFisher_95', 'CVaR_CornishFisher_95']
    df_frontier_var_cvar = df_frontier_var_cvar[cols_order]
    
    print(f"   [ok] VaR/CVaR calculado para {len(frontier_var_cvar)} portafolios de la frontera")

    # 8) Visualización
    print("\n8. Generando gráficos...")

    border_color = 'darkblue'
    border_width = 1.5

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')

    # Panel 1: Frontera Eficiente (estilo versión referencia) + CML + puntos clave
    ax1 = plt.subplot(2, 2, 1)

    # Puntos clave
    label_min_var = f'Min. Varianza (R:{min_var_return:.2f}%, σ:{min_var_risk:.2f}%)'
    label_max_ret = f'Máx. Retorno (R:{max_ret_return:.2f}%, σ:{max_ret_risk:.2f}%)'
    label_tangent = f'Tangente (R:{tangente_return:.2f}%, σ:{tangente_risk:.2f}%, SR:{sharpe_ratio:.3f})'

    plt.scatter(min_var_risk, min_var_return, color='red', s=100, label=label_min_var, zorder=5, edgecolors='black')
    plt.scatter(max_ret_risk,  max_ret_return, color='green', s=100, label=label_max_ret, zorder=5, edgecolors='black')
    plt.scatter(tangente_risk, tangente_return, color='purple', s=100, label=label_tangent, zorder=5, edgecolors='black')

    # Frontera (línea azul) y CML (línea naranja punteada)
    plt.plot(frontier_risks, frontier_returns, label="Frontera Eficiente Completa", color='blue', linewidth=2)
    plt.plot(cml_risks, cml_returns, label='CML (Capital Market Line)', color='orange', linestyle='--', linewidth=2)

    plt.xlabel('Riesgo Anual (%)')
    plt.ylabel('Retorno Esperado Anual Logarítmico (%)')
    plt.title('Frontera Eficiente de Markowitz (Log Anualizado) + CML')
    legend = plt.legend(loc='best', fontsize=8, frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(border_color)
    legend.get_frame().set_linewidth(border_width)
    legend.get_frame().set_alpha(0.95)
    plt.grid(True, alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_edgecolor(border_color); spine.set_linewidth(border_width)

    # Panel 2: Pesos
    ax2 = plt.subplot(2, 2, 2)
    x = np.arange(len(TICKERS))
    width = 0.25
    plt.bar(x - width, min_var_weights, width, label='Min. Varianza', color='red',   alpha=0.7)
    plt.bar(x,          max_ret_weights, width, label='Máx. Retorno',  color='green', alpha=0.7)
    plt.bar(x + width,  tangente_weights, width, label='Tangente',     color='purple',alpha=0.7)
    plt.xlabel('Activos'); plt.ylabel('Peso'); plt.title('Composición de Portafolios')
    plt.xticks(x, TICKERS)
    legend = plt.legend(frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(border_color)
    legend.get_frame().set_linewidth(border_width)
    legend.get_frame().set_alpha(0.95)
    plt.grid(True, alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_edgecolor(border_color); spine.set_linewidth(border_width)

    # Panel 3: VaR (Normal)
    ax3 = plt.subplot(2, 2, 3)
    if var_cvar_results["Normal"]:
        for name, dfv in var_cvar_results["Normal"].items():
            if dfv is not None:
                plt.plot(dfv['Confidence Level'], dfv['VaR'], label=f'{name}', linewidth=2)
    plt.title('Value at Risk (VaR) - Paramétrico (Normal)')
    plt.xlabel('Nivel de Confianza'); plt.ylabel('VaR (%)')
    legend = plt.legend(frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(border_color)
    legend.get_frame().set_linewidth(border_width)
    legend.get_frame().set_alpha(0.95)
    plt.grid(True, alpha=0.3); plt.axvline(x=0.95, color='gray', linestyle=':', alpha=0.5)
    for spine in ax3.spines.values():
        spine.set_edgecolor(border_color); spine.set_linewidth(border_width)

    # Panel 4: CVaR (Normal)
    ax4 = plt.subplot(2, 2, 4)
    if var_cvar_results["Normal"]:
        for name, dfv in var_cvar_results["Normal"].items():
            if dfv is not None:
                plt.plot(dfv['Confidence Level'], dfv['CVaR'], label=f'{name}', linewidth=2)
    plt.title('Conditional VaR (CVaR) - Paramétrico (Normal)')
    plt.xlabel('Nivel de Confianza'); plt.ylabel('CVaR (%)')
    legend = plt.legend(frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(border_color)
    legend.get_frame().set_linewidth(border_width)
    legend.get_frame().set_alpha(0.95)
    plt.grid(True, alpha=0.3); plt.axvline(x=0.95, color='gray', linestyle=':', alpha=0.5)
    for spine in ax4.spines.values():
        spine.set_edgecolor(border_color); spine.set_linewidth(border_width)

    plt.suptitle(f'Análisis de Portafolio - {", ".join(TICKERS)}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Figuras extra: Histórico y Cornish–Fisher
    # Histórico
    fig_h = plt.figure(figsize=(14, 5))
    fig_h.patch.set_facecolor('white')

    axh1 = plt.subplot(1, 2, 1)
    for name, dfh in var_cvar_results["Historico"].items():
        if dfh is not None:
            plt.plot(dfh['Confidence Level'], dfh['VaR'], label=f'{name}', linewidth=2)
    plt.title('VaR - Histórico')
    plt.xlabel('Nivel de Confianza'); plt.ylabel('VaR (%)')
    leg = plt.legend(frameon=True); leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor(border_color); leg.get_frame().set_linewidth(border_width); leg.get_frame().set_alpha(0.95)
    plt.grid(True, alpha=0.3); plt.axvline(x=0.95, color='gray', linestyle=':', alpha=0.5)
    for sp in axh1.spines.values(): sp.set_edgecolor(border_color); sp.set_linewidth(border_width)

    axh2 = plt.subplot(1, 2, 2)
    for name, dfh in var_cvar_results["Historico"].items():
        if dfh is not None:
            plt.plot(dfh['Confidence Level'], dfh['CVaR'], label=f'{name}', linewidth=2)
    plt.title('CVaR - Histórico')
    plt.xlabel('Nivel de Confianza'); plt.ylabel('CVaR (%)')
    leg = plt.legend(frameon=True); leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor(border_color); leg.get_frame().set_linewidth(border_width); leg.get_frame().set_alpha(0.95)
    plt.grid(True, alpha=0.3); plt.axvline(x=0.95, color='gray', linestyle=':', alpha=0.5)
    for sp in axh2.spines.values(): sp.set_edgecolor(border_color); sp.set_linewidth(border_width)

    plt.tight_layout()
    plt.show()

    # Cornish–Fisher
    fig_cf = plt.figure(figsize=(14, 5))
    fig_cf.patch.set_facecolor('white')

    axc1 = plt.subplot(1, 2, 1)
    for name, dfc in var_cvar_results["CornishFisher"].items():
        if dfc is not None:
            plt.plot(dfc['Confidence Level'], dfc['VaR'], label=f'{name}', linewidth=2)
    plt.title('VaR - Cornish–Fisher')
    plt.xlabel('Nivel de Confianza'); plt.ylabel('VaR (%)')
    leg = plt.legend(frameon=True); leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor(border_color); leg.get_frame().set_linewidth(border_width); leg.get_frame().set_alpha(0.95)
    plt.grid(True, alpha=0.3); plt.axvline(x=0.95, color='gray', linestyle=':', alpha=0.5)
    for sp in axc1.spines.values(): sp.set_edgecolor(border_color); sp.set_linewidth(border_width)

    axc2 = plt.subplot(1, 2, 2)
    for name, dfc in var_cvar_results["CornishFisher"].items():
        if dfc is not None:
            plt.plot(dfc['Confidence Level'], dfc['CVaR'], label=f'{name}', linewidth=2)
    plt.title('CVaR - Cornish–Fisher')
    plt.xlabel('Nivel de Confianza'); plt.ylabel('CVaR (%)')
    leg = plt.legend(frameon=True); leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor(border_color); leg.get_frame().set_linewidth(border_width); leg.get_frame().set_alpha(0.95)
    plt.grid(True, alpha=0.3); plt.axvline(x=0.95, color='gray', linestyle=':', alpha=0.5)
    for sp in axc2.spines.values(): sp.set_edgecolor(border_color); sp.set_linewidth(border_width)

    plt.tight_layout()
    plt.show()

    # 9) Exportar a Excel
    print(f"\n9. Exportando resultados a Excel: {OUTPUT_FILE}")
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            # Hoja: Precios ajustados
            data.to_excel(writer, sheet_name="Precios Ajustados")

            # Hoja: Retornos logarítmicos
            log_returns.to_excel(writer, sheet_name="Retornos Log Diarios")

            # Hoja: Matriz de covarianza anual (regularizada)
            df_cov_matrix = pd.DataFrame(cov_matrix_annual, index=TICKERS, columns=TICKERS)
            df_cov_matrix.to_excel(writer, sheet_name="Matriz Covarianza Anual")

            # Hoja: Resumen de portafolios (incluye Tangente)
            df_portfolios = pd.DataFrame({
                "Portafolio": ["Min Varianza", "Máx. Retorno", "Tangente"],
                "Riesgo Anual (%)": [min_var_risk, max_ret_risk, tangente_risk],
                "Retorno Anual Logarítmico (%)": [min_var_return, max_ret_return, tangente_return],
                "Sharpe Ratio": [
                    (min_var_return/100.0 - risk_free_rate) / (min_var_risk/100.0),
                    (max_ret_return/100.0 - risk_free_rate) / (max_ret_risk/100.0),
                    (tangente_return/100.0 - risk_free_rate) / (tangente_risk/100.0)
                ]
            })
            for i, tkr in enumerate(TICKERS):
                df_portfolios[tkr] = [min_var_weights[i], max_ret_weights[i], tangente_weights[i]]
            df_portfolios.to_excel(writer, sheet_name="Resumen Portafolios", index=False)

            # Hoja: Frontera Eficiente (formato versión referencia)
            df_efficient_frontier.to_excel(writer, sheet_name="Frontera Eficiente", index=False)

            # NUEVA HOJA: VaR/CVaR para toda la Frontera Eficiente
            df_frontier_var_cvar.to_excel(writer, sheet_name="VaR_CVaR_Frontera_Eficiente", index=False)

            # Hojas: VaR/CVaR por método y portafolio
            for method, dct in var_cvar_results.items():
                for name, dfv in dct.items():
                    if dfv is not None:
                        dfv.to_excel(writer, sheet_name=f"{method}_{name}".replace(" ", "_"), index=False)

            # Hoja: Estadísticas por activo
            stats = pd.DataFrame({
                'Ticker': TICKERS,
                'Retorno Anual (%)': mean_returns_annual * 100.0,
                'Volatilidad Anual (%)': np.sqrt(np.diag(cov_matrix_annual)) * 100.0,
                'Sharpe (activo)': (mean_returns_annual - risk_free_rate) / np.sqrt(np.diag(cov_matrix_annual))
            })
            stats.to_excel(writer, sheet_name="Estadisticas", index=False)

            # Hoja: Parámetros
            pd.DataFrame({
                "Parametro": [
                    "Tickers", "Start", "End", "TRADING_DAYS",
                    "Risk-free Symbol", "Risk-free (decimal)",
                    "VaR Horizon (dias)", "Cov Regularization (eps)"
                ],
                "Valor": [
                    ", ".join(TICKERS), START_DATE, END_DATE, TRADING_DAYS,
                    RISK_FREE_SYMBOL, risk_free_rate,
                    VAR_HORIZON_DAYS, COV_REG_EPS
                ]
            }).to_excel(writer, sheet_name="Parametros", index=False)

            # Hoja: Resumen VaR/CVaR al 95% (todos los métodos y portafolios)
            rows_95 = []
            for method, dct in var_cvar_results.items():
                for name, dfv in dct.items():
                    try:
                        if dfv is None:
                            raise RuntimeError("sin datos")
                        idx = (dfv['Confidence Level'] - 0.95).abs().idxmin()
                        rows_95.append({
                            "Metodo": method,
                            "Portafolio": name,
                            "VaR95(%)": float(dfv.loc[idx, "VaR"]),
                            "CVaR95(%)": float(dfv.loc[idx, "CVaR"])
                        })
                    except Exception:
                        rows_95.append({
                            "Metodo": method,
                            "Portafolio": name,
                            "VaR95(%)": np.nan,
                            "CVaR95(%)": np.nan
                        })
            pd.DataFrame(rows_95).to_excel(writer, sheet_name="VaR_CVaR_95", index=False)

        print("   [ok] Archivo exportado exitosamente")
        print(f"   [ok] Nueva hoja 'VaR_CVaR_Frontera_Eficiente' agregada con {len(frontier_var_cvar)} portafolios")
    except Exception as e:
        print(f"   [x] Error al exportar: {e}")

    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)

if __name__ == "__main__":
    main()



