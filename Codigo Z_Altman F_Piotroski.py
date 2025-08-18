"""
Autor: Patricio Román Mery Araya. Se autoriza su uso bajo los términos y condiciones de la 
licencia Creative Commons Atribución-No Comercial 4.0 Internacional (CC BY-NC 4.0): 
https://creativecommons.org/licenses/by-nc/4.0/

Copia el Script en la carpeta 'C:/users/name_usuario/sec-edgar-filings' que es la que utiliza
SEC_Downloader al ejecutarlo se crearan las tablas

"""

import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ===================== CONFIGURACION =====================

TICKERS = ["ORCL", "META", "QCOM", "CRM", "INTC"]

USER_AGENT = "tu email"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
BASE_DIR = os.path.abspath("sec-edgar-tech-scores")
os.makedirs(BASE_DIR, exist_ok=True)

MIN_YEARS = 5
PAUSE_SEC = 0.60
MAX_RETRIES = 5
PRICE_WINDOW_DAYS = 10

# Para la tabla "ultimos 5 anos" mostramos Altman original en FY vs GF-like.
GF_COMPARISON_MODE = True

# Modo de MVE para la vista "GF-like":
# - 'current'  => Market Cap actual (precio * shares actuales)
# - 'fy_price' => precio cercano al FY (con shares EOP/SEC) manteniendo formula original (5 factores)
GF_MVE_MODE = "current"  # "current" | "fy_price"

# ===================== UTILIDADES HTTP / SEC =====================

def fetch_json(url: str, headers: dict, max_retries: int = MAX_RETRIES, pause: float = 0.6):
    last_status = None
    for i in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=45)
            last_status = resp.status_code
            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception:
                    return json.loads(resp.content.decode("utf-8", errors="ignore"))
            elif resp.status_code in (403, 429):
                wait = pause * (i + 1)
                print(f"  * SEC respondio {resp.status_code} en {url}. Esperando {wait:.1f}s y reintentando...")
                time.sleep(wait)
            else:
                wait = pause * (i + 1)
                print(f"  * HTTP {resp.status_code} en {url}. Reintento en {wait:.1f}s...")
                time.sleep(wait)
        except requests.exceptions.RequestException as e:
            wait = pause * (i + 1)
            print(f"  * Error de red: {e}. Reintento en {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"No se pudo descargar JSON desde {url} (status={last_status}). "
                       f"Verifica USER_AGENT y reduce frecuencia de consultas.")

def load_ticker_cik_map(headers: dict) -> Dict[str, Tuple[str, Optional[int], Optional[str]]]:
    url = "https://www.sec.gov/files/company_tickers.json"
    data = fetch_json(url, headers)
    mapping = {}
    if isinstance(data, dict):
        it = data.items()
    else:
        it = enumerate(data)
    for _, row in it:
        tkr = str(row.get("ticker", "")).upper().strip()
        cik = str(row.get("cik_str", "")).strip()
        sic = row.get("sic", None)
        sicd = row.get("sic_description", None)
        if tkr and cik:
            mapping[tkr] = (str(cik).zfill(10), int(sic) if sic else None, sicd)
    return mapping

def get_company_facts(cik10: str, headers: dict) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    return fetch_json(url, headers)

# ===================== TAGS XBRL (ampliados) =====================

TAGS = {
    "Assets": [("us-gaap", "Assets")],
    "Liabilities": [
        ("us-gaap", "Liabilities"),
        ("us-gaap", "LiabilitiesAndStockholdersEquity"),  # cuidado: validamos contra TA/SE
        ("us-gaap", "TotalLiabilities"),
        ("us-gaap", "LiabilitiesNoncurrent"),
    ],
    "AssetsCurrent": [("us-gaap", "AssetsCurrent")],
    "LiabilitiesCurrent": [("us-gaap", "LiabilitiesCurrent")],
    "RetainedEarnings": [
        ("us-gaap", "RetainedEarningsAccumulatedDeficit"),
        ("us-gaap", "AccumulatedDeficit"),
        ("us-gaap", "RetainedEarnings")
    ],
    "OperatingIncome": [
        ("us-gaap", "OperatingIncomeLoss"),
        ("us-gaap", "EarningsBeforeInterestAndTaxes"),
        ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
        ("us-gaap", "IncomeFromOperations")
    ],
    "Revenue": [
        ("us-gaap", "SalesRevenueNet"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueGoodsNet"),
        ("us-gaap", "TotalRevenues")
    ],
    "NetIncome": [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "ProfitLoss")
    ],
    "CFO": [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
    ],
    "LongTermDebt": [
        ("us-gaap", "LongTermDebtNoncurrent"),
        ("us-gaap", "LongTermDebt"),
        ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
        ("us-gaap", "NoncurrentLiabilities")
    ],
    "SharesOutstanding": [
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic")
    ],
    "GrossProfit": [("us-gaap", "GrossProfit")],
    "StockholdersEquity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ("us-gaap", "TotalEquity")
    ]
}

# ===================== AUXILIARES DE SERIES =====================

def _collect_units(facts: dict, taxonomy: str, tag: str) -> dict:
    try:
        return facts["facts"][taxonomy][tag]["units"]
    except KeyError:
        return {}

def _prefer_units(units_dict: dict, preferred_units: Tuple[str, ...]) -> List[dict]:
    for u in preferred_units:
        if u in units_dict:
            return units_dict[u]
    for _, v in units_dict.items():
        return v
    return []

def _filter_annual(obs_list: List[dict]) -> List[dict]:
    ks = []
    for x in obs_list:
        form = str(x.get("form", "")).upper()
        if form in {"10-K", "10-K/A"}:
            ks.append(x)
    if not ks:
        ks = [x for x in obs_list if x.get("fy")]
    def _end(e):
        try:
            return datetime.strptime(e.get("end", ""), "%Y-%m-%d")
        except Exception:
            return datetime.min
    ks.sort(key=_end)
    return ks

def merge_candidate_series(facts: dict, candidates: List[Tuple[str, str]],
                           preferred_units: Tuple[str, ...]) -> List[dict]:
    bag: List[dict] = []
    for taxonomy, tag in candidates:
        units = _collect_units(facts, taxonomy, tag)
        if not units:
            continue
        obs = _prefer_units(units, preferred_units)
        if not obs:
            continue
        bag.extend(_filter_annual(obs))
    if not bag:
        return []
    by_fy: Dict[int, dict] = {}
    for e in bag:
        fy, end, val = e.get("fy"), e.get("end"), e.get("val")
        if not fy or not end:
            continue
        try:
            fy_i = int(fy)
        except Exception:
            continue
        if fy_i not in by_fy or (end > by_fy[fy_i].get("end", "")):
            by_fy[fy_i] = e
    out = [by_fy[k] for k in sorted(by_fy.keys())]
    return out

def to_float_or_none(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

# ===================== MEJORAS: Fallbacks & Validaciones =====================

def calculate_total_liabilities(TA: Optional[float], BVE: Optional[float], TL_direct: Optional[float]) -> Optional[float]:
    """
    Calcula TL con fallback:
    - Si TL_direct es razonable, usalo.
    - Si TL_direct ~= TA (probable 'LiabilitiesAndStockholdersEquity'), descartalo.
    - Si hay TA y BVE: TL = TA - BVE.
    """
    if TL_direct is not None and TL_direct > 0:
        if TA is not None:
            # sanity: TL no debe superar a TA de forma relevante
            if TL_direct > TA * 1.05:
                TL_direct = None
            # si es casi igual a TA y hay equity positivo, podria ser L+SE (descartamos)
            if BVE is not None and BVE > 0 and abs(TL_direct - TA) / max(1.0, TA) < 0.02:
                TL_direct = None
        if TL_direct is not None:
            return TL_direct
    if TA is not None and BVE is not None and TA > BVE:
        return TA - BVE
    return None

def validate_shares_outstanding(shares: Optional[float], ticker: str, fy: int) -> Optional[float]:
    """
    Valida/corrige unidades anomalas en shares outstanding (p.ej., *100 o *1000).
    """
    if shares is None:
        return None
    limits_m = {
        "ORCL": (2000, 6000),
        "META": (2000, 3500),
        "QCOM": (1000, 2000),
        "CRM":  (500, 1500),
        "INTC": (3000, 6000)
    }
    shares_m = shares / 1_000_000.0
    if ticker in limits_m:
        lo, hi = limits_m[ticker]
        if shares_m > hi * 1000:
            print(f"  WARNING: Corrigiendo shares anomalas {ticker} FY{fy}: {shares:,.0f} -> {shares/1000:,.0f}")
            return shares / 1000.0
        elif shares_m > hi * 100:
            print(f"  WARNING: Corrigiendo shares anomalas {ticker} FY{fy}: {shares:,.0f} -> {shares/100:,.0f}")
            return shares / 100.0
    return shares

def compute_altman_with_equity_fallback(WC, TA, RE, EBIT, MVE, TL, BVE, SALES) -> Optional[float]:
    """
    Altman original con fallback: si falta TL, intenta TL = TA - BVE (equity).
    """
    TL_use = TL
    if (TL_use is None or TL_use <= 0) and (TA is not None and BVE is not None and TA > BVE):
        TL_use = TA - BVE
        if TL_use is not None and TL_use > 0:
            print(f"    INFO: TL fallback = TA - BVE = {TA:,.0f} - {BVE:,.0f} = {TL_use:,.0f}")
    return compute_altman_original(WC, TA, RE, EBIT, MVE, TL_use, SALES)

# ===================== METRICAS =====================

def compute_altman_original(WC, TA, RE, EBIT, MVE, TL, SALES) -> Optional[float]:
    try:
        if TA in (None, 0) or TL in (None, 0):
            return None
        X1 = (WC or 0.0) / TA
        X2 = (RE or 0.0) / TA
        X3 = (EBIT or 0.0) / TA
        X4 = (MVE or 0.0) / TL
        X5 = (SALES or 0.0) / TA
        return 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    except Exception:
        return None

def compute_altman_zpp(WC, TA, RE, EBIT, TL, BVE) -> Optional[float]:
    try:
        if TA in (None, 0) or TL in (None, 0):
            return None
        X1 = (WC or 0.0) / TA
        X2 = (RE or 0.0) / TA
        X3 = (EBIT or 0.0) / TA
        X4 = (BVE or 0.0) / TL
        return 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    except Exception:
        return None

def safe_ratio(num, den) -> Optional[float]:
    try:
        if num is None or den in (None, 0):
            return None
        return float(num) / float(den)
    except Exception:
        return None

def compute_piotroski_signals_FY(t_vals: dict, t1_vals: dict) -> Tuple[int, dict]:
    signals = {}
    roa_t = safe_ratio(t_vals.get("NI"), t_vals.get("TA"))
    roa_t1 = safe_ratio(t1_vals.get("NI"), t1_vals.get("TA")) if t1_vals else None
    signals["ROA_pos"] = 1 if (roa_t is not None and roa_t > 0) else 0
    signals["Delta_ROA_pos"] = 1 if (roa_t is not None and roa_t1 is not None and roa_t > roa_t1) else 0
    cfo_t = t_vals.get("CFO")
    ni_t = t_vals.get("NI")
    signals["CFO_pos"] = 1 if (cfo_t is not None and cfo_t > 0) else 0
    signals["Accruals"] = 1 if (cfo_t is not None and ni_t is not None and cfo_t > ni_t) else 0
    lev_t  = safe_ratio(t_vals.get("LT"), t_vals.get("TA"))
    lev_t1 = safe_ratio(t1_vals.get("LT"), t1_vals.get("TA")) if t1_vals else None
    signals["Delta_Leverage_down"] = 1 if (lev_t is not None and lev_t1 is not None and lev_t <= lev_t1) else 0
    cr_t  = safe_ratio(t_vals.get("CA"), t_vals.get("CL"))
    cr_t1 = safe_ratio(t1_vals.get("CA"), t1_vals.get("CL")) if t1_vals else None
    signals["Delta_CurrentRatio_pos"] = 1 if (cr_t is not None and cr_t1 is not None and cr_t > cr_t1) else 0
    sh_t  = t_vals.get("Shares")
    sh_t1 = t1_vals.get("Shares") if t1_vals else None
    signals["No_Equity_Issuance"] = 1 if (sh_t is not None and sh_t1 is not None and sh_t <= sh_t1) else 0
    gm_t  = safe_ratio(t_vals.get("GP"), t_vals.get("Sales"))
    gm_t1 = safe_ratio(t1_vals.get("GP"), t1_vals.get("Sales")) if t1_vals else None
    signals["Delta_GrossMargin_pos"] = 1 if (gm_t is not None and gm_t1 is not None and gm_t > gm_t1) else 0
    at_t  = safe_ratio(t_vals.get("Sales"), t_vals.get("TA"))
    at_t1 = safe_ratio(t1_vals.get("Sales"), t1_vals.get("TA")) if t1_vals else None
    signals["Delta_AssetTurnover_pos"] = 1 if (at_t is not None and at_t1 is not None and at_t > at_t1) else 0
    score = int(sum(signals.values()))
    return score, signals

def compute_piotroski_signals_GF_like(curr: dict, prev: dict) -> Tuple[int, dict]:
    signals = {}
    TA_curr = curr.get("TA")
    TA_prev = prev.get("TA") if prev else None
    TA_beg  = TA_prev
    TA_avg  = (TA_curr + TA_prev)/2.0 if (TA_curr is not None and TA_prev is not None) else None

    roa_t  = safe_ratio(curr.get("NI"), TA_beg)
    roa_t1 = safe_ratio(prev.get("NI"), prev.get("TA")) if prev else None
    signals["ROA_pos"] = 1 if (roa_t is not None and roa_t > 0) else 0
    signals["Delta_ROA_pos"] = 1 if (roa_t is not None and roa_t1 is not None and roa_t > roa_t1) else 0

    cfo_t = curr.get("CFO"); ni_t = curr.get("NI")
    signals["CFO_pos"] = 1 if (cfo_t is not None and cfo_t > 0) else 0
    signals["Accruals"] = 1 if (cfo_t is not None and ni_t is not None and cfo_t > ni_t) else 0

    lev_t  = safe_ratio(curr.get("LT"), TA_avg)
    lev_t1 = safe_ratio(prev.get("LT"), prev.get("TA")) if prev else None
    signals["Delta_Leverage_down"] = 1 if (lev_t is not None and lev_t1 is not None and lev_t <= lev_t1) else 0

    cr_t  = safe_ratio(curr.get("CA"), curr.get("CL"))
    cr_t1 = safe_ratio(prev.get("CA"), prev.get("CL")) if prev else None
    signals["Delta_CurrentRatio_pos"] = 1 if (cr_t is not None and cr_t1 is not None and cr_t > cr_t1) else 0

    sh_t  = curr.get("Shares"); sh_t1 = prev.get("Shares") if prev else None
    signals["No_Equity_Issuance"] = 1 if (sh_t is not None and sh_t1 is not None and sh_t <= sh_t1) else 0

    gm_t  = safe_ratio(curr.get("GP"), curr.get("Sales"))
    gm_t1 = safe_ratio(prev.get("GP"), prev.get("Sales")) if prev else None
    signals["Delta_GrossMargin_pos"] = 1 if (gm_t is not None and gm_t1 is not None and gm_t > gm_t1) else 0

    at_t  = safe_ratio(curr.get("Sales"), TA_beg)
    at_t1 = safe_ratio(prev.get("Sales"), prev.get("TA")) if prev else None
    signals["Delta_AssetTurnover_pos"] = 1 if (at_t is not None and at_t1 is not None and at_t > at_t1) else 0

    score = int(sum(signals.values()))
    return score, signals

# ===================== PRECIOS (robustos) =====================

def _find_price_around(yobj, target_dt: datetime, days: int, interval: str) -> Optional[float]:
    start = (target_dt - timedelta(days=days)).strftime("%Y-%m-%d")
    end   = (target_dt + timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        hist = yobj.history(start=start, end=end, interval=interval, auto_adjust=False)
        if hist is None or hist.empty:
            return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        if target_dt in hist.index:
            return float(hist.loc[target_dt, "Close"])
        distances = (hist.index.to_series().apply(lambda d: abs((d - target_dt).days)))
        idx_min = distances.idxmin()
        return float(hist.loc[idx_min, "Close"])
    except Exception:
        return None

def price_on_or_near_date(ticker: str, end_date_str: str, window_days: int = PRICE_WINDOW_DAYS) -> Optional[float]:
    if not end_date_str:
        return None
    try:
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    except Exception:
        return None
    y = yf.Ticker(ticker)
    for iv in ("1d", "1wk", "1mo"):
        px = _find_price_around(y, end_dt, days=window_days, interval=iv)
        if px is not None:
            return px
    for days in (60, 120, 365):
        for iv in ("1d", "1wk", "1mo"):
            px = _find_price_around(y, end_dt, days=days, interval=iv)
            if px is not None:
                return px
    try:
        fast = getattr(y, "fast_info", None) or {}
        px = fast.get("last_price", None)
        if px is None:
            recent = y.history(period="1mo")
            if recent is not None and not recent.empty:
                px = float(recent["Close"].iloc[-1])
        return float(px) if px is not None else None
    except Exception:
        return None

def current_price_and_shares(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    y = yf.Ticker(ticker)
    px = None; sh = None
    try:
        fast = getattr(y, "fast_info", None) or {}
        px = fast.get("last_price", None)
        if px is None:
            recent = y.history(period="1mo")
            if recent is not None and not recent.empty:
                px = float(recent["Close"].iloc[-1])
    except Exception:
        px = None
    try:
        info = getattr(y, "info", {}) or {}
        sh = info.get("sharesOutstanding", None)
    except Exception:
        sh = None
    return (float(px) if px is not None else None,
            float(sh) if sh is not None else None)

# ===================== PIPELINE (FIXED) =====================

def compute_multi_year_scores_for_ticker_FIXED(ticker: str, headers: dict, min_years: int = MIN_YEARS) -> pd.DataFrame:
    cik_map = load_ticker_cik_map(headers)
    m = cik_map.get(ticker.upper())
    if not m:
        print(f"[{ticker}] No se encontro CIK. Revisa el ticker o el User-Agent.")
        return pd.DataFrame()
    cik10, sic, sic_desc = m[0], (m[1] if len(m) > 1 else None), (m[2] if len(m) > 2 else None)

    facts = get_company_facts(cik10, headers)

    ser_assets = merge_candidate_series(facts, TAGS["Assets"], ("USD",))
    ser_liab   = merge_candidate_series(facts, TAGS["Liabilities"], ("USD",))
    ser_ca     = merge_candidate_series(facts, TAGS["AssetsCurrent"], ("USD",))
    ser_cl     = merge_candidate_series(facts, TAGS["LiabilitiesCurrent"], ("USD",))
    ser_re     = merge_candidate_series(facts, TAGS["RetainedEarnings"], ("USD",))
    ser_ebit   = merge_candidate_series(facts, TAGS["OperatingIncome"], ("USD",))
    ser_rev    = merge_candidate_series(facts, TAGS["Revenue"], ("USD",))
    ser_ni     = merge_candidate_series(facts, TAGS["NetIncome"], ("USD",))
    ser_cfo    = merge_candidate_series(facts, TAGS["CFO"], ("USD",))
    ser_lt     = merge_candidate_series(facts, TAGS["LongTermDebt"], ("USD",))
    ser_sh     = merge_candidate_series(facts, TAGS["SharesOutstanding"], ("shares", "Shares", "pure"))
    ser_gp     = merge_candidate_series(facts, TAGS["GrossProfit"], ("USD",))
    ser_se     = merge_candidate_series(facts, TAGS["StockholdersEquity"], ("USD",))

    def s2map(series: List[dict]) -> Dict[int, Tuple[str, float]]:
        out = {}
        for e in series:
            fy, end, val = e.get("fy"), e.get("end"), to_float_or_none(e.get("val"))
            if not fy or not end:
                continue
            try:
                fy_i = int(fy)
            except Exception:
                continue
            if fy_i not in out or (end > out[fy_i][0]):
                out[fy_i] = (end, val)
        return out

    M_assets = s2map(ser_assets)
    M_liab   = s2map(ser_liab)
    M_ca     = s2map(ser_ca)
    M_cl     = s2map(ser_cl)
    M_re     = s2map(ser_re)
    M_ebit   = s2map(ser_ebit)
    M_rev    = s2map(ser_rev)
    M_ni     = s2map(ser_ni)
    M_cfo    = s2map(ser_cfo)
    M_lt     = s2map(ser_lt)
    M_sh     = s2map(ser_sh)
    M_gp     = s2map(ser_gp)
    M_se     = s2map(ser_se)

    all_fy = sorted(set(M_assets.keys()) | set(M_liab.keys()) | set(M_ni.keys()))
    if len(all_fy) == 0:
        print(f"[{ticker}] Sin FY disponibles. Puede ser rate-limit o falta de tags.")
        return pd.DataFrame()

    rows = []
    px_cache: Dict[str, float] = {}
    px_curr, sh_curr = current_price_and_shares(ticker)
    MVE_CURRENT = float(px_curr * sh_curr) if (px_curr and sh_curr) else None

    for i, fy in enumerate(all_fy):
        end_date = (M_assets.get(fy, (None, None))[0] or
                    M_liab.get(fy, (None, None))[0] or
                    M_ni.get(fy, (None, None))[0])

        TA  = M_assets.get(fy, (None, None))[1]
        TL0 = M_liab.get(fy, (None, None))[1]
        CA  = M_ca.get(fy, (None, None))[1]
        CL  = M_cl.get(fy, (None, None))[1]
        RE  = M_re.get(fy, (None, None))[1]
        EBIT  = M_ebit.get(fy, (None, None))[1]
        SALES = M_rev.get(fy, (None, None))[1]
        NI  = M_ni.get(fy, (None, None))[1]
        CFO = M_cfo.get(fy, (None, None))[1]
        LT  = M_lt.get(fy, (None, None))[1]
        SH_EOP = M_sh.get(fy, (None, None))[1]
        GP  = M_gp.get(fy, (None, None))[1]
        SE  = M_se.get(fy, (None, None))[1]

        # BVE preferente: SE
        if SE is not None:
            BVE = SE
        elif TA is not None and TL0 is not None:
            BVE = TA - TL0
        else:
            BVE = None

        # TL con fallback (descarta LiabilitiesAndStockholdersEquity)
        TL = calculate_total_liabilities(TA, BVE, TL0)

        # valida shares
        if SH_EOP:
            SH_EOP = validate_shares_outstanding(SH_EOP, ticker, fy)

        WC  = (CA - CL) if (CA is not None and CL is not None) else None

        # precio cercano al FY
        px_key = f"{ticker}|{end_date}"
        if px_key in px_cache:
            px_fy = px_cache[px_key]
        else:
            px_fy = price_on_or_near_date(ticker, end_date, window_days=PRICE_WINDOW_DAYS) if end_date else None
            px_cache[px_key] = px_fy

        # proxy shares Yahoo
        _, y_sh_cur = current_price_and_shares(ticker)
        SH_YAHOO = y_sh_cur if (y_sh_cur and y_sh_cur > 0) else None

        # MVE FY
        if SH_EOP and px_fy:
            MVE_FY = float(SH_EOP) * float(px_fy)
        elif SH_EOP and not px_fy and px_curr:
            MVE_FY = float(SH_EOP) * float(px_curr)
        elif SH_YAHOO and px_fy:
            MVE_FY = float(SH_YAHOO) * float(px_fy)
        else:
            MVE_FY = float(SH_YAHOO * px_curr) if (SH_YAHOO and px_curr) else None

        # Valida MVE extremo
        if MVE_FY and MVE_FY > 5_000_000_000_000:  # > $5T
            print(f"  WARNING: MVE anomalo para {ticker} FY{fy}: ${MVE_FY:,.0f}. Se invalida.")
            MVE_FY = None

        # --- Altman FY (con fallback de TL por equity) ---
        Z_orig_FY = compute_altman_with_equity_fallback(WC, TA, RE, EBIT, MVE_FY, TL, BVE, SALES)
        Z_pp_FY   = compute_altman_zpp(WC, TA, RE, EBIT, TL, BVE)

        # --- Altman GF-like ---
        MVE_GF = (MVE_CURRENT if GF_MVE_MODE == "current" else MVE_FY)
        Z_orig_GF = compute_altman_with_equity_fallback(WC, TA, RE, EBIT, MVE_GF, TL, BVE, SALES)

        # --- Piotroski FY y GF-like ---
        if i == 0:
            F_FY = None
            F_GF = None
        else:
            t_vals = {"NI": NI, "TA": TA, "CFO": CFO, "LT": LT, "CA": CA, "CL": CL,
                      "Shares": SH_EOP, "GP": GP, "Sales": SALES}
            t1_vals = {
                "NI": M_ni.get(all_fy[i-1], (None, None))[1],
                "TA": M_assets.get(all_fy[i-1], (None, None))[1],
                "CFO": M_cfo.get(all_fy[i-1], (None, None))[1],
                "LT": M_lt.get(all_fy[i-1], (None, None))[1],
                "CA": M_ca.get(all_fy[i-1], (None, None))[1],
                "CL": M_cl.get(all_fy[i-1], (None, None))[1],
                "Shares": M_sh.get(all_fy[i-1], (None, None))[1],
                "GP": M_gp.get(all_fy[i-1], (None, None))[1],
                "Sales": M_rev.get(all_fy[i-1], (None, None))[1],
            }
            F_FY, _ = compute_piotroski_signals_FY(t_vals, t1_vals)
            curr = {"NI": NI, "CFO": CFO, "Sales": SALES, "GP": GP,
                    "LT": LT, "CA": CA, "CL": CL, "TA": TA, "Shares": SH_EOP}
            prev = {"NI": t1_vals["NI"], "CFO": t1_vals["CFO"], "Sales": t1_vals["Sales"], "GP": t1_vals["GP"],
                    "LT": t1_vals["LT"], "CA": t1_vals["CA"], "CL": t1_vals["CL"], "TA": t1_vals["TA"],
                    "Shares": t1_vals["Shares"], "TA_prev": None}
            F_GF, _ = compute_piotroski_signals_GF_like(curr, prev)

        is_manu = False
        if m[1] is not None:
            try:
                is_manu = (2000 <= int(m[1]) <= 3999)
            except Exception:
                is_manu = False

        rows.append({
            "Ticker": ticker,
            "CIK10": cik10,
            "FiscalYear": fy,
            "FY_EndDate": end_date,
            "TA": TA, "TL": TL, "CA": CA, "CL": CL, "RE": RE,
            "EBIT": EBIT, "Sales": SALES, "NI": NI, "CFO": CFO,
            "LT": LT, "Shares_EOP": SH_EOP, "GP": GP, "SE": SE,
            "WC": WC, "BVE": BVE,
            "Price_Close_near_FYEnd": px_fy,
            "Shares_Yahoo_proxy": SH_YAHOO,
            # FY
            "Altman_Z_original_FY": Z_orig_FY,
            "Altman_Zpp_FY": Z_pp_FY,
            "Piotroski_F_FY": F_FY,
            # GF-like
            "Altman_Z_original_GF": Z_orig_GF,
            "Piotroski_F_GF": F_GF,
            "GF_MVE_Mode": GF_MVE_MODE,
            "GF_MVE_Current": MVE_CURRENT,
            # meta
            "SIC": m[1], "SIC_Desc": m[2], "Is_Manufacturing": is_manu
        })

    df = pd.DataFrame(rows).sort_values(["FiscalYear"]).reset_index(drop=True)

    def coverage(col): return df[col].notna().sum()
    print(f"  Cobertura {ticker}: FY={len(df)} | TA={coverage('TA')} TL={coverage('TL')} "
          f"NI={coverage('NI')} CFO={coverage('CFO')} Sales={coverage('Sales')} EBIT={coverage('EBIT')} "
          f"Shares(EOP)={coverage('Shares_EOP')} | MVE_FY~={'ok' if df['Altman_Z_original_FY'].notna().any() else 'NA'}")

    if len(df["FiscalYear"].unique()) < min_years:
        print(f"  WARNING: {ticker}: menos de {min_years} anos con FY validos ({len(df)})")

    return df

# ===================== DIAGNOSTICO =====================

def diagnose_missing_data(df_all: pd.DataFrame):
    print("\n" + "="*80)
    print("DIAGNOSTICO DE DATOS FALTANTES")
    print("="*80)
    for ticker in df_all['Ticker'].unique():
        df_t = df_all[df_all['Ticker'] == ticker].sort_values("FiscalYear").tail(5)
        missing_cols = []
        for col in ['TA','TL','CA','CL','RE','EBIT','Sales','NI','CFO']:
            if col in df_t.columns and df_t[col].isna().any():
                pct = df_t[col].isna().sum() / len(df_t) * 100
                missing_cols.append(f"{col}({pct:.0f}%)")
        if missing_cols:
            print(f"\n{ticker}:")
            print(f"  Datos faltantes en ultimos 5 anos: {', '.join(missing_cols)}")
            if 'TL' in df_t.columns and df_t['TL'].isna().all():
                print(f"  ERROR: No se puede calcular Altman Z (falta TL)")
                if 'TA' in df_t.columns and not df_t['TA'].isna().all():
                    print(f"  TIP: Sugerencia: usar TL = TA - Stockholders' Equity (SE)")
        if ticker == "QCOM":
            shares_2011 = df_all[(df_all['Ticker']=='QCOM') & (df_all['FiscalYear']==2011)]['Shares_EOP'].values
            if len(shares_2011)>0 and shares_2011[0] is not None and shares_2011[0] > 100_000_000_000:
                print(f"  WARNING: Anomalia detectada: Shares 2011 = {shares_2011[0]:,.0f}")

# ===================== OUTPUTS =====================

def print_last5_for_gf_view(df_all: pd.DataFrame, ticker: str):
    df_t = df_all[df_all["Ticker"] == ticker].copy()
    if df_t.empty:
        return
    df_t = df_t.sort_values("FiscalYear").tail(5).copy()
    show_cols = [
        "Ticker","FiscalYear","FY_EndDate",
        "Altman_Z_original_FY","Altman_Zpp_FY","Piotroski_F_FY",
        "Altman_Z_original_GF","Piotroski_F_GF"
    ]
    for c in ["Altman_Z_original_FY","Altman_Zpp_FY","Altman_Z_original_GF"]:
        if c in df_t.columns:
            df_t[c] = df_t[c].map(lambda x: None if pd.isna(x) else float(f"{x:.4f}"))
    for c in ["Piotroski_F_FY","Piotroski_F_GF"]:
        if c in df_t.columns:
            df_t[c] = df_t[c].map(lambda x: None if pd.isna(x) else int(x))
    print(f"\n=== {ticker} -- ultimos 5 anos (Altman original: FY vs GF-like) ===")
    print(df_t[show_cols].to_string(index=False))

# ===================== MAIN =====================

def main():
    print("="*90)
    print("Altman Z (original & Z'') y Piotroski F -- SEC XBRL + yfinance (FY vs GF-like)")
    print("="*90)
    print(f"User-Agent: {USER_AGENT}")
    print(f"Tickers: {', '.join(TICKERS)}\n")
    print("Metodologias: FY (cierre) y GF-like (Altman original + MVE actual por defecto).\n")

    all_frames = []
    for t in TICKERS:
        try:
            print(f"Descargando y calculando: {t} ...")
            df_t = compute_multi_year_scores_for_ticker_FIXED(t, HEADERS, min_years=MIN_YEARS)
            if df_t is None or df_t.empty:
                print(f"  WARNING: {t}: sin datos devueltos (revisa User-Agent o rate-limit).")
            else:
                out_csv_t = os.path.join(BASE_DIR, f"scores_{t}.csv")
                df_t.to_csv(out_csv_t, index=False)
                print(f"  OK: {t}: {len(df_t)} anos contables. Guardado en {out_csv_t}")
                all_frames.append(df_t.copy())
            time.sleep(PAUSE_SEC)
        except Exception as e:
            print(f"  ERROR: Error procesando {t}: {e}")

    if all_frames:
        df_all = pd.concat(all_frames, ignore_index=True)
        cols_show = [
            "Ticker","FiscalYear","FY_EndDate",
            "Altman_Z_original_FY","Altman_Zpp_FY","Piotroski_F_FY",
            "Altman_Z_original_GF","Piotroski_F_GF","GF_MVE_Mode",
            "TA","TL","SE","BVE","Sales","NI","CFO",
            "Shares_EOP","Shares_Yahoo_proxy","Price_Close_near_FYEnd","GF_MVE_Current"
        ]
        cols_show = [c for c in cols_show if c in df_all.columns]
        df_all_show = df_all[cols_show].copy()
        out_csv_all = os.path.join(BASE_DIR, "scores_altman_piotroski_FY_vs_GF_FIXED.csv")
        df_all.to_csv(out_csv_all, index=False)
        pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

        print("\nResumen consolidado (FY vs GF-like):")
        print(df_all_show.sort_values(["Ticker","FiscalYear"]).to_string(index=False))
        print(f"\nOK: Consolidado exportado: {out_csv_all}")

        # Diagnostico de cobertura y faltantes
        diagnose_missing_data(df_all)

        # Ultimos 5 anos (vista comparativa)
        if GF_COMPARISON_MODE:
            for t in TICKERS:
                print_last5_for_gf_view(df_all, t)
    else:
        print("\nNo se generaron datos consolidados.")

if __name__ == "__main__":
    main()