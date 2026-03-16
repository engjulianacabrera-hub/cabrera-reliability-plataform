import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

st.set_page_config(
    page_title="Reliability AI Industrial Platform",
    layout="wide"
)

# ==============================
# ESTADO
# ==============================
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False


# ==============================
# CSS / VISUAL
# ==============================
st.markdown("""
<style>
:root {
    --white: #ffffff;
    --red1: #3a060c;
    --red2: #7f0d17;
    --red3: #b71520;
    --red4: #e41e2b;
    --card-border: rgba(255,255,255,0.15);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top right, rgba(255,255,255,0.06) 0%, transparent 18%),
        radial-gradient(circle at center, rgba(0,0,0,0.10) 0%, transparent 30%),
        linear-gradient(160deg, var(--red1) 0%, var(--red2) 35%, var(--red3) 68%, var(--red4) 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101010 0%, #1b1b1b 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: var(--white) !important;
}

.block-container,
.block-container p,
.block-container li,
.block-container label,
.block-container span,
.block-container h1,
.block-container h2,
.block-container h3,
.block-container h4,
.block-container h5,
.block-container h6,
div[data-testid="stMarkdownContainer"] p {
    color: var(--white) !important;
}

.hero-wrap {
    background: linear-gradient(135deg, rgba(0,0,0,0.28), rgba(255,255,255,0.04));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px;
    padding: 24px 26px 18px 26px;
    margin-bottom: 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    backdrop-filter: blur(8px);
}

.hero-topline {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #ffe0e3 !important;
    margin-bottom: 8px;
}

.block-title {
    font-size: 34px;
    font-weight: 800;
    color: var(--white) !important;
    margin-bottom: 4px;
}

.hero-subtitle {
    font-size: 15px;
    color: #fff0f1 !important;
    margin-bottom: 12px;
}

.hero-chip-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.hero-chip {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.14);
    color: var(--white) !important;
    padding: 8px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}

.card {
    background: linear-gradient(180deg, rgba(18,18,18,0.76) 0%, rgba(34,34,34,0.58) 100%);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid var(--card-border);
    box-shadow: 0 10px 24px rgba(0,0,0,0.20);
    margin-bottom: 14px;
    backdrop-filter: blur(8px);
}

.metric-title {
    font-size: 13px;
    color: #dddddd !important;
    margin-bottom: 8px;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: var(--white) !important;
    line-height: 1.0;
}

.small-text {
    font-size: 12px;
    color: #efefef !important;
    margin-top: 8px;
}

.section-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #ffe0e3 !important;
    margin-bottom: 8px;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(18,18,18,0.68) 0%, rgba(34,34,34,0.55) 100%);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 14px 16px;
    border-radius: 16px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.14);
}

[data-testid="stMetricLabel"] {
    color: #e1e1e1 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

button[kind="primary"] {
    background: linear-gradient(180deg, #ff3b4b 0%, #e41e2b 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
}

button[kind="secondary"] {
    background: linear-gradient(180deg, #3a3a3a 0%, #262626 100%) !important;
    border: 1px solid #6a6a6a !important;
    color: white !important;
    border-radius: 14px !important;
}

.stButton > button:hover {
    border-color: #ffffff !important;
    color: white !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    border-radius: 12px !important;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stInfo"],
[data-testid="stSuccess"],
[data-testid="stWarning"],
[data-testid="stError"] {
    background: rgba(18,18,18,0.56);
    border: 1px solid rgba(255,255,255,0.10);
}

/* CAMPOS BRANCOS DA LATERAL ESQUERDA */
[data-testid="stTextInput"] input {
    color: black !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #666666 !important;
}

[data-testid="stNumberInput"] input {
    color: black !important;
}

[data-testid="stTextArea"] textarea {
    color: black !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #666666 !important;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# FUNÇÕES
# ==============================
def calc_basic_metrics(ttf):
    mtbf = np.mean(ttf)
    b10_emp = np.percentile(ttf, 10)
    b50_emp = np.percentile(ttf, 50)
    b90_emp = np.percentile(ttf, 90)
    return mtbf, b10_emp, b50_emp, b90_emp


def fit_weibull(ttf):
    shape, loc, scale = weibull_min.fit(ttf, floc=0)
    beta = shape
    eta = scale
    return beta, eta


def build_weibull_curves(beta, eta, horizonte):
    x = np.linspace(1, horizonte, 300)
    r_curve = np.exp(-((x / eta) ** beta))
    f_curve = 1 - r_curve
    lam_curve = (beta / eta) * ((x / eta) ** (beta - 1))
    return x, r_curve, f_curve, lam_curve


def weibull_percentile(eta, beta, q):
    return eta * ((-np.log(1 - q)) ** (1 / beta))


def weibull_interpretation(beta):
    if beta < 1:
        return "Mortalidade infantil / falhas prematuras"
    elif abs(beta - 1) < 0.15:
        return "Taxa de falha aproximadamente constante"
    else:
        return "Desgaste progressivo / envelhecimento"


def probability_plot_data(ttf):
    ttf_sorted = np.sort(ttf)
    n = len(ttf_sorted)
    median_ranks = np.array([(i - 0.3) / (n + 0.4) for i in range(1, n + 1)])
    x_plot = np.log(ttf_sorted)
    y_plot = np.log(-np.log(1 - median_ranks))
    return x_plot, y_plot


def monte_carlo_samples(beta, eta, n_sim):
    return eta * np.random.weibull(beta, n_sim)


def expected_cycle_cost_and_prob(t, beta, eta, custo_prev, custo_corr):
    prob_fail = 1 - np.exp(-((t / eta) ** beta))
    expected_cost = custo_prev * (1 - prob_fail) + custo_corr * prob_fail
    return expected_cost, prob_fail


def expected_cost_rate(t, beta, eta, custo_prev, custo_corr):
    expected_cost, prob_fail = expected_cycle_cost_and_prob(t, beta, eta, custo_prev, custo_corr)
    effective_time = max(t, 1e-9)
    cost_rate = expected_cost / effective_time
    return cost_rate, expected_cost, prob_fail


def optimize_interval(beta, eta, custo_prev, custo_corr, t_min, t_max, n_points=180):
    intervals = np.linspace(t_min, t_max, n_points)
    cost_rates = []
    cycle_costs = []
    probs = []

    for t in intervals:
        rate, cycle_cost, pf = expected_cost_rate(t, beta, eta, custo_prev, custo_corr)
        cost_rates.append(rate)
        cycle_costs.append(cycle_cost)
        probs.append(pf)

    cost_rates = np.array(cost_rates)
    cycle_costs = np.array(cycle_costs)
    probs = np.array(probs)

    best_idx = np.argmin(cost_rates)
    best_t = intervals[best_idx]
    best_rate = cost_rates[best_idx]
    best_cycle_cost = cycle_costs[best_idx]

    return intervals, cost_rates, cycle_costs, probs, best_t, best_rate, best_cycle_cost


def apply_risk_adjustment(best_t, criticidade, ambiente):
    fator = 1 + ((criticidade - 3) * 0.08) + (ambiente * 0.15)
    return round(best_t / fator, 1)


def br_money(value):
    s = f"{value:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def load_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, sheet_name="Base_Falhas")
    else:
        return None

    df.columns = [str(col).strip() for col in df.columns]
    return df


def clean_sap_base(df):
    expected_cols = [
        "Centro",
        "Data Inicio Real",
        "Linha",
        "Tipo de Parada",
        "chave do parada",
        "Conjunto",
        "Componente",
        "Modo de falha",
        "Minutos de paradas"
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    out = df.copy()

    out["Centro"] = out["Centro"].astype(str).str.strip()
    out["Linha"] = out["Linha"].astype(str).str.strip()
    out["chave do parada"] = out["chave do parada"].astype(str).str.strip()

    out["Data Inicio Real"] = pd.to_datetime(out["Data Inicio Real"], errors="coerce", dayfirst=True)
    out["Minutos de paradas"] = pd.to_numeric(out["Minutos de paradas"], errors="coerce")

    out = out.dropna(subset=["Data Inicio Real", "chave do parada"])
    return out


def build_asset_label(row):
    return f"{row['Centro']} | {row['Linha']} | {row['chave do parada']}"


def compute_ttf_from_events(asset_df):
    df = asset_df.sort_values("Data Inicio Real").copy()
    df["TTF_h"] = df["Data Inicio Real"].diff().dt.total_seconds() / 3600.0
    ttf = df["TTF_h"].dropna()
    ttf = ttf[ttf > 0]
    return df, ttf.values.astype(float)


def infer_asset_names(asset_df):
    centro = str(asset_df["Centro"].dropna().iloc[0]) if asset_df["Centro"].notna().any() else "N/D"
    linha = str(asset_df["Linha"].dropna().iloc[0]) if asset_df["Linha"].notna().any() else "N/D"
    tag = str(asset_df["chave do parada"].dropna().iloc[0]) if asset_df["chave do parada"].notna().any() else "N/D"

    componente = ""
    if "Componente" in asset_df.columns and asset_df["Componente"].notna().any():
        componente = str(asset_df["Componente"].dropna().iloc[0]).strip()

    conjunto = ""
    if "Conjunto" in asset_df.columns and asset_df["Conjunto"].notna().any():
        conjunto = str(asset_df["Conjunto"].dropna().iloc[0]).strip()

    equipamento = componente if componente not in ["", "nan", "None"] else tag
    tipo_ativo = conjunto if conjunto not in ["", "nan", "None"] else "N/D"
    area = f"{centro} | {linha}"

    return equipamento, tag, area, tipo_ativo


def analyze_single_asset(ttf, equipamento, tag, area, tipo_ativo, escopo, criticidade, ambiente, custo_prev, custo_corr, nsim):
    mtbf, b10_emp, b50_emp, b90_emp = calc_basic_metrics(ttf)
    beta, eta = fit_weibull(ttf)
    x, r_curve, f_curve, lam_curve = build_weibull_curves(beta, eta, 2000)

    b10_model = weibull_percentile(eta, beta, 0.10)
    comportamento = weibull_interpretation(beta)

    x_plot, y_plot = probability_plot_data(ttf)
    coef = np.polyfit(x_plot, y_plot, 1)
    y_fit = coef[0] * x_plot + coef[1]

    mc_samples = monte_carlo_samples(beta, eta, nsim)
    mc_p50 = np.percentile(mc_samples, 50)

    t_min_model = max(1.0, min(np.min(ttf) * 0.25, np.mean(ttf) * 0.25))
    t_max_model = max(np.max(ttf) * 1.5, np.mean(ttf) * 2.0, 500.0)

    intervals, cost_rates, cycle_costs, probs, best_t, best_rate, best_cycle_cost = optimize_interval(
        beta, eta, custo_prev, custo_corr, t_min_model, t_max_model
    )

    best_t_adj = apply_risk_adjustment(best_t, criticidade, ambiente)
    best_rate_adj, best_cycle_cost_adj, best_pf_adj = expected_cost_rate(
        best_t_adj, beta, eta, custo_prev, custo_corr
    )

    result = {
        "equipamento": equipamento,
        "tag": tag,
        "area": area,
        "tipo_ativo": tipo_ativo,
        "escopo": escopo,
        "beta": beta,
        "eta": eta,
        "mtbf": mtbf,
        "b10_model": b10_model,
        "p50": mc_p50,
        "interval": best_t_adj,
        "cost": best_cycle_cost_adj,
        "cost_rate": best_rate_adj,
        "prob_falha": best_pf_adj,
        "comportamento": comportamento,
        "x": x,
        "r_curve": r_curve,
        "f_curve": f_curve,
        "lam_curve": lam_curve,
        "x_plot": x_plot,
        "y_plot": y_plot,
        "y_fit": y_fit,
        "mc_samples": mc_samples,
        "xs": intervals,
        "cost_rates": cost_rates,
        "cycle_costs": cycle_costs,
        "probs": probs,
        "ttf": ttf
    }
    return result


def analyze_portfolio(df, custo_prev, custo_corr, criticidade, ambiente, nsim):
    rows = []

    grouped = df.groupby(["Centro", "Linha", "chave do parada"], dropna=False)

    for (centro, linha, chave), group in grouped:
        try:
            _, ttf = compute_ttf_from_events(group)

            if len(ttf) < 3:
                continue

            equipamento, tag, area, tipo_ativo = infer_asset_names(group)

            beta, eta = fit_weibull(ttf)
            mtbf, _, _, _ = calc_basic_metrics(ttf)
            b10_model = weibull_percentile(eta, beta, 0.10)

            mc_samples = monte_carlo_samples(beta, eta, nsim)
            mc_p50 = np.percentile(mc_samples, 50)

            t_min_model = max(1.0, min(np.min(ttf) * 0.25, np.mean(ttf) * 0.25))
            t_max_model = max(np.max(ttf) * 1.5, np.mean(ttf) * 2.0, 500.0)

            intervals, cost_rates, cycle_costs, probs, best_t, best_rate, best_cycle_cost = optimize_interval(
                beta, eta, custo_prev, custo_corr, t_min_model, t_max_model
            )

            best_t_adj = apply_risk_adjustment(best_t, criticidade, ambiente)
            best_rate_adj, best_cycle_cost_adj, best_pf_adj = expected_cost_rate(
                best_t_adj, beta, eta, custo_prev, custo_corr
            )

            rows.append({
                "Centro": centro,
                "Linha": linha,
                "TAG": chave,
                "Equipamento": equipamento,
                "Tipo_Ativo": tipo_ativo,
                "Eventos_Parada": len(group),
                "TTFs_Validos": len(ttf),
                "Beta": round(float(beta), 4),
                "Eta_h": round(float(eta), 2),
                "MTTF_h": round(float(mtbf), 2),
                "B10_h": round(float(b10_model), 2),
                "P50_MC_h": round(float(mc_p50), 2),
                "Intervalo_Otimo_h": round(float(best_t_adj), 2),
                "Custo_Ciclo_R$": round(float(best_cycle_cost_adj), 2),
                "Custo_por_h_R$": round(float(best_rate_adj), 4),
                "Prob_Falha": round(float(best_pf_adj), 8)
            })
        except Exception:
            continue

    if len(rows) == 0:
        return pd.DataFrame()

    ranking_df = pd.DataFrame(rows).sort_values(
        by=["Custo_por_h_R$", "B10_h"],
        ascending=[False, True]
    ).reset_index(drop=True)

    ranking_df["Prioridade"] = np.arange(1, len(ranking_df) + 1)

    cols = [
        "Prioridade", "Centro", "Linha", "TAG", "Equipamento", "Tipo_Ativo",
        "Eventos_Parada", "TTFs_Validos", "Beta", "Eta_h", "MTTF_h", "B10_h",
        "P50_MC_h", "Intervalo_Otimo_h", "Custo_Ciclo_R$", "Custo_por_h_R$", "Prob_Falha"
    ]
    return ranking_df[cols]


# ==============================
# CABEÇALHO
# ==============================
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-topline'>Cabrera Reliability</div>
    <div class='block-title'>Reliability AI Industrial Platform</div>
    <div class='hero-subtitle'>Protótipo analítico para confiabilidade, manutenção e gestão de ativos.</div>
    <div class='hero-chip-row'>
        <div class='hero-chip'>Base SAP</div>
        <div class='hero-chip'>TTF Automático</div>
        <div class='hero-chip'>Weibull</div>
        <div class='hero-chip'>Monte Carlo</div>
        <div class='hero-chip'>Portfólio</div>
        <div class='hero-chip'>Custo por hora</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.markdown("<div class='section-label'>Configuração da análise</div>", unsafe_allow_html=True)

    st.header("1) Fonte de dados")
    uploaded_file = st.file_uploader("Importar Excel ou CSV", type=["xlsx", "csv"])
    uploaded_df = load_uploaded_file(uploaded_file)

    sap_df = None
    ttf_preview_text = "Carregue uma base SAP para gerar os TTFs automaticamente."
    equipamento = "N/D"
    tag = "N/D"
    area = "N/D"
    tipo_ativo = "N/D"

    st.header("2) Parâmetros")
    escopo = st.selectbox("Escopo da análise", ["Equipamento", "Sistema", "Componente"])
    criticidade = st.slider("Nível de criticidade (1-5)", 1, 5, 4)
    ambiente = st.slider("Ambiente agressivo (0-1)", 0.0, 1.0, 0.4)

    st.header("3) Custos")
    custo_prev = st.number_input("Custo preventiva", value=2000.0)
    custo_corr = st.number_input("Custo corretiva", value=4000.0)

    st.header("4) Simulação")
    nsim = st.slider("Monte Carlo", 100, 20000, 5000)

    if uploaded_df is not None:
        sap_df = clean_sap_base(uploaded_df)

        st.header("5) Filtros SAP")
        tipos_disponiveis = sorted(sap_df["Tipo de Parada"].dropna().astype(str).unique().tolist())
        tipos_selecionados = st.multiselect(
            "Tipo de Parada",
            tipos_disponiveis,
            default=tipos_disponiveis
        )

        if len(tipos_selecionados) > 0:
            sap_df = sap_df[sap_df["Tipo de Parada"].astype(str).isin(tipos_selecionados)].copy()

        sap_df["Asset_Label"] = sap_df.apply(build_asset_label, axis=1)
        asset_options = sorted(sap_df["Asset_Label"].dropna().unique().tolist())

        if len(asset_options) > 0:
            selected_asset = st.selectbox("Selecionar ativo da base", asset_options)
            asset_df = sap_df[sap_df["Asset_Label"] == selected_asset].copy()

            equipamento, tag, area, tipo_ativo = infer_asset_names(asset_df)
            event_df, ttf_vals = compute_ttf_from_events(asset_df)

            if len(ttf_vals) > 0:
                ttf_preview_text = "\n".join([f"{x:.2f}" for x in ttf_vals[:50]])
            else:
                ttf_preview_text = "Ativo com menos de 2 eventos válidos para cálculo do TTF."

            st.caption("TTF calculado automaticamente pela diferença entre eventos consecutivos de Data Inicio Real.")
        else:
            asset_df = pd.DataFrame()
            ttf_vals = np.array([])

    st.header("6) Prévia do TTF calculado")
    st.text_area("TTF automático (horas)", ttf_preview_text, height=180)

    colb1, colb2 = st.columns(2)
    with colb1:
        executar = st.button("Executar análise completa", type="primary")
    with colb2:
        limpar = st.button("Limpar")

if limpar:
    st.session_state.last_result = None
    st.session_state.analysis_ready = False
    st.rerun()


# ==============================
# PROCESSAR ANÁLISE
# ==============================
if executar:
    if uploaded_df is None:
        st.error("Anexe a base SAP para executar esta versão da plataforma.")
        st.session_state.analysis_ready = False
    elif "asset_df" not in locals() or asset_df.empty:
        st.error("Não foi possível identificar um ativo válido na base.")
        st.session_state.analysis_ready = False
    else:
        _, ttf = compute_ttf_from_events(asset_df)

        if len(ttf) < 3:
            st.error("Esse ativo possui menos de 3 TTFs válidos. Escolha outro ativo ou revise a base.")
            st.session_state.analysis_ready = False
        else:
            try:
                result = analyze_single_asset(
                    ttf=ttf,
                    equipamento=equipamento,
                    tag=tag,
                    area=area,
                    tipo_ativo=tipo_ativo,
                    escopo=escopo,
                    criticidade=criticidade,
                    ambiente=ambiente,
                    custo_prev=custo_prev,
                    custo_corr=custo_corr,
                    nsim=nsim
                )
                st.session_state.last_result = result
                st.session_state.analysis_ready = True

            except Exception as e:
                st.error(f"Erro ao processar a análise: {e}")
                st.session_state.analysis_ready = False


# ==============================
# TABS PRINCIPAIS
# ==============================
tab1, tab2 = st.tabs(["Análise do ativo", "Portfólio"])

with tab1:
    if st.session_state.analysis_ready and st.session_state.last_result is not None:
        r = st.session_state.last_result

        st.markdown("<div class='section-label'>Dashboard executivo</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("""
            <div class='card'>
                <div class='metric-title'>Modelo selecionado</div>
                <div class='metric-value'>Weibull</div>
                <div class='small-text'>Ajuste estatístico real</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='card'>
                <div class='metric-title'>β (Weibull)</div>
                <div class='metric-value'>{r['beta']:.2f}</div>
                <div class='small-text'>{r['comportamento']}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='card'>
                <div class='metric-title'>η (Weibull)</div>
                <div class='metric-value'>{r['eta']:.1f} h</div>
                <div class='small-text'>Vida característica</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class='card'>
                <div class='metric-title'>Intervalo econômico</div>
                <div class='metric-value'>{r['interval']:.1f} h</div>
                <div class='small-text'>Otimizado por custo por hora</div>
            </div>
            """, unsafe_allow_html=True)

        a1, a2, _ = st.columns([1, 1, 2])

        with a1:
            if st.button("Salvar cenário atual"):
                export_result = {
                    "equipamento": r["equipamento"],
                    "tag": r["tag"],
                    "area": r["area"],
                    "tipo_ativo": r["tipo_ativo"],
                    "escopo": r["escopo"],
                    "beta": round(float(r["beta"]), 4),
                    "eta_h": round(float(r["eta"]), 2),
                    "mttf_h": round(float(r["mtbf"]), 2),
                    "b10_model_h": round(float(r["b10_model"]), 2),
                    "mc_p50_h": round(float(r["p50"]), 2),
                    "intervalo_otimo_h": round(float(r["interval"]), 2),
                    "custo_ciclo_r$": round(float(r["cost"]), 2),
                    "custo_por_h_r$": round(float(r["cost_rate"]), 6),
                    "prob_falha": round(float(r["prob_falha"]), 8),
                }
                st.session_state.saved_scenarios.append(export_result)
                st.success("Cenário salvo com sucesso.")

        with a2:
            export_df = pd.DataFrame([{
                "equipamento": r["equipamento"],
                "tag": r["tag"],
                "area": r["area"],
                "tipo_ativo": r["tipo_ativo"],
                "escopo": r["escopo"],
                "beta": round(float(r["beta"]), 4),
                "eta_h": round(float(r["eta"]), 2),
                "mttf_h": round(float(r["mtbf"]), 2),
                "b10_model_h": round(float(r["b10_model"]), 2),
                "mc_p50_h": round(float(r["p50"]), 2),
                "intervalo_otimo_h": round(float(r["interval"]), 2),
                "custo_ciclo_r$": round(float(r["cost"]), 2),
                "custo_por_h_r$": round(float(r["cost_rate"]), 6),
                "prob_falha": round(float(r["prob_falha"]), 8),
            }])
            csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Exportar resultado CSV",
                data=csv_data,
                file_name=f"resultado_{r['tag'].replace(' ', '_')}.csv",
                mime="text/csv"
            )

        st.subheader("Indicadores")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MTTF observado", f"{r['mtbf']:.1f} h")
        k2.metric("B10 modelo", f"{r['b10_model']:.1f} h")
        k3.metric("P50 Monte Carlo", f"{r['p50']:.1f} h")
        k4.metric("Custo por hora", br_money(r["cost_rate"]))

        left, right = st.columns([1, 1.2])

        with left:
            st.subheader("Resumo do cenário")
            df = pd.DataFrame({
                "Campo": [
                    "Equipamento", "TAG", "Área", "Tipo de ativo", "Escopo",
                    "Beta", "Eta (h)", "MTTF (h)", "Intervalo ótimo (h)",
                    "Custo do ciclo", "Custo por hora"
                ],
                "Valor": [
                    r["equipamento"],
                    r["tag"],
                    r["area"],
                    r["tipo_ativo"],
                    r["escopo"],
                    round(float(r["beta"]), 4),
                    round(float(r["eta"]), 2),
                    round(float(r["mtbf"]), 2),
                    round(float(r["interval"]), 2),
                    br_money(r["cost"]),
                    br_money(r["cost_rate"])
                ]
            })
            st.dataframe(df, use_container_width=True)

            st.subheader("TTFs calculados automaticamente")
            df_ttf = pd.DataFrame({"TTF (h)": r["ttf"]})
            st.dataframe(df_ttf, use_container_width=True)

            st.subheader("Leitura executiva")
            st.write(
                f"""
                - O ativo **{r['equipamento']}** na área **{r['area']}** apresentou **β = {r['beta']:.2f}**, indicando **{r['comportamento'].lower()}**.
                - A **vida característica η** estimada foi de **{r['eta']:.1f} h**.
                - O **MTTF observado** calculado automaticamente a partir dos eventos foi **{r['mtbf']:.1f} h**.
                - O **intervalo econômico ajustado** ficou em **{r['interval']:.1f} h**.
                - O **custo do ciclo esperado** nesse cenário é de **{br_money(r['cost'])}**.
                - O **custo esperado por hora** ficou em **{br_money(r['cost_rate'])}**.
                - A **probabilidade de falha** no intervalo recomendado é **{r['prob_falha']:.8f}**.
                """
            )

        with right:
            st.subheader("Curvas Weibull — R(t), F(t) e λ(t)")
            fig1, ax1 = plt.subplots(figsize=(9, 4.2))
            ax1.plot(r["x"], r["r_curve"], label="Confiabilidade R(t)")
            ax1.plot(r["x"], r["f_curve"], label="Probabilidade de Falha F(t)")
            ax1.plot(r["x"], r["lam_curve"], label="Taxa de Falha λ(t)")
            ax1.axvline(r["interval"], linestyle="--", linewidth=1, label="Intervalo sugerido")
            ax1.set_xlabel("Tempo (horas)")
            ax1.set_ylabel("Valor")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

            g1, g2 = st.columns(2)

            with g1:
                st.subheader("Monte Carlo")
                fig2, ax2 = plt.subplots(figsize=(5.2, 3.4))
                ax2.hist(r["mc_samples"], bins=28)
                ax2.set_xlabel("Vida simulada (h)")
                ax2.set_ylabel("Frequência")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)

            with g2:
                st.subheader("Custo por hora x intervalo")
                fig3, ax3 = plt.subplots(figsize=(5.2, 3.4))
                ax3.plot(r["xs"], r["cost_rates"])
                ax3.axvline(r["interval"], linestyle="--", linewidth=1)
                ax3.set_xlabel("Intervalo")
                ax3.set_ylabel("Custo por hora")
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)

            st.subheader("Weibull Probability Plot")
            fig4, ax4 = plt.subplots(figsize=(9, 4.0))
            ax4.scatter(r["x_plot"], r["y_plot"], label="Falhas observadas")
            ax4.plot(r["x_plot"], r["y_fit"], label="Ajuste Weibull")
            ax4.set_xlabel("ln(TTF)")
            ax4.set_ylabel("ln(-ln(1-F))")
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            st.pyplot(fig4)

        st.success("Última análise carregada com sucesso.")
    else:
        st.info("Anexe a base SAP, selecione o ativo e execute a análise.")

with tab2:
    st.markdown("<div class='section-label'>Portfólio de ativos</div>", unsafe_allow_html=True)

    if uploaded_df is not None:
        sap_df_port = clean_sap_base(uploaded_df)

        if "Tipo de Parada" in sap_df_port.columns and len(sap_df_port) > 0:
            portfolio_df = analyze_portfolio(
                df=sap_df_port,
                custo_prev=custo_prev,
                custo_corr=custo_corr,
                criticidade=criticidade,
                ambiente=ambiente,
                nsim=nsim
            )

            if portfolio_df.empty:
                st.warning("Não foi possível gerar o ranking. Verifique se há pelo menos 4 eventos por ativo para gerar 3 TTFs válidos.")
            else:
                st.subheader("Ranking automático do portfólio")
                st.dataframe(portfolio_df, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Ativos analisados", len(portfolio_df))
                c2.metric("Maior custo por hora", br_money(portfolio_df["Custo_por_h_R$"].max()))
                c3.metric("Menor B10", f"{portfolio_df['B10_h'].min():.1f} h")

                st.subheader("Top 10 prioridades por custo por hora")
                top10 = portfolio_df.head(10)

                figp, axp = plt.subplots(figsize=(10, 4.5))
                labels = top10["TAG"].astype(str)
                axp.bar(labels, top10["Custo_por_h_R$"])
                axp.set_ylabel("Custo por hora (R$)")
                axp.set_xlabel("TAG")
                axp.tick_params(axis="x", rotation=45)
                axp.grid(True, alpha=0.3)
                st.pyplot(figp)

                csv_portfolio = portfolio_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Exportar ranking do portfólio CSV",
                    data=csv_portfolio,
                    file_name="ranking_portfolio_sap.csv",
                    mime="text/csv"
                )

                top1 = portfolio_df.iloc[0]
                st.subheader("Leitura executiva do portfólio")
                st.write(
                    f"""
                    - O ativo com maior prioridade atual é **{top1['TAG']}** na área **{top1['Centro']} | {top1['Linha']}**.
                    - O **custo por hora** deste ativo está em **{br_money(top1['Custo_por_h_R$'])}**.
                    - O **B10** calculado foi **{top1['B10_h']:.1f} h**.
                    - O ranking usa como critério principal o **custo por hora**, com apoio do **B10**.
                    """
                )
        else:
            st.info("Base importada sem dados válidos para portfólio.")
    else:
        st.info("Para usar o ranking do portfólio, importe a base SAP.")


# ==============================
# HISTÓRICO
# ==============================
st.markdown("<div class='section-label'>Portfólio de cenários</div>", unsafe_allow_html=True)

if len(st.session_state.saved_scenarios) > 0:
    st.subheader("Histórico de cenários salvos")

    hist_df = pd.DataFrame(st.session_state.saved_scenarios)
    st.dataframe(hist_df, use_container_width=True)

    csv_hist = hist_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Exportar histórico CSV",
        data=csv_hist,
        file_name="historico_cenarios.csv",
        mime="text/csv"
    )
else:
    st.info("Nenhum cenário salvo ainda. Execute uma análise e clique em 'Salvar cenário atual'.")