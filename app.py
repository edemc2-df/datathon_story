from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_PATH = APP_DIR / "model.pkl"
DATA_PATH = ROOT_DIR / "dados_combinados.csv"

FEATURES = ["IDA", "IEG", "IPS", "IPP", "IAN", "IPV"]
FEATURE_LABELS = {
    "IDA": "Desempenho acadêmico",
    "IEG": "Engajamento",
    "IPS": "Psicossocial",
    "IPP": "Psicopedagógico",
    "IAN": "Adequação de nível",
    "IPV": "Ponto de virada",
}
FEATURE_HELP = {
    "IDA": "Qualidade do desempenho acadêmico do aluno.",
    "IEG": "Nível de participação, constância e envolvimento nas atividades.",
    "IPS": "Indicadores psicossociais observados ao longo do acompanhamento.",
    "IPP": "Leitura psicopedagógica do desenvolvimento e do suporte necessário.",
    "IAN": "Aderência do aluno ao nível/fase esperada.",
    "IPV": "Sinal de evolução e mudança positiva na trajetória do estudante.",
}
FEATURE_DESCRIPTIONS = {
    "IDA": "Principal pilar acadêmico do modelo. Quedas aqui tendem a puxar o risco para cima.",
    "IEG": "Representa tração, presença e participação. O notebook mostra que ele caminha junto com o IDA.",
    "IPS": "Tem baixa correlação direta com o risco, mas pode sinalizar efeitos indiretos ou mais tardios.",
    "IPP": "Apoio psicopedagógico aparece como alavanca relevante para evolução e para o IPV.",
    "IAN": "Ajuda a contextualizar o nível do aluno, mas não é o principal fator explicativo do risco.",
    "IPV": "Resume o quanto o aluno está virando a curva. Tem forte relação com INDE, IDA e IEG.",
}
ACTION_LIBRARY = {
    "IDA": (
        "Reforço pedagógico focalizado",
        "Priorizar revisão de lacunas em português e matemática, metas quinzenais e trilhas curtas de recuperação.",
    ),
    "IEG": (
        "Plano de engajamento",
        "Aumentar participação com rotina de acompanhamento, pactos de presença, mentorias e metas de entrega.",
    ),
    "IPS": (
        "Acolhimento e escuta",
        "Monitorar fatores socioemocionais e abrir acompanhamento preventivo para evitar efeito tardio sobre desempenho.",
    ),
    "IPP": (
        "Ação psicopedagógica integrada",
        "Conectar coordenação, psicopedagogia e tutoria para revisar estratégias de estudo e barreiras de aprendizagem.",
    ),
    "IAN": (
        "Adequação de trilha",
        "Rever se a fase, o ritmo e o plano pedagógico estão compatíveis com o momento do aluno.",
    ),
    "IPV": (
        "Consolidação do ponto de virada",
        "Definir marcos de curto prazo, celebrar avanços e manter o aluno em trajetória positiva nas próximas semanas.",
    ),
}
NOTEBOOK_METRICS = {
    "Acurácia": "90,8%",
    "F1-score": "0,91",
    "Recall risco": "92%",
}
DEFAULT_IMPORTANCES = pd.Series(
    {"IDA": 0.33, "IEG": 0.23, "IPV": 0.19, "IPS": 0.08, "IPP": 0.08, "IAN": 0.07}
)
DEFAULT_BENCHMARKS = pd.DataFrame(
    {
        "media": {
            "IDA": 6.376,
            "IEG": 7.946,
            "IPS": 6.287,
            "IPP": 7.555,
            "IAN": 7.179,
            "IPV": 7.545,
        },
        "mediana": {
            "IDA": 6.667,
            "IEG": 8.600,
            "IPS": 7.500,
            "IPP": 7.500,
            "IAN": 5.000,
            "IPV": 7.583,
        },
        "q25": {
            "IDA": 5.100,
            "IEG": 7.300,
            "IPS": 5.020,
            "IPP": 7.083,
            "IAN": 5.000,
            "IPV": 6.984,
        },
    }
)
DEFAULT_YEAR_SUMMARY = pd.DataFrame(
    {
        "INDE": {2022: 7.036, 2023: 7.342, 2024: 7.397},
        "IDA": {2022: 6.093, 2023: 6.663, 2024: 6.351},
        "IEG": {2022: 7.891, 2023: 8.699, 2024: 7.375},
        "IPS": {2022: 6.905, 2023: 5.120, 2024: 6.830},
        "IPP": {2022: None, 2023: 7.563, 2024: 7.548},
        "IAN": {2022: 6.424, 2023: 7.244, 2024: 7.684},
        "IPV": {2022: 7.254, 2023: 8.028, 2024: 7.354},
    }
)
DEFAULT_CORRELATIONS = pd.Series(
    {"IDA": 0.785, "IEG": 0.745, "IPV": 0.721, "IPP": 0.540, "IAN": 0.405, "IPS": 0.200}
)
DEFAULT_SAMPLE_SIZE = 2845


st.set_page_config(
    page_title="Passos Mágicos | Painel Preditivo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink: #333333;
                --muted: #5f6b75;
                --bg-soft: #f5f5f5;
                --surface: rgba(255, 255, 255, 0.94);
                --border: rgba(27, 125, 161, 0.16);
                --brand: #1b7da1;
                --brand-soft: #8bb9da;
                --accent: #fe682d;
                --critical: #fe2e34;
                --warning: #fdc20b;
                --positive: #1b7da1;
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(139, 185, 218, 0.20), transparent 24%),
                    radial-gradient(circle at top right, rgba(253, 194, 11, 0.16), transparent 22%),
                    linear-gradient(180deg, #ffffff 0%, #f5f5f5 100%);
            }

            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
                max-width: 1280px;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(245, 245, 245, 0.93));
                border: 1px solid rgba(27, 125, 161, 0.14);
                border-radius: 28px;
                padding-left: 1.35rem;
                padding-right: 1.35rem;
                box-shadow: 0 20px 45px rgba(27, 125, 161, 0.08);
                backdrop-filter: blur(6px);
            }

            html, body, [class*="css"] {
                font-family: "Trebuchet MS", "Segoe UI", sans-serif;
                color: var(--ink);
            }

            h1, h2, h3 {
                color: var(--ink);
            }

            [data-testid="stHeadingWithActionElements"] h1,
            [data-testid="stHeadingWithActionElements"] h2,
            [data-testid="stHeadingWithActionElements"] h3,
            [data-testid="stHeadingWithActionElements"] p {
                color: var(--ink) !important;
            }

            .block-container [data-testid="stMarkdownContainer"] p,
            .block-container [data-testid="stMarkdownContainer"] li,
            .block-container [data-testid="stMarkdownContainer"] span,
            .block-container label,
            .block-container .stSelectbox label,
            .block-container .stTextInput label,
            .block-container .stSlider label {
                color: var(--ink) !important;
            }

            .block-container [data-testid="stCaptionContainer"] p,
            .block-container .section-caption {
                color: var(--muted) !important;
            }

            .hero {
                background:
                    radial-gradient(circle at top right, rgba(254, 104, 45, 0.24), transparent 26%),
                    linear-gradient(135deg, rgba(27, 125, 161, 0.98), rgba(139, 185, 218, 0.96));
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 24px;
                padding: 1.8rem 2rem;
                margin-bottom: 1rem;
                color: #ffffff;
                box-shadow: 0 20px 45px rgba(27, 125, 161, 0.18);
            }

            .hero-grid {
                display: grid;
                grid-template-columns: 2.2fr 1fr;
                gap: 1rem;
                align-items: stretch;
            }

            .eyebrow {
                display: inline-block;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.14);
                font-size: 0.82rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin-bottom: 0.85rem;
            }

            .hero h1 {
                font-family: Georgia, "Times New Roman", serif;
                font-size: 2.3rem;
                line-height: 1.08;
                margin: 0 0 0.6rem 0;
                color: #ffffff;
            }

            .hero p {
                color: rgba(255, 255, 255, 0.88);
                font-size: 1rem;
                line-height: 1.55;
                margin-bottom: 0;
            }

            .hero-mini {
                display: grid;
                gap: 0.8rem;
            }

            .hero-mini-card {
                padding: 1rem 1.05rem;
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.18);
                border: 1px solid rgba(255, 255, 255, 0.20);
            }

            .hero-mini-card span {
                display: block;
                color: rgba(255, 255, 255, 0.78);
                font-size: 0.8rem;
                margin-bottom: 0.25rem;
            }

            .hero-mini-card strong {
                color: #ffffff;
                font-size: 1.1rem;
            }

            .soft-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 12px 25px rgba(27, 125, 161, 0.06);
            }

            .soft-card h4 {
                margin: 0 0 0.4rem 0;
                font-size: 1rem;
                color: var(--ink);
            }

            .soft-card p {
                margin: 0;
                color: var(--muted);
                line-height: 1.5;
                font-size: 0.93rem;
            }

            .risk-panel {
                border-radius: 22px;
                padding: 1.2rem 1.3rem;
                margin: 0.2rem 0 0.6rem 0;
                border: 1px solid transparent;
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: center;
            }

            .risk-panel h3 {
                margin: 0 0 0.25rem 0;
            }

            .risk-panel p {
                margin: 0;
                color: rgba(51, 51, 51, 0.88);
                max-width: 760px;
            }

            .risk-score {
                font-family: Georgia, "Times New Roman", serif;
                font-size: 2rem;
                font-weight: 700;
                white-space: nowrap;
            }

            .risk-high {
                background: rgba(254, 46, 52, 0.10);
                border-color: rgba(254, 46, 52, 0.22);
            }

            .risk-mid {
                background: rgba(253, 194, 11, 0.16);
                border-color: rgba(253, 194, 11, 0.28);
            }

            .risk-low {
                background: rgba(27, 125, 161, 0.10);
                border-color: rgba(27, 125, 161, 0.24);
            }

            .note-grid {
                display: grid;
                gap: 0.75rem;
            }

            .section-caption {
                color: var(--muted);
                margin-top: -0.2rem;
                margin-bottom: 0.6rem;
            }

            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                box-shadow: 0 12px 24px rgba(27, 125, 161, 0.06);
                min-height: 100%;
            }

            [data-testid="stMetricLabel"] p {
                color: var(--muted) !important;
                font-size: 0.9rem;
                font-weight: 600;
            }

            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] > div,
            [data-testid="stMetricValue"] p {
                color: var(--ink) !important;
                font-family: Georgia, "Times New Roman", serif;
                font-weight: 700;
                font-size: clamp(1.15rem, 1.5vw, 1.7rem) !important;
                line-height: 1.15 !important;
            }

            [data-testid="stMetricDelta"] {
                font-weight: 600;
            }

            [data-baseweb="tab-list"] {
                gap: 0.45rem;
                margin-bottom: 0.8rem;
            }

            [data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.84);
                border-radius: 999px;
                border: 1px solid rgba(139, 185, 218, 0.40);
                padding: 0.45rem 1rem;
                color: var(--ink) !important;
            }

            [data-baseweb="tab"] p,
            [data-baseweb="tab"] span {
                color: var(--ink) !important;
            }

            [data-baseweb="tab"][aria-selected="true"] {
                background: rgba(27, 125, 161, 0.14);
                border-color: rgba(27, 125, 161, 0.42);
                color: var(--brand) !important;
                box-shadow: inset 0 -3px 0 var(--critical);
            }

            [data-baseweb="tab"][aria-selected="true"] p,
            [data-baseweb="tab"][aria-selected="true"] span {
                color: var(--brand) !important;
                font-weight: 700;
            }

            [data-baseweb="tab"]:hover {
                background: rgba(255, 255, 255, 0.96);
            }

            @media (max-width: 980px) {
                .hero-grid {
                    grid-template-columns: 1fr;
                }

                .risk-panel {
                    flex-direction: column;
                    align-items: flex-start;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_decimal(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}".replace(".", ",")


def format_pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%".replace(".", ",")


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def join_labels(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} e {items[-1]}"


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_analytics_base() -> Optional[pd.DataFrame]:
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)

    for column in FEATURES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = pd.NA

    df["INDE"] = pd.NA
    inde_mapping = {2022: "INDE 22", 2023: "INDE 2023", 2024: "INDE 2024"}

    for year, column in inde_mapping.items():
        if column in df.columns:
            mask = df["ANO"] == year
            df.loc[mask, "INDE"] = pd.to_numeric(df.loc[mask, column], errors="coerce")

    df["INDE"] = pd.to_numeric(df["INDE"], errors="coerce")
    return df


def get_feature_importances(model) -> pd.Series:
    try:
        importances = pd.Series(model.feature_importances_, index=FEATURES, dtype="float64")
        return importances.sort_values(ascending=False)
    except Exception:
        return DEFAULT_IMPORTANCES.sort_values(ascending=False)


def get_benchmarks(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return DEFAULT_BENCHMARKS.copy()

    summary = pd.DataFrame(
        {
            "media": df[FEATURES].mean(numeric_only=True),
            "mediana": df[FEATURES].median(numeric_only=True),
            "q25": df[FEATURES].quantile(0.25, numeric_only=True),
        }
    )
    return summary.fillna(DEFAULT_BENCHMARKS)


def get_year_summary(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return DEFAULT_YEAR_SUMMARY.copy()

    columns = ["INDE"] + FEATURES
    summary = df.groupby("ANO")[columns].mean(numeric_only=True).reindex([2022, 2023, 2024])
    return summary.fillna(DEFAULT_YEAR_SUMMARY)


def get_correlations(df: Optional[pd.DataFrame]) -> pd.Series:
    if df is None:
        return DEFAULT_CORRELATIONS.copy()

    corr = df[["INDE"] + FEATURES].corr(numeric_only=True)
    series = corr["INDE"].drop(labels=["INDE"]).sort_values(ascending=False)
    return series.fillna(DEFAULT_CORRELATIONS)


def get_sample_size(df: Optional[pd.DataFrame]) -> int:
    if df is None:
        return DEFAULT_SAMPLE_SIZE
    return int(df["INDE"].notna().sum())


def get_institution_count(df: Optional[pd.DataFrame]) -> Optional[int]:
    if df is None:
        return None

    for column in ["Instituição de ensino", "INSTITUICAO_ENSINO"]:
        if column in df.columns:
            return int(df[column].dropna().nunique())
    return None


def predict_risk(model, input_series: pd.Series) -> Tuple[int, float]:
    payload = pd.DataFrame([[input_series[feature] for feature in FEATURES]], columns=FEATURES)
    prediction = int(model.predict(payload)[0])
    probability = float(model.predict_proba(payload)[0][1])
    return prediction, probability


def classify_risk(probability: float) -> Dict[str, str]:
    if probability >= 0.70:
        return {
            "label": "Prioridade imediata",
            "tone": "risk-high",
            "message": "O aluno apresenta combinação de sinais que pedem atuação rápida da equipe pedagógica e monitoramento próximo.",
        }
    if probability >= 0.40:
        return {
            "label": "Atenção dirigida",
            "tone": "risk-mid",
            "message": "Há sinais relevantes de atenção. O caso pede plano de acompanhamento com foco nos pilares que mais deslocam o risco.",
        }
    return {
        "label": "Estável com monitoramento",
        "tone": "risk-low",
        "message": "O perfil está mais protegido no cenário atual, mas vale acompanhar a consistência dos indicadores ao longo dos próximos ciclos.",
    }


def build_priority_table(
    input_series: pd.Series, benchmarks: pd.DataFrame, importances: pd.Series
) -> pd.DataFrame:
    rows = []

    for feature in FEATURES:
        current = float(input_series[feature])
        median = float(benchmarks.loc[feature, "mediana"])
        q25 = float(benchmarks.loc[feature, "q25"])
        importance = float(importances.get(feature, DEFAULT_IMPORTANCES.get(feature, 0.0)))
        gap = max(median - current, 0.0)
        status = "Crítico" if current < q25 else "Abaixo da mediana" if current < median else "Acima da mediana"

        rows.append(
            {
                "Feature": feature,
                "Indicador": FEATURE_LABELS[feature],
                "Valor atual": current,
                "Mediana da base": median,
                "Gap": gap,
                "Importância": importance,
                "Prioridade": gap * importance,
                "Status": status,
            }
        )

    priority_df = pd.DataFrame(rows).sort_values(["Prioridade", "Gap"], ascending=False)
    return priority_df


def build_strengths_table(input_series: pd.Series, benchmarks: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature in FEATURES:
        current = float(input_series[feature])
        median = float(benchmarks.loc[feature, "mediana"])
        if current >= median:
            rows.append(
                {
                    "Indicador": FEATURE_LABELS[feature],
                    "Valor atual": current,
                    "Mediana da base": median,
                    "Ganho": current - median,
                }
            )

    strengths = pd.DataFrame(rows)
    if strengths.empty:
        return strengths
    return strengths.sort_values("Ganho", ascending=False)


def build_scenarios(
    model, input_series: pd.Series, benchmarks: pd.DataFrame, base_probability: float
) -> Tuple[pd.DataFrame, float]:
    rows = []

    for feature in FEATURES:
        scenario = input_series.copy()
        target = max(float(input_series[feature]), float(benchmarks.loc[feature, "mediana"]))
        scenario[feature] = min(target, 10.0)
        _, new_probability = predict_risk(model, scenario)
        rows.append(
            {
                "Feature": feature,
                "Indicador": FEATURE_LABELS[feature],
                "Valor alvo": scenario[feature],
                "Risco projetado": new_probability,
                "Redução potencial": max(base_probability - new_probability, 0.0),
            }
        )

    scenario_df = pd.DataFrame(rows).sort_values(
        ["Redução potencial", "Risco projetado"], ascending=[False, True]
    )

    priority_features = [
        row["Feature"]
        for _, row in build_priority_table(input_series, benchmarks, DEFAULT_IMPORTANCES).iterrows()
        if row["Gap"] > 0
    ][:3]

    combined = input_series.copy()
    for feature in priority_features:
        combined[feature] = max(float(combined[feature]), float(benchmarks.loc[feature, "mediana"]))

    _, combined_probability = predict_risk(model, combined)
    return scenario_df, combined_probability


def build_recommendations(priority_df: pd.DataFrame) -> List[Tuple[str, str]]:
    relevant = priority_df[priority_df["Gap"] > 0.10].head(3)
    if relevant.empty:
        return [
            (
                "Manter o ritmo e prevenir regressão",
                "O perfil está alinhado ou acima da mediana da base nos principais pilares. O foco deve ser preservar consistência e monitorar pequenas oscilações de engajamento e desempenho.",
            )
        ]

    recommendations = []
    for _, row in relevant.iterrows():
        recommendations.append(ACTION_LIBRARY[row["Feature"]])
    return recommendations


def build_case_summary(
    band: Dict[str, str],
    priority_df: pd.DataFrame,
    strengths_df: pd.DataFrame,
    base_probability: float,
    combined_probability: float,
) -> str:
    main_alerts = priority_df[priority_df["Gap"] > 0.10]["Indicador"].head(2).tolist()
    main_strengths = strengths_df["Indicador"].head(2).tolist() if not strengths_df.empty else []

    sentences = [f"A leitura atual enquadra o caso como {band['label'].lower()}."]

    if main_alerts:
        sentences.append(f"Os maiores desvios aparecem em {join_labels(main_alerts)}.")
    else:
        sentences.append("Os indicadores centrais estão acima da mediana da base, o que reduz a pressão de risco no curto prazo.")

    if main_strengths:
        sentences.append(f"As fortalezas mais claras estão em {join_labels(main_strengths)}.")

    if combined_probability < base_probability:
        sentences.append(
            f"No cenário combinado, o risco projetado cai para {format_pct(combined_probability)}."
        )
    elif combined_probability > base_probability:
        sentences.append(
            f"No cenário combinado, o risco projetado fica em {format_pct(combined_probability)}, indicando que a equipe deve validar a intervenção com contexto pedagógico."
        )
    else:
        sentences.append(
            f"No cenário combinado, o risco projetado permanece em {format_pct(combined_probability)}."
        )
    return " ".join(sentences)


def plot_profile_comparison(input_series: pd.Series, benchmarks: pd.DataFrame):
    labels = [FEATURE_LABELS[feature] for feature in FEATURES]
    base = benchmarks.loc[FEATURES, "mediana"].astype(float)
    current = input_series.loc[FEATURES].astype(float)
    colors = ["#0f766e" if current[feature] >= base[feature] else "#e76f51" for feature in FEATURES]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.barh(labels, base.values, color="#d9e2ec", height=0.58, label="Mediana da base")
    ax.barh(labels, current.values, color=colors, height=0.34, label="Aluno avaliado")

    for idx, value in enumerate(current.values):
        ax.text(min(value + 0.08, 10.1), idx, format_decimal(value, 1), va="center", fontsize=9, color="#14324a")

    ax.set_xlim(0, 10)
    ax.set_xlabel("Escala dos indicadores")
    ax.set_ylabel("")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_year_summary(year_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.7, 4.6))
    colors = {
        "INDE": "#14324a",
        "IDA": "#f08a5d",
        "IEG": "#0f766e",
        "IPV": "#2f5d8a",
        "IAN": "#7b8da6",
    }

    for column in ["INDE", "IDA", "IEG", "IPV", "IAN"]:
        ax.plot(
            year_summary.index,
            year_summary[column],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=column,
            color=colors[column],
        )

    ax.set_xticks(year_summary.index.tolist())
    ax.set_ylim(5, 9)
    ax.set_ylabel("Média do indicador")
    ax.set_xlabel("Ano")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    return fig


def plot_correlation_chart(correlations: pd.Series):
    ordered = correlations.sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    colors = ["#d9e2ec" if value < 0.5 else "#0f766e" for value in ordered.values]
    ax.barh([FEATURE_LABELS[idx] for idx in ordered.index], ordered.values, color=colors)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Correlação com INDE")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)

    for idx, value in enumerate(ordered.values):
        ax.text(value + 0.01, idx, format_decimal(value, 2), va="center", fontsize=9, color="#14324a")

    fig.tight_layout()
    return fig


def plot_importance_chart(importances: pd.Series):
    ordered = importances.sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.barh([FEATURE_LABELS[idx] for idx in ordered.index], ordered.values, color="#f08a5d")
    ax.set_xlabel("Importância relativa")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)

    for idx, value in enumerate(ordered.values):
        ax.text(value + 0.006, idx, format_decimal(value, 2), va="center", fontsize=9, color="#14324a")

    fig.tight_layout()
    return fig


def render_soft_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="soft-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_css()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Não foi possível carregar o modelo treinado em `{MODEL_PATH.name}`. Detalhe técnico: {exc}")
    st.stop()

analytics_base = load_analytics_base()
importances = get_feature_importances(model)
benchmarks = get_benchmarks(analytics_base)
year_summary = get_year_summary(analytics_base)
correlations = get_correlations(analytics_base)
sample_size = get_sample_size(analytics_base)
institution_count = get_institution_count(analytics_base)

defaults = {
    feature: float(round(benchmarks.loc[feature, "mediana"], 1)) if pd.notna(benchmarks.loc[feature, "mediana"]) else 6.0
    for feature in FEATURES
}


st.markdown(
    f"""
    <div class="hero">
        <div class="hero-grid">
            <div>
                <span class="eyebrow">Datathon • Passos Mágicos</span>
                <h1>Painel preditivo para triagem, priorização e ação pedagógica</h1>
                <p>
                    Aplicação desenhada para apoiar a Passos Mágicos na identificação precoce de risco,
                    transformar indicadores em leitura gerencial e orientar intervenções com base em dados.
                    O modelo foi conectado aos achados do notebook analítico e a benchmarks da base PEDE 2022-2024.
                </p>
            </div>
            <div class="hero-mini">
                <div class="hero-mini-card">
                    <span>Base analisada</span>
                    <strong>{format_int(sample_size)} registros úteis</strong>
                </div>
                <div class="hero-mini-card">
                    <span>Desempenho do modelo</span>
                    <strong>{NOTEBOOK_METRICS["Acurácia"]} de acurácia</strong>
                </div>
                <div class="hero-mini-card">
                    <span>Pilares centrais</span>
                    <strong>IDA, IEG e IPV</strong>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
metric_columns[0].metric("Recall da classe de risco", NOTEBOOK_METRICS["Recall risco"])
metric_columns[1].metric(
    "INDE médio em 2024",
    format_decimal(float(year_summary.loc[2024, "INDE"])),
    f"{format_decimal(float(year_summary.loc[2024, 'INDE'] - year_summary.loc[2022, 'INDE']))} vs 2022",
)
metric_columns[2].metric(
    "IEG em 2024",
    format_decimal(float(year_summary.loc[2024, "IEG"])),
    f"{format_decimal(float(year_summary.loc[2024, 'IEG'] - year_summary.loc[2023, 'IEG']))} vs 2023",
)
metric_columns[3].metric(
    "Instituições mapeadas",
    format_int(institution_count) if institution_count is not None else "-",
    "na base consolidada" if institution_count is not None else "benchmark padrão",
)

with st.sidebar:
    st.markdown("### Como usar")
    st.write("1. Ajuste os seis indicadores do aluno.")
    st.write("2. Leia o risco previsto e os fatores que mais puxam o caso.")
    st.write("3. Use as simulações para priorizar a intervenção pedagógica.")
    st.write("4. Navegue pelas abas para ver o contexto analítico da base.")
    st.markdown("### Faixas operacionais")
    st.write("Até 39%: monitoramento.")
    st.write("40% a 69%: atenção dirigida.")
    st.write("70% ou mais: prioridade imediata.")
    st.caption("Os resultados apoiam a decisão pedagógica, mas não substituem avaliação humana.")


tab_triagem, tab_insights, tab_modelo = st.tabs(
    ["Triagem individual", "Insights da base", "Modelo e governança"]
)


with tab_triagem:
    st.subheader("Leitura individual com benchmark da base")
    st.caption(
        "Os controles abaixo atualizam a leitura automaticamente. Use a análise para triagem, reunião de caso ou monitoramento recorrente."
    )

    col_form, col_preview = st.columns([1.05, 0.95], gap="large")

    with col_form:
        st.text_input("Aluno ou identificador", placeholder="Ex.: RA-245 ou nome interno", key="student_name")
        st.selectbox(
            "Contexto de uso",
            ["Triagem inicial", "Monitoramento mensal", "Discussão de caso"],
            key="usage_context",
        )
        left, right = st.columns(2, gap="medium")

        with left:
            ida = st.slider("IDA", 0.0, 10.0, defaults["IDA"], 0.1, help=FEATURE_HELP["IDA"])
            ieg = st.slider("IEG", 0.0, 10.0, defaults["IEG"], 0.1, help=FEATURE_HELP["IEG"])
            ips = st.slider("IPS", 0.0, 10.0, defaults["IPS"], 0.1, help=FEATURE_HELP["IPS"])

        with right:
            ipp = st.slider("IPP", 0.0, 10.0, defaults["IPP"], 0.1, help=FEATURE_HELP["IPP"])
            ian = st.slider("IAN", 0.0, 10.0, defaults["IAN"], 0.1, help=FEATURE_HELP["IAN"])
            ipv = st.slider("IPV", 0.0, 10.0, defaults["IPV"], 0.1, help=FEATURE_HELP["IPV"])

        st.caption("Escala de referência dos indicadores: 0 a 10.")

    input_series = pd.Series(
        {"IDA": ida, "IEG": ieg, "IPS": ips, "IPP": ipp, "IAN": ian, "IPV": ipv}
    )
    prediction, probability = predict_risk(model, input_series)
    band = classify_risk(probability)
    priority_df = build_priority_table(input_series, benchmarks, importances)
    strengths_df = build_strengths_table(input_series, benchmarks)
    scenario_df, combined_probability = build_scenarios(model, input_series, benchmarks, probability)
    recommendations = build_recommendations(priority_df)
    case_summary = build_case_summary(band, priority_df, strengths_df, probability, combined_probability)

    with col_preview:
        benchmark_fig = plot_profile_comparison(input_series, benchmarks)
        st.pyplot(benchmark_fig, use_container_width=True)
        plt.close(benchmark_fig)

        below_median = int((priority_df["Gap"] > 0).sum())
        top_gap = priority_df.iloc[0]
        st.markdown(
            f"""
            <div class="soft-card">
                <h4>Resumo instantâneo</h4>
                <p>
                    {below_median} de 6 indicadores estão abaixo da mediana da base.
                    O maior desvio atual aparece em <strong>{top_gap["Indicador"]}</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="risk-panel {band["tone"]}">
            <div>
                <span class="eyebrow" style="background: rgba(20,50,74,0.08); color: #14324a;">Leitura preditiva</span>
                <h3>{band["label"]}</h3>
                <p>{band["message"]}</p>
            </div>
            <div class="risk-score">{format_pct(probability)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(int(probability * 100))
    result_cols = st.columns(4)
    result_cols[0].metric("Classe prevista", "Risco" if prediction == 1 else "Baixo risco")
    result_cols[1].metric("Abaixo da mediana", f"{below_median}/6")
    result_cols[2].metric(
        "Melhor cenário projetado",
        format_pct(float(scenario_df.iloc[0]["Risco projetado"])),
        f"-{format_pct(float(scenario_df.iloc[0]['Redução potencial']))}",
    )
    result_cols[3].metric(
        "Plano conjunto",
        format_pct(combined_probability),
        f"-{format_pct(max(probability - combined_probability, 0.0))}",
    )

    st.info(case_summary)

    analysis_left, analysis_right = st.columns([1.05, 0.95], gap="large")

    with analysis_left:
        st.markdown("#### Prioridades de intervenção")
        priority_display = priority_df.copy()
        priority_display["Valor atual"] = priority_display["Valor atual"].map(lambda x: format_decimal(x, 1))
        priority_display["Mediana da base"] = priority_display["Mediana da base"].map(lambda x: format_decimal(x, 1))
        priority_display["Gap"] = priority_display["Gap"].map(lambda x: format_decimal(x, 2))
        priority_display["Importância"] = priority_display["Importância"].map(lambda x: format_decimal(x, 2))
        st.dataframe(
            priority_display[
                ["Indicador", "Status", "Valor atual", "Mediana da base", "Gap", "Importância"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Recomendações acionáveis")
        for title, description in recommendations:
            render_soft_card(title, description)

    with analysis_right:
        st.markdown("#### Simulações de melhoria")
        scenario_display = scenario_df.copy()
        scenario_display["Valor alvo"] = scenario_display["Valor alvo"].map(lambda x: format_decimal(x, 1))
        scenario_display["Risco projetado"] = scenario_display["Risco projetado"].map(lambda x: format_pct(x))
        scenario_display["Redução potencial"] = scenario_display["Redução potencial"].map(lambda x: format_pct(x))
        st.dataframe(
            scenario_display[["Indicador", "Valor alvo", "Risco projetado", "Redução potencial"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Fortalezas do perfil")
        if strengths_df.empty:
            render_soft_card(
                "Sem fortalezas evidentes acima da mediana",
                "O caso pede foco em recuperação dos pilares centrais antes de ampliar outras frentes.",
            )
        else:
            for _, row in strengths_df.head(3).iterrows():
                render_soft_card(
                    row["Indicador"],
                    f"Indicador acima da mediana da base em {format_decimal(row['Ganho'], 2)} ponto(s), o que ajuda a amortecer o risco atual.",
                )


with tab_insights:
    st.subheader("Leitura executiva da base")
    st.caption("Resumo do que o notebook aponta sobre evolução, correlação e prioridades de atuação.")

    overview_cols = st.columns(3, gap="medium")
    overview_cols[0].metric(
        "INDE 2022 -> 2024",
        format_decimal(float(year_summary.loc[2024, "INDE"])),
        f"{format_decimal(float(year_summary.loc[2024, 'INDE'] - year_summary.loc[2022, 'INDE']))} no período",
    )
    overview_cols[1].metric(
        "IDA 2024",
        format_decimal(float(year_summary.loc[2024, "IDA"])),
        f"{format_decimal(float(year_summary.loc[2024, 'IDA'] - year_summary.loc[2023, 'IDA']))} vs 2023",
    )
    overview_cols[2].metric(
        "IEG 2024",
        format_decimal(float(year_summary.loc[2024, "IEG"])),
        f"{format_decimal(float(year_summary.loc[2024, 'IEG'] - year_summary.loc[2023, 'IEG']))} vs 2023",
    )

    chart_left, chart_right = st.columns([1.05, 0.95], gap="large")

    with chart_left:
        st.markdown("#### Evolução dos indicadores-chave")
        trend_fig = plot_year_summary(year_summary)
        st.pyplot(trend_fig, use_container_width=True)
        plt.close(trend_fig)

    with chart_right:
        st.markdown("#### O que mais move o INDE")
        corr_fig = plot_correlation_chart(correlations)
        st.pyplot(corr_fig, use_container_width=True)
        plt.close(corr_fig)

    insight_cols = st.columns(3, gap="medium")
    with insight_cols[0]:
        render_soft_card(
            "Evolução positiva do programa",
            f"O INDE sobe de {format_decimal(float(year_summary.loc[2022, 'INDE']))} em 2022 para {format_decimal(float(year_summary.loc[2024, 'INDE']))} em 2024, reforçando a efetividade global da jornada.",
        )
    with insight_cols[1]:
        render_soft_card(
            "Engajamento explica desempenho",
            f"O notebook mostra que IEG e IDA andam juntos. Em 2024, a queda do IEG ajuda a explicar a perda de consistência no desempenho acadêmico.",
        )
    with insight_cols[2]:
        render_soft_card(
            "IPV como ponte de evolução",
            f"O IPV se conecta fortemente a INDE ({format_decimal(float(correlations['IPV']), 2)}) e responde bem a desempenho, engajamento e apoio psicopedagógico.",
        )

    st.markdown("#### Tabela-resumo dos ciclos")
    year_display = year_summary.copy().rename(
        columns={
            "INDE": "INDE",
            "IDA": "IDA",
            "IEG": "IEG",
            "IPS": "IPS",
            "IPP": "IPP",
            "IAN": "IAN",
            "IPV": "IPV",
        }
    )
    st.dataframe(year_display.round(2), use_container_width=True)

    st.markdown("#### Direcionadores estratégicos")
    render_soft_card(
        "Priorizar IDA, IEG e IPV",
        "Esses três pilares concentram a maior parte da explicação do risco. Quando eles melhoram, o INDE tende a responder mais rápido.",
    )
    render_soft_card(
        "Usar IPP como alavanca de recuperação",
        "O suporte psicopedagógico não aparece como fator dominante do risco isolado, mas funciona como acelerador de evolução, especialmente via IPV.",
    )
    render_soft_card(
        "Tratar IPS como radar preventivo",
        "O psicossocial mostra baixa correlação direta com o INDE. Isso sugere uso como radar de prevenção e não como gatilho único de decisão.",
    )

    st.caption(
        "Fonte: notebook `datathon_ed.ipynb`, base consolidada PEDE 2022-2024 e modelo preditivo salvo em `model.pkl`."
    )


with tab_modelo:
    st.subheader("Como o modelo apoia a operação")
    st.caption("Explicação de negócio, leitura das variáveis e recomendações de governança para uso contínuo.")

    model_cols = st.columns(3, gap="medium")
    model_cols[0].metric("Algoritmo", "Random Forest")
    model_cols[1].metric("Variável-alvo", "INDE abaixo da mediana")
    model_cols[2].metric("Features", str(len(FEATURES)))

    governance_left, governance_right = st.columns([1.0, 1.0], gap="large")

    with governance_left:
        st.markdown("#### Importância dos indicadores no modelo")
        importance_fig = plot_importance_chart(importances)
        st.pyplot(importance_fig, use_container_width=True)
        plt.close(importance_fig)

        importances_table = importances.rename(index=FEATURE_LABELS).reset_index()
        importances_table.columns = ["Indicador", "Importância"]
        importances_table["Importância"] = importances_table["Importância"].map(lambda x: format_decimal(x, 2))
        st.dataframe(importances_table, use_container_width=True, hide_index=True)

    with governance_right:
        st.markdown("#### Dicionário operacional das features")
        dictionary_rows = [
            {
                "Indicador": feature,
                "Leitura de negócio": FEATURE_LABELS[feature],
                "Como interpretar": FEATURE_DESCRIPTIONS[feature],
            }
            for feature in FEATURES
        ]
        st.dataframe(pd.DataFrame(dictionary_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Governança recomendada")
    render_soft_card(
        "Usar como triagem e priorização",
        "O modelo é ideal para ordenar casos, apoiar reuniões pedagógicas e concentrar esforço onde a chance de risco é maior.",
    )
    render_soft_card(
        "Recalibrar periodicamente",
        "A base deve ser atualizada a cada novo ciclo e o modelo precisa ser reavaliado para capturar mudanças no perfil dos alunos e no programa.",
    )
    render_soft_card(
        "Combinar previsão com contexto humano",
        "A decisão final deve considerar histórico individual, contexto familiar, escola e evidências qualitativas trazidas pela equipe da Passos Mágicos.",
    )
