from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
WORKBOOK_PATH = ROOT_DIR / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
MODEL_PATH = APP_DIR / "model.pkl"
FEATURES_PATH = APP_DIR / "features.pkl"

PALETTE = {
    "ink": "#333333",
    "muted": "#5F6B75",
    "surface": "#FFFFFF",
    "soft": "#F5F5F5",
    "brand": "#1B7DA1",
    "brand_soft": "#8BB9DA",
    "accent": "#FE682D",
    "critical": "#FE2E34",
    "warning": "#FDC20B",
}

FEATURE_META: Dict[str, Dict[str, str]] = {
    "IDA": {
        "label": "Desempenho acadêmico",
        "help": "Qualidade do desempenho escolar observado pela associação.",
        "reading": "Indica o nível de consistência acadêmica do aluno no ciclo atual.",
        "action": "Acionar reforço focalizado, metas de recuperação e acompanhamento por componente.",
    },
    "IEG": {
        "label": "Engajamento",
        "help": "Participação, presença e adesão do aluno nas atividades.",
        "reading": "Caminha junto com desempenho e ponto de virada.",
        "action": "Priorizar plano de engajamento com rotina, mentoria e combinados de participação.",
    },
    "IPS": {
        "label": "Psicossocial",
        "help": "Termômetro socioemocional e de vulnerabilidades do aluno.",
        "reading": "Funciona como alerta preventivo para quedas futuras.",
        "action": "Ampliar escuta e suporte preventivo para evitar perda de tração no ciclo seguinte.",
    },
    "IPP": {
        "label": "Psicopedagógico",
        "help": "Leitura psicopedagógica do potencial e das barreiras de aprendizagem.",
        "reading": "Foi apontado como principal motor do ponto de virada.",
        "action": "Conectar equipe pedagógica e psicopedagógica para destravar potencial latente.",
    },
    "IAN": {
        "label": "Adequação de nível",
        "help": "Aderência do aluno à fase esperada para a idade.",
        "reading": "Contextualiza defasagem, mas não explica sozinho o potencial do aluno.",
        "action": "Usar para revisar trilha e ritmo, sem reduzir a leitura do caso à série/idade.",
    },
    "IPV": {
        "label": "Ponto de virada",
        "help": "Sinal de transformação e mudança positiva na trajetória.",
        "reading": "Relaciona esforço, estrutura cognitiva e suporte recebido.",
        "action": "Consolidar marcos de curto prazo e garantir continuidade do acompanhamento.",
    },
    "IAA": {
        "label": "Autoavaliação",
        "help": "Percepção do aluno sobre o próprio desempenho.",
        "reading": "Ajuda a entender a visão do aluno, mas pode superestimar a realidade acadêmica.",
        "action": "Usar como insumo de conversa, não como substituto de performance real.",
    },
}

STONE_COLORS = {
    "quartzo": "#FE2E34",
    "agata": "#FE682D",
    "ametista": "#8BB9DA",
    "topazio": "#1B7DA1",
}

NOTEBOOK_MODEL_STORY = {
    "modelo_destacado": "Regressão Logística",
    "auc": "0,85",
    "recall_risco": "76%",
    "tradeoff": "Priorizou alerta precoce e cobertura dos casos em risco.",
}


st.set_page_config(
    page_title="Passos Mágicos | Painel Preditivo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --ink: {PALETTE["ink"]};
                --muted: {PALETTE["muted"]};
                --surface: {PALETTE["surface"]};
                --soft: {PALETTE["soft"]};
                --brand: {PALETTE["brand"]};
                --brand-soft: {PALETTE["brand_soft"]};
                --accent: {PALETTE["accent"]};
                --critical: {PALETTE["critical"]};
                --warning: {PALETTE["warning"]};
            }}

            [data-testid="stAppViewContainer"] {{
                background:
                    radial-gradient(circle at top left, rgba(139, 185, 218, 0.18), transparent 24%),
                    radial-gradient(circle at top right, rgba(253, 194, 11, 0.14), transparent 24%),
                    linear-gradient(180deg, #ffffff 0%, #f5f5f5 100%);
            }}

            .block-container {{
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
            }}

            html, body, [class*="css"] {{
                font-family: "Trebuchet MS", "Segoe UI", sans-serif;
                color: var(--ink);
            }}

            h1, h2, h3 {{
                color: var(--ink);
            }}

            [data-testid="stHeadingWithActionElements"] h1,
            [data-testid="stHeadingWithActionElements"] h2,
            [data-testid="stHeadingWithActionElements"] h3,
            [data-testid="stHeadingWithActionElements"] p {{
                color: var(--ink) !important;
            }}

            .block-container [data-testid="stMarkdownContainer"] p,
            .block-container [data-testid="stMarkdownContainer"] li,
            .block-container [data-testid="stMarkdownContainer"] span,
            .block-container label,
            .block-container .stSelectbox label,
            .block-container .stTextInput label,
            .block-container .stSlider label {{
                color: var(--ink) !important;
            }}

            .block-container [data-testid="stCaptionContainer"] p,
            .block-container .section-caption {{
                color: var(--muted) !important;
            }}

            .hero {{
                background:
                    radial-gradient(circle at top right, rgba(254, 104, 45, 0.24), transparent 26%),
                    linear-gradient(135deg, rgba(27, 125, 161, 0.98), rgba(139, 185, 218, 0.96));
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 24px;
                padding: 1.8rem 2rem;
                margin-bottom: 1rem;
                color: white;
                box-shadow: 0 20px 45px rgba(27, 125, 161, 0.18);
            }}

            .hero-grid {{
                display: grid;
                grid-template-columns: 2.2fr 1fr;
                gap: 1rem;
                align-items: stretch;
            }}

            .eyebrow {{
                display: inline-block;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.14);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.85rem;
            }}

            .hero h1 {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: 2.3rem;
                line-height: 1.08;
                margin: 0 0 0.6rem 0;
                color: white;
            }}

            .hero p {{
                color: rgba(255, 255, 255, 0.88);
                font-size: 1rem;
                line-height: 1.55;
                margin-bottom: 0;
            }}

            .hero-mini {{
                display: grid;
                gap: 0.8rem;
            }}

            .hero-mini-card {{
                padding: 1rem 1.05rem;
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.18);
                border: 1px solid rgba(255, 255, 255, 0.20);
            }}

            .hero-mini-card span {{
                display: block;
                color: rgba(255, 255, 255, 0.78);
                font-size: 0.8rem;
                margin-bottom: 0.25rem;
            }}

            .hero-mini-card strong {{
                color: white;
                font-size: 1.1rem;
            }}

            .soft-card {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 12px 25px rgba(27, 125, 161, 0.06);
            }}

            .soft-card h4 {{
                margin: 0 0 0.4rem 0;
                color: var(--ink);
                font-size: 1rem;
            }}

            .soft-card p {{
                margin: 0;
                color: var(--muted);
                line-height: 1.5;
                font-size: 0.93rem;
            }}

            .risk-panel {{
                border-radius: 22px;
                padding: 1.2rem 1.3rem;
                margin: 0.2rem 0 0.6rem 0;
                border: 1px solid transparent;
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
            }}

            .risk-panel h3 {{
                margin: 0 0 0.25rem 0;
                color: var(--ink);
            }}

            .risk-panel p {{
                margin: 0;
                color: rgba(51,51,51,0.88);
                max-width: 760px;
            }}

            .risk-score {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: 2rem;
                font-weight: 700;
                color: #0E5D78 !important;
                white-space: nowrap;
            }}

            .risk-high {{
                background: rgba(254,46,52,0.10);
                border-color: rgba(254,46,52,0.22);
            }}

            .risk-high .risk-score {{
                color: #B5121A !important;
            }}

            .risk-mid {{
                background: rgba(253,194,11,0.16);
                border-color: rgba(253,194,11,0.28);
            }}

            .risk-mid .risk-score {{
                color: #7A5700 !important;
            }}

            .risk-low {{
                background: rgba(27,125,161,0.10);
                border-color: rgba(27,125,161,0.24);
            }}

            .note-grid {{
                display: grid;
                gap: 0.75rem;
            }}

            .section-caption {{
                color: var(--muted);
                margin-top: -0.2rem;
                margin-bottom: 0.6rem;
            }}

            [data-testid="stMetric"] {{
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 0.85rem 1rem;
                box-shadow: 0 12px 24px rgba(27, 125, 161, 0.06);
                min-height: 100%;
            }}

            [data-testid="stMetricLabel"] p {{
                color: var(--muted) !important;
                font-size: 0.82rem;
                font-weight: 600;
            }}

            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] > div,
            [data-testid="stMetricValue"] p {{
                color: var(--ink) !important;
                font-family: Georgia, "Times New Roman", serif;
                font-weight: 700;
                font-size: clamp(0.98rem, 1.05vw, 1.18rem) !important;
                line-height: 1.18 !important;
                max-width: 100%;
                overflow: visible !important;
                overflow-wrap: anywhere;
                text-overflow: clip !important;
                white-space: normal !important;
            }}

            [data-testid="stMetricDelta"] {{
                font-weight: 600;
            }}

            [data-baseweb="tab-list"] {{
                gap: 0.45rem;
                margin-bottom: 0.8rem;
            }}

            [data-baseweb="tab"] {{
                background: rgba(255, 255, 255, 0.84);
                border-radius: 999px;
                border: 1px solid rgba(139, 185, 218, 0.40);
                padding: 0.45rem 1rem;
                color: var(--ink) !important;
            }}

            [data-baseweb="tab"] p,
            [data-baseweb="tab"] span {{
                color: var(--ink) !important;
            }}

            [data-baseweb="tab"][aria-selected="true"] {{
                background: rgba(27,125,161,0.14);
                border-color: rgba(27,125,161,0.42);
                color: var(--brand) !important;
                box-shadow: inset 0 -3px 0 var(--critical);
            }}

            [data-baseweb="tab"][aria-selected="true"] p,
            [data-baseweb="tab"][aria-selected="true"] span {{
                color: var(--brand) !important;
                font-weight: 700;
            }}

            [data-baseweb="tab"]:hover {{
                background: rgba(255, 255, 255, 0.96);
            }}

            @media (max-width: 980px) {{
                .hero-grid {{
                    grid-template-columns: 1fr;
                }}

                .risk-panel {{
                    flex-direction: column;
                    align-items: flex-start;
                }}
            }}
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


def feature_label(feature: str) -> str:
    return FEATURE_META.get(feature.upper(), {}).get("label", feature.upper())


def feature_help(feature: str) -> str:
    return FEATURE_META.get(feature.upper(), {}).get("help", feature.upper())


def feature_column(feature: str) -> str:
    return feature.lower()


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


def join_labels(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} e {items[-1]}"


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_model_features(_model) -> List[str]:
    raw = [str(item) for item in getattr(_model, "feature_names_in_", [])]
    if raw:
        return raw

    if FEATURES_PATH.exists():
        loaded = joblib.load(FEATURES_PATH)
        return [str(item) for item in loaded]

    raise FileNotFoundError("Não foi possível inferir os indicadores esperados pelo modelo.")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = (
        clean.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    return clean


def normalize_stone(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.lower().str.strip()
    normalized = normalized.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    return normalized.replace("incluir", pd.NA)


def clean_age(value) -> Optional[int]:
    if isinstance(value, datetime.datetime):
        return value.day
    try:
        return int(float(value))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_story_dataframe() -> pd.DataFrame:
    df_2022 = clean_columns(pd.read_excel(WORKBOOK_PATH, sheet_name="PEDE2022"))
    df_2023 = clean_columns(pd.read_excel(WORKBOOK_PATH, sheet_name="PEDE2023"))
    df_2024 = clean_columns(pd.read_excel(WORKBOOK_PATH, sheet_name="PEDE2024"))

    if "destaque_ipv.1" in df_2023.columns:
        df_2023 = df_2023.drop(columns=["destaque_ipv.1"])
    if "ativo/_inativo.1" in df_2024.columns:
        df_2024 = df_2024.drop(columns=["ativo/_inativo.1"])

    if "pedra_23" in df_2023.columns:
        df_2023 = df_2023.drop(columns=["pedra_23"])
    df_2023 = df_2023.rename(columns={"pedra_2023": "pedra_23"})
    df_2024 = df_2024.rename(columns={"pedra_2024": "pedra_24"})

    if "inde_23" in df_2023.columns:
        df_2023 = df_2023.drop(columns=["inde_23"])
    df_2023 = df_2023.rename(columns={"inde_2023": "inde_23"})
    df_2024 = df_2024.rename(columns={"inde_2024": "inde_24"})

    df_2022 = df_2022.rename(
        columns={
            "nome": "nome_anonimizado",
            "portug": "por",
            "defas": "defasagem",
            "ano_nasc": "data_de_nasc",
            "idade_22": "idade",
            "ingles": "ing",
            "matem": "mat",
            "n_av": "no_av",
            "n_av.1": "no_av",
            "no_av": "no_av",
        }
    )

    df_2024["rec_av3"] = pd.NA
    df_2024["rec_av4"] = pd.NA

    df_2022["ano_base"] = 2022
    df_2023["ano_base"] = 2023
    df_2024["ano_base"] = 2024

    df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True, sort=False)

    if "nome_anonimizado" in df.columns:
        df = df.drop(columns=["nome_anonimizado"])

    df["ra"] = pd.to_numeric(
        df["ra"].astype("string").str.replace("RA-", "", regex=False), errors="coerce"
    ).astype("Int64")
    df["fase"] = (
        df["fase"].astype("string").replace("ALFA", "0").str.extract(r"(\d+)").astype("Int64")
    )

    df["turma"] = df["turma"].astype("string").replace("9", "a definir")
    df["turma"] = (
        df["turma"]
        .str.replace("ALFA ", "", regex=False)
        .str.replace(" - G0/G1", "", regex=False)
        .str.replace(" - G2/G3", "", regex=False)
        .str.replace(r"^\d+", "", regex=True)
    )

    df["genero"] = (
        df["genero"].astype("string").str.lower().str.strip().replace({"menino": "masculino", "menina": "feminino"})
    )

    for idx in range(20, 25):
        column = f"pedra_{idx}"
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = normalize_stone(df[column])

    df["data_de_nasc"] = df["data_de_nasc"].apply(
        lambda value: f"01/01/{value}" if isinstance(value, (int, float)) else value
    )
    df["data_de_nasc"] = pd.to_datetime(df["data_de_nasc"], errors="coerce").dt.strftime("%d/%m/%Y")
    df["idade"] = df["idade"].apply(clean_age).astype("Int64")

    for column in ["instituicao_de_ensino", "escola"]:
        if column not in df.columns:
            df[column] = "outros"
        df[column] = df[column].fillna("outros")

    integer_columns = ["ano_ingresso", "cg", "cf", "ct", "no_av", "defasagem", "ano_base"]
    for column in integer_columns:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    numeric_columns = [
        "inde_22",
        "inde_23",
        "inde_24",
        "iaa",
        "ieg",
        "ips",
        "ida",
        "mat",
        "por",
        "ing",
        "ipv",
        "ian",
        "ipp",
    ]
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = pd.NA
        if column == "inde_24":
            df[column] = df[column].replace("INCLUIR", pd.NA)
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Float64")

    df["inde_atual"] = pd.NA
    df.loc[df["ano_base"] == 2022, "inde_atual"] = df.loc[df["ano_base"] == 2022, "inde_22"]
    df.loc[df["ano_base"] == 2023, "inde_atual"] = df.loc[df["ano_base"] == 2023, "inde_23"]
    df.loc[df["ano_base"] == 2024, "inde_atual"] = df.loc[df["ano_base"] == 2024, "inde_24"]
    df["inde_atual"] = pd.to_numeric(df["inde_atual"], errors="coerce").astype("Float64")

    df["pedra_atual"] = pd.NA
    df.loc[df["ano_base"] == 2022, "pedra_atual"] = df.loc[df["ano_base"] == 2022, "pedra_22"]
    df.loc[df["ano_base"] == 2023, "pedra_atual"] = df.loc[df["ano_base"] == 2023, "pedra_23"]
    df.loc[df["ano_base"] == 2024, "pedra_atual"] = df.loc[df["ano_base"] == 2024, "pedra_24"]

    ordered = df.sort_values(by=["ra", "ano_base"]).copy()
    ordered["defasagem_futura"] = ordered.groupby("ra")["defasagem"].shift(-1)
    ordered["risco_futuro"] = pd.NA
    mask_future = ordered["defasagem_futura"].notna()
    ordered.loc[mask_future, "risco_futuro"] = (ordered.loc[mask_future, "defasagem_futura"] > 0).astype(int)
    return ordered


def get_benchmarks(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = {}
    for feature in features:
        column = feature_column(feature)
        if column not in df.columns:
            rows[feature] = {"media": 6.0, "mediana": 6.0, "q25": 5.0}
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        rows[feature] = {
            "media": float(series.mean()) if series.notna().any() else 6.0,
            "mediana": float(series.median()) if series.notna().any() else 6.0,
            "q25": float(series.quantile(0.25)) if series.notna().any() else 5.0,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def get_model_signal(model, features: List[str]) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        raw = pd.Series(model.feature_importances_, index=features, dtype="float64")
    elif hasattr(model, "coef_"):
        coef = model.coef_[0] if getattr(model.coef_, "ndim", 1) > 1 else model.coef_
        raw = pd.Series(coef, index=features, dtype="float64").abs()
    else:
        raw = pd.Series([1.0] * len(features), index=features, dtype="float64")

    total = float(raw.sum())
    if total > 0:
        raw = raw / total
    return raw.sort_values(ascending=False)


def predict_risk(model, features: List[str], values: Dict[str, float]) -> Tuple[int, float]:
    payload = pd.DataFrame([[values[feature] for feature in features]], columns=features)
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
    features: List[str],
    input_values: Dict[str, float],
    benchmarks: pd.DataFrame,
    signal_weights: pd.Series,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        current = float(input_values[feature])
        median = float(benchmarks.loc[feature, "mediana"])
        q25 = float(benchmarks.loc[feature, "q25"])
        weight = float(signal_weights.get(feature, 0.0))
        gap = max(median - current, 0.0)
        status = "Crítico" if current < q25 else "Abaixo da mediana" if current < median else "Acima da mediana"
        rows.append(
            {
                "Feature": feature,
                "Indicador": feature_label(feature),
                "Valor atual": current,
                "Mediana da base": median,
                "Gap": gap,
                "Peso": weight,
                "Prioridade": gap * max(weight, 0.01),
                "Status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(["Prioridade", "Gap"], ascending=False)


def build_strengths_table(
    features: List[str], input_values: Dict[str, float], benchmarks: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for feature in features:
        current = float(input_values[feature])
        median = float(benchmarks.loc[feature, "mediana"])
        if current >= median:
            rows.append(
                {
                    "Indicador": feature_label(feature),
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
    model,
    features: List[str],
    input_values: Dict[str, float],
    benchmarks: pd.DataFrame,
    base_probability: float,
    priority_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    rows = []
    for feature in features:
        scenario = dict(input_values)
        target = max(float(input_values[feature]), float(benchmarks.loc[feature, "mediana"]))
        scenario[feature] = min(target, 10.0)
        _, new_probability = predict_risk(model, features, scenario)
        rows.append(
            {
                "Feature": feature,
                "Indicador": feature_label(feature),
                "Valor alvo": scenario[feature],
                "Risco projetado": new_probability,
                "Redução potencial": max(base_probability - new_probability, 0.0),
            }
        )

    scenario_df = pd.DataFrame(rows).sort_values(
        ["Redução potencial", "Risco projetado"], ascending=[False, True]
    )

    combined = dict(input_values)
    for feature in priority_df.loc[priority_df["Gap"] > 0, "Feature"].head(3):
        combined[feature] = max(float(combined[feature]), float(benchmarks.loc[feature, "mediana"]))
    _, combined_probability = predict_risk(model, features, combined)
    return scenario_df, combined_probability


def build_recommendations(priority_df: pd.DataFrame) -> List[Tuple[str, str]]:
    recommendations: List[Tuple[str, str]] = []
    for feature in priority_df.loc[priority_df["Gap"] > 0.10, "Feature"].head(3):
        meta = FEATURE_META.get(feature, {})
        recommendations.append((feature_label(feature), meta.get("action", "Acompanhar mais de perto o indicador.")))

    if recommendations:
        return recommendations

    return [
        (
            "Manter consistência do caso",
            "Os principais indicadores estão em linha com a mediana da base. O foco aqui é preservar rotina, engajamento e continuidade da evolução.",
        )
    ]


def build_case_summary(
    band: Dict[str, str],
    priority_df: pd.DataFrame,
    strengths_df: pd.DataFrame,
    base_probability: float,
    combined_probability: float,
) -> str:
    alerts = priority_df.loc[priority_df["Gap"] > 0.10, "Indicador"].head(2).tolist()
    strengths = strengths_df["Indicador"].head(2).tolist() if not strengths_df.empty else []

    sentences = [f"O caso foi classificado como {band['label'].lower()}."]
    if alerts:
        sentences.append(f"Os maiores desvios concentram-se em {join_labels(alerts)}.")
    if strengths:
        sentences.append(f"As fortalezas mais claras aparecem em {join_labels(strengths)}.")

    if combined_probability < base_probability:
        sentences.append(
            f"Se os três principais gaps chegarem ao menos até a mediana da base, o risco projetado cai para {format_pct(combined_probability)}."
        )
    else:
        sentences.append(
            f"No cenário combinado, o risco projetado fica em {format_pct(combined_probability)}, então a decisão deve considerar o contexto qualitativo do aluno."
        )
    return " ".join(sentences)


def get_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["inde_atual", "ida", "ieg", "ips", "ipp", "ian", "ipv", "iaa", "defasagem"]
    return df.groupby("ano_base")[cols].mean(numeric_only=True).reindex([2022, 2023, 2024])


def get_feature_correlations(df: pd.DataFrame) -> pd.Series:
    corr = df[["inde_atual", "ida", "ieg", "ips", "ipp", "ian", "ipv", "iaa"]].corr(numeric_only=True)
    return corr["inde_atual"].drop(labels=["inde_atual"]).sort_values(ascending=False)


def get_self_perception_stats(df: pd.DataFrame) -> Dict[str, float]:
    corr = df[["iaa", "ida", "ieg"]].corr(numeric_only=True)
    return {"iaa_ida": float(corr.loc["iaa", "ida"]), "iaa_ieg": float(corr.loc["iaa", "ieg"])}


def get_defasagem_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df.dropna(subset=["defasagem"]).copy()
    rows = []
    for year, group in base.groupby("ano_base"):
        rows.append(
            {
                "ano_base": int(year),
                "severa": float((group["defasagem"] < -1).mean()),
                "leve": float((group["defasagem"] == -1).mean()),
                "ideal": float((group["defasagem"] == 0).mean()),
                "avancado": float((group["defasagem"] > 0).mean()),
                "qtd": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("ano_base")


def get_stone_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df.dropna(subset=["pedra_atual"]).copy()
    summary = (
        base.groupby(["ano_base", "pedra_atual"]).size()
        / base.groupby("ano_base").size()
    ).rename("share").reset_index()
    return summary


def get_high_performance_share(stone_summary: pd.DataFrame) -> Dict[int, float]:
    pivot = stone_summary.pivot(index="ano_base", columns="pedra_atual", values="share").fillna(0.0)
    result = {}
    for year in pivot.index.tolist():
        result[int(year)] = float(pivot.loc[year, ["ametista", "topazio"]].sum())
    return result


def get_early_warning_stats(df: pd.DataFrame) -> pd.DataFrame:
    base = df.dropna(subset=["ips", "risco_futuro"]).copy()
    base["risco_futuro"] = pd.to_numeric(base["risco_futuro"], errors="coerce")
    if base.empty:
        return pd.DataFrame(columns=["faixa_ips", "risco", "qtd"])

    base["faixa_ips"] = pd.qcut(base["ips"], 4, labels=["Q1 mais baixo", "Q2", "Q3", "Q4 mais alto"])
    summary = (
        base.groupby("faixa_ips")
        .agg(risco=("risco_futuro", "mean"), qtd=("risco_futuro", "size"))
        .reset_index()
    )
    return summary


def plot_profile_benchmark(features: List[str], input_values: Dict[str, float], benchmarks: pd.DataFrame):
    labels = [feature_label(feature) for feature in features]
    base = benchmarks.loc[features, "mediana"].astype(float)
    current = pd.Series({feature: input_values[feature] for feature in features}, dtype="float64")
    colors = [PALETTE["brand"] if current[feature] >= base[feature] else PALETTE["accent"] for feature in features]

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.barh(labels, base.values, color="#dfe9f0", height=0.58, label="Mediana da base")
    ax.barh(labels, current.values, color=colors, height=0.34, label="Aluno avaliado")

    for idx, value in enumerate(current.values):
        ax.text(min(value + 0.06, 10.1), idx, format_decimal(value, 1), va="center", fontsize=9, color=PALETTE["ink"])

    ax.set_xlim(0, 10)
    ax.set_xlabel("Escala dos indicadores")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_year_trends(year_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    trend_colors = {
        "inde_atual": PALETTE["brand"],
        "ida": PALETTE["accent"],
        "ieg": "#0F766E",
        "ipv": "#3D5A80",
    }
    labels = {
        "inde_atual": "INDE",
        "ida": "IDA",
        "ieg": "IEG",
        "ipv": "IPV",
    }

    for column in ["inde_atual", "ida", "ieg", "ipv"]:
        ax.plot(
            year_summary.index,
            year_summary[column],
            marker="o",
            linewidth=2.3,
            markersize=7,
            label=labels[column],
            color=trend_colors[column],
        )

    ax.set_xticks(year_summary.index.tolist())
    ax.set_ylim(5, 9)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Media do indicador")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    return fig


def plot_correlations(correlations: pd.Series):
    ordered = correlations.sort_values()
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    colors = [PALETTE["brand_soft"] if value < 0.5 else PALETTE["brand"] for value in ordered.values]
    ax.barh([feature_label(name.upper()) for name in ordered.index], ordered.values, color=colors)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Correlacao com INDE")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    for idx, value in enumerate(ordered.values):
        ax.text(value + 0.01, idx, format_decimal(value, 2), va="center", fontsize=9, color=PALETTE["ink"])
    fig.tight_layout()
    return fig


def plot_stone_shift(stone_summary: pd.DataFrame):
    pivot = (
        stone_summary.pivot(index="ano_base", columns="pedra_atual", values="share")
        .fillna(0.0)
        .reindex(columns=["quartzo", "agata", "ametista", "topazio"])
    )
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    bottom = pd.Series([0.0] * len(pivot.index), index=pivot.index, dtype="float64")

    for stone in pivot.columns:
        ax.bar(
            pivot.index.astype(str),
            pivot[stone].values,
            bottom=bottom.values,
            color=STONE_COLORS.get(stone, "#CCCCCC"),
            label=stone.title(),
        )
        bottom += pivot[stone]

    ax.set_ylim(0, 1)
    ax.set_ylabel("Participação na base")
    ax.set_xlabel("Ano")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    return fig


def plot_defasagem(defasagem_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    width = 0.23
    positions = range(len(defasagem_summary))
    ax.bar([p - width for p in positions], defasagem_summary["severa"], width=width, color=PALETTE["critical"], label="Severa")
    ax.bar(positions, defasagem_summary["ideal"], width=width, color=PALETTE["brand"], label="Fase ideal")
    ax.bar([p + width for p in positions], defasagem_summary["avancado"], width=width, color=PALETTE["warning"], label="Acima da fase")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(defasagem_summary["ano_base"].astype(str).tolist())
    ax.set_ylim(0, max(0.5, float(defasagem_summary[["severa", "ideal", "avancado"]].max().max()) + 0.08))
    ax.set_ylabel("Participação")
    ax.set_xlabel("Ano")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    return fig


def plot_model_signal(signal_weights: pd.Series):
    ordered = signal_weights.sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.barh([feature_label(name) for name in ordered.index], ordered.values, color=PALETTE["accent"])
    ax.set_xlabel("Peso relativo no modelo carregado")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    for idx, value in enumerate(ordered.values):
        ax.text(value + 0.01, idx, format_decimal(value, 2), va="center", fontsize=9, color=PALETTE["ink"])
    fig.tight_layout()
    return fig


inject_css()

missing_files = [path for path in [WORKBOOK_PATH, MODEL_PATH] if not path.exists()]
if missing_files:
    st.error(
        "Não foi possível localizar todos os dados necessários no pacote do app.\n\n"
        + "\n".join(str(path) for path in missing_files)
    )
    st.stop()

try:
    model = load_model()
    model_features = load_model_features(model)
    story_df = load_story_dataframe()
except Exception as exc:
    st.error(f"Falha ao carregar os dados do aplicativo. Detalhe técnico: {exc}")
    st.stop()

signal_weights = get_model_signal(model, model_features)
benchmarks = get_benchmarks(story_df, model_features)
year_summary = get_year_summary(story_df)
correlations = get_feature_correlations(story_df)
defasagem_summary = get_defasagem_summary(story_df)
stone_summary = get_stone_summary(story_df)
high_performance_share = get_high_performance_share(stone_summary)
early_warning_summary = get_early_warning_stats(story_df)
self_perception = get_self_perception_stats(story_df)

defaults = {
    feature: float(round(benchmarks.loc[feature, "mediana"], 1)) if feature in benchmarks.index else 6.0
    for feature in model_features
}

hero_model_name = type(model).__name__
sample_size = int(story_df["inde_atual"].notna().sum())
institution_count = int(story_df["instituicao_de_ensino"].fillna("outros").nunique())

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-grid">
            <div>
                <span class="eyebrow">Datathon | Passos Mágicos</span>
                <h1>Painel preditivo para triagem, priorização e ação pedagógica</h1>
                <p>
                    Aplicação desenhada para apoiar a Passos Mágicos na identificação precoce de risco,
                    transformar indicadores em leitura gerencial e orientar intervenções com base em dados.
                    O modelo combina a base histórica PEDE 2022-2024 com uma leitura preventiva do aluno.
                </p>
            </div>
            <div class="hero-mini">
                <div class="hero-mini-card">
                    <span>Base analisada</span>
                    <strong>{format_int(sample_size)} registros com INDE atual</strong>
                </div>
                <div class="hero-mini-card">
                    <span>Modelo preditivo</span>
                    <strong>{hero_model_name}</strong>
                </div>
                <div class="hero-mini-card">
                    <span>Leitura preventiva</span>
                    <strong>{NOTEBOOK_MODEL_STORY["modelo_destacado"]} como referência analítica</strong>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_metrics = st.columns(4)
top_metrics[0].metric("INDE médio em 2024", format_decimal(float(year_summary.loc[2024, "inde_atual"])))
top_metrics[1].metric(
    "Alto rendimento em 2024",
    format_pct(high_performance_share.get(2024, 0.0)),
    f"{format_pct(high_performance_share.get(2024, 0.0) - high_performance_share.get(2022, 0.0))} vs 2022",
)
top_metrics[2].metric(
    "Defasagem severa em 2024",
    format_pct(float(defasagem_summary.loc[defasagem_summary['ano_base'] == 2024, 'severa'].iloc[0])),
    f"{format_pct(float(defasagem_summary.loc[defasagem_summary['ano_base'] == 2024, 'severa'].iloc[0] - defasagem_summary.loc[defasagem_summary['ano_base'] == 2022, 'severa'].iloc[0]))} vs 2022",
)
top_metrics[3].metric("Instituições mapeadas", format_int(institution_count))

with st.sidebar:
    st.markdown("### Como usar")
    st.write("1. Ajuste os indicadores disponíveis do aluno.")
    st.write("2. Leia o risco previsto e os fatores que mais puxam o caso.")
    st.write("3. Use as simulações para priorizar a intervenção pedagógica.")
    st.write("4. Navegue pelas abas para ver o contexto analítico da base.")
    st.markdown("### Faixas operacionais")
    st.write("Até 39%: monitoramento.")
    st.write("40% a 69%: atenção dirigida.")
    st.write("70% ou mais: prioridade imediata.")
    st.caption("Os resultados apoiam a decisão pedagógica, mas não substituem avaliação humana.")


tab_triagem, tab_story, tab_governanca = st.tabs(
    ["Triagem individual", "Insights da base", "Modelo e governança"]
)


with tab_triagem:
    st.subheader("Leitura individual com foco em alerta precoce")
    st.caption(
        "Os controles abaixo usam o modelo preditivo para apoiar uma interpretação preventiva do caso."
    )

    col_form, col_preview = st.columns([1.05, 0.95], gap="large")

    with col_form:
        st.text_input("Aluno ou identificador", placeholder="Ex.: RA-245 ou código interno", key="story_student")
        st.selectbox(
            "Contexto de uso",
            ["Triagem preventiva", "Reunião de caso", "Monitoramento mensal"],
            key="story_context",
        )

        slider_cols = st.columns(2, gap="medium")
        input_values: Dict[str, float] = {}
        for idx, feature in enumerate(model_features):
            target_col = slider_cols[idx % 2]
            with target_col:
                input_values[feature] = st.slider(
                    feature_label(feature),
                    min_value=0.0,
                    max_value=10.0,
                    value=defaults[feature],
                    step=0.1,
                    help=feature_help(feature),
                )

        st.caption("Os valores iniciais foram preenchidos com a mediana da base analisada.")

    prediction, probability = predict_risk(model, model_features, input_values)
    band = classify_risk(probability)
    priority_df = build_priority_table(model_features, input_values, benchmarks, signal_weights)
    strengths_df = build_strengths_table(model_features, input_values, benchmarks)
    scenario_df, combined_probability = build_scenarios(
        model, model_features, input_values, benchmarks, probability, priority_df
    )
    recommendations = build_recommendations(priority_df)
    case_summary = build_case_summary(band, priority_df, strengths_df, probability, combined_probability)

    with col_preview:
        benchmark_fig = plot_profile_benchmark(model_features, input_values, benchmarks)
        st.pyplot(benchmark_fig, use_container_width=True)
        plt.close(benchmark_fig)

        below_median = int((priority_df["Gap"] > 0).sum())
        strongest_gap = priority_df.iloc[0]["Indicador"]
        render_soft_card(
            "Resumo instantâneo",
            f"{below_median} de {len(model_features)} indicadores estão abaixo da mediana. O maior gap atual aparece em {strongest_gap}.",
        )

    st.markdown(
        f"""
        <div class="risk-panel {band["tone"]}">
            <div>
                <span class="eyebrow" style="background: rgba(27,125,161,0.10); color: #1B7DA1;">Triagem operacional</span>
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
    result_cols[1].metric("Abaixo da mediana", f"{below_median}/{len(model_features)}")
    result_cols[2].metric(
        "Melhor cenário isolado",
        format_pct(float(scenario_df.iloc[0]["Risco projetado"])),
        f"-{format_pct(float(scenario_df.iloc[0]['Redução potencial']))}",
    )
    result_cols[3].metric(
        "Cenário combinado",
        format_pct(combined_probability),
        f"-{format_pct(max(probability - combined_probability, 0.0))}",
    )

    st.info(case_summary)

    analysis_left, analysis_right = st.columns([1.05, 0.95], gap="large")

    with analysis_left:
        st.markdown("#### Prioridades de intervenção")
        priority_display = priority_df.copy()
        for column in ["Valor atual", "Mediana da base", "Gap", "Peso"]:
            priority_display[column] = priority_display[column].map(
                lambda value: format_decimal(value, 2 if column in {"Gap", "Peso"} else 1)
            )
        st.dataframe(
            priority_display[["Indicador", "Status", "Valor atual", "Mediana da base", "Gap", "Peso"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Recomendações acionáveis")
        for title, description in recommendations:
            render_soft_card(title, description)

    with analysis_right:
        st.markdown("#### Simulações de melhoria")
        scenario_display = scenario_df.copy()
        scenario_display["Valor alvo"] = scenario_display["Valor alvo"].map(lambda value: format_decimal(value, 1))
        scenario_display["Risco projetado"] = scenario_display["Risco projetado"].map(lambda value: format_pct(value))
        scenario_display["Redução potencial"] = scenario_display["Redução potencial"].map(lambda value: format_pct(value))
        st.dataframe(
            scenario_display[["Indicador", "Valor alvo", "Risco projetado", "Redução potencial"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Fortalezas do perfil")
        if strengths_df.empty:
            render_soft_card(
                "Sem fortalezas evidentes acima da mediana",
                "O caso pede foco na redução dos gaps mais importantes antes de ampliar outras frentes.",
            )
        else:
            for _, row in strengths_df.head(3).iterrows():
                render_soft_card(
                    row["Indicador"],
                    f"Indicador {format_decimal(row['Ganho'], 2)} ponto(s) acima da mediana da base.",
                )


with tab_story:
    st.subheader("Contexto analítico da base PEDE")
    st.caption("Esta aba resume os principais achados da base para apoiar a leitura executiva.")

    overview_cols = st.columns(4, gap="medium")
    overview_cols[0].metric(
        "AUC do modelo",
        NOTEBOOK_MODEL_STORY["auc"],
        "referência analítica",
    )
    overview_cols[1].metric(
        "Recall da classe de risco",
        NOTEBOOK_MODEL_STORY["recall_risco"],
        "foco em alerta precoce",
    )
    overview_cols[2].metric(
        "IAA x IDA",
        format_decimal(self_perception["iaa_ida"], 2),
        "correlação muito baixa",
    )
    overview_cols[3].metric(
        "IAA x IEG",
        format_decimal(self_perception["iaa_ieg"], 2),
        "otimismo acima da realidade",
    )

    chart_left, chart_right = st.columns([1.05, 0.95], gap="large")
    with chart_left:
        st.markdown("#### Evolução dos indicadores centrais")
        trends_fig = plot_year_trends(year_summary)
        st.pyplot(trends_fig, use_container_width=True)
        plt.close(trends_fig)

    with chart_right:
        st.markdown("#### O que mais move o INDE")
        corr_fig = plot_correlations(correlations)
        st.pyplot(corr_fig, use_container_width=True)
        plt.close(corr_fig)

    chart_bottom_left, chart_bottom_right = st.columns([1.0, 1.0], gap="large")
    with chart_bottom_left:
        st.markdown("#### Mobilidade entre pedras")
        stone_fig = plot_stone_shift(stone_summary)
        st.pyplot(stone_fig, use_container_width=True)
        plt.close(stone_fig)

    with chart_bottom_right:
        st.markdown("#### Defasagem: severa x fase ideal")
        defasagem_fig = plot_defasagem(defasagem_summary)
        st.pyplot(defasagem_fig, use_container_width=True)
        plt.close(defasagem_fig)

    insight_cols = st.columns(3, gap="medium")
    with insight_cols[0]:
        render_soft_card(
            "Defasagem severa cai ano a ano",
            "A base mostra redução proporcional contínua da defasagem severa e crescimento da fase ideal entre 2022 e 2024.",
        )
    with insight_cols[1]:
        render_soft_card(
            "Engajamento sustenta desempenho",
            f"O IEG aparece como tração do aluno. Na base, IEG x IDA = {format_decimal(float(story_df[['ieg', 'ida']].corr(numeric_only=True).iloc[0,1]), 2)}.",
        )
    with insight_cols[2]:
        render_soft_card(
            "Autoavaliação não explica performance",
            f"O aluno tende a se superestimar: IAA x IDA = {format_decimal(self_perception['iaa_ida'], 2)} e IAA x IEG = {format_decimal(self_perception['iaa_ieg'], 2)}.",
        )

    second_row = st.columns(3, gap="medium")
    with second_row[0]:
        render_soft_card(
            "IPS como alerta precoce",
            "O IPS funciona como radar preditivo. Quando ele cai, cresce a chance de perda de desempenho e engajamento no ciclo seguinte.",
        )
    with second_row[1]:
        render_soft_card(
            "IPP revela potencial latente",
            "IPP e IAN não contam a mesma história. O IPP ajuda a capturar capacidade de aprendizagem, mesmo com defasagem formal.",
        )
    with second_row[2]:
        render_soft_card(
            "Excelência precisa de base completa",
            "IDA e IEG puxam o INDE, mas o topo absoluto depende de sustentação emocional e cognitiva via IPS e IPP.",
        )

    st.markdown("#### Sinal de alerta por faixa de IPS")
    early_warning_display = early_warning_summary.copy()
    if not early_warning_display.empty:
        early_warning_display["risco"] = early_warning_display["risco"].map(lambda value: format_pct(value))
        early_warning_display["qtd"] = early_warning_display["qtd"].map(lambda value: format_int(int(value)))
        st.dataframe(early_warning_display, use_container_width=True, hide_index=True)

    st.caption(
        "Leitura sintetizada a partir da base PEDE, combinando contexto analítico e escore operacional."
    )


with tab_governanca:
    st.subheader("Modelo, sinais e governança")
    st.caption(
        "Esta área apresenta os indicadores usados pelo modelo e os cuidados de interpretação para uso pedagógico."
    )

    governance_metrics = st.columns(4, gap="medium")
    governance_metrics[0].metric("Modelo carregado", hero_model_name)
    governance_metrics[1].metric("Indicadores usados", str(len(model_features)))
    governance_metrics[2].metric("Referência analítica", NOTEBOOK_MODEL_STORY["modelo_destacado"])
    governance_metrics[3].metric("Tradeoff analítico", NOTEBOOK_MODEL_STORY["tradeoff"])

    governance_left, governance_right = st.columns([1.0, 1.0], gap="large")

    with governance_left:
        st.markdown("#### Peso relativo dos indicadores")
        signal_fig = plot_model_signal(signal_weights)
        st.pyplot(signal_fig, use_container_width=True)
        plt.close(signal_fig)

        signal_table = signal_weights.rename(index=lambda name: feature_label(name)).reset_index()
        signal_table.columns = ["Indicador", "Peso relativo"]
        signal_table["Peso relativo"] = signal_table["Peso relativo"].map(lambda value: format_decimal(value, 2))
        st.dataframe(signal_table, use_container_width=True, hide_index=True)

    with governance_right:
        st.markdown("#### Dicionário operacional")
        dictionary_rows = []
        for feature in model_features:
            meta = FEATURE_META.get(feature, {})
            dictionary_rows.append(
                {
                    "Indicador": feature_label(feature),
                    "Leitura": meta.get("reading", "Indicador operacional do modelo."),
                    "Ação sugerida": meta.get("action", "Acompanhar o caso com a equipe."),
                }
            )
        st.dataframe(pd.DataFrame(dictionary_rows), use_container_width=True, hide_index=True)

    guidance_cols = st.columns(3, gap="medium")
    with guidance_cols[0]:
        render_soft_card(
            "Foco do modelo",
            "A leitura prioriza alerta precoce e cobertura dos casos em risco, apoiando decisões de acompanhamento pedagógico.",
        )
    with guidance_cols[1]:
        render_soft_card(
            "Uso recomendado",
            "Use o escore para ordenar prioridades, discutir planos de ação e acompanhar evolução dos indicadores.",
        )
    with guidance_cols[2]:
        render_soft_card(
            "Limite de interpretação",
            "O resultado apoia a decisão pedagógica, mas deve ser combinado com histórico, escuta e avaliação humana.",
        )
