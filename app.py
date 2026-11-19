"""Streamlit interface for the two existing paraphrase classifiers."""

import json
from pathlib import Path
from threading import Lock

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OPTIONS = ["TF-IDF Baseline", "Sentence-BERT", "Compare Both Models"]
EXAMPLES = {
    "Learning Python": ("How can I learn Python?", "What is the best way to study Python?"),
    "Rain and cake": ("What causes rain?", "How do I bake a chocolate cake?"),
    "Reversed roles": ("The dog chased the cat.", "The cat chased the dog."),
    "Car and automobile": ("He bought a car.", "He purchased an automobile."),
    "Coffee and negation": ("I like coffee.", "I do not like coffee."),
    "Postponed meeting": ("The meeting was postponed.", "The meeting was moved to a later date."),
}


def validate_inputs(first, second):
    for name, text in (("Sentence 1", first), ("Sentence 2", second)):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Enter text in {name}; spaces alone are not enough.")


def check_artifacts(name, root=ROOT):
    if name == "TF-IDF Baseline":
        paths = ["baseline_model.joblib", "tfidf_vectorizer.joblib", "scalar_scaler.joblib"]
    elif name == "Sentence-BERT":
        paths = ["sbert_classifier.joblib", "sbert_encoder/modules.json",
                 "sbert_encoder/config.json", "sbert_encoder/model.safetensors"]
    else:
        raise ValueError("Choose a supported model.")
    missing = [p for p in paths if not (root / "models" / p).is_file()]
    if missing:
        raise FileNotFoundError("Missing local artifacts: " + ", ".join(missing) +
                                ". Restore the saved project artifacts before analyzing.")


@st.cache_resource(show_spinner=False)
def get_predictor(name):
    """Cache existing model loaders and a lock for shared CPU inference resources."""
    check_artifacts(name)
    if name == "TF-IDF Baseline":
        from src.predict_baseline import load_artifacts, predict_paraphrase
    else:
        from src.predict_sbert import load_artifacts, predict_paraphrase
    load_artifacts()
    return predict_paraphrase, Lock()


def analyze(first, second, selection):
    """No input normalization or prediction caching; call only after submission."""
    validate_inputs(first, second)
    names = OPTIONS[:2] if selection == OPTIONS[2] else [selection]
    results = {}
    for name in names:
        check_artifacts(name)
        predict, lock = get_predictor(name)
        with lock:
            results[name] = predict(first, second)
    return results


def performance():
    return {name: json.loads((ROOT / "results" / filename).read_text())["metrics"]
            for name, filename in zip(OPTIONS[:2], ["baseline_metrics.json", "sbert_metrics.json"])}


def load_example():
    st.session_state.sentence1, st.session_state.sentence2 = EXAMPLES[st.session_state.example]


def result_card(name, result):
    with st.container(border=True):
        st.subheader(name)
        st.markdown(f"### {result['label']}")
        probability = result["paraphrase_probability"]
        st.metric("Paraphrase probability", f"{probability:.2%}")
        st.progress(float(probability), text="Uncalibrated model estimate")
        if "cosine_similarity" in result:
            st.metric("Cosine similarity", f"{result['cosine_similarity']:.4f}")
            st.caption("Embedding similarity alone does not determine semantic equivalence.")


def main():
    st.set_page_config(page_title="Semantic Paraphrase Detection", page_icon="↔", layout="wide")
    st.markdown("""<style>
    .stApp {background: #fff; color: #17243b; color-scheme: light;}
    [data-testid="stHeader"] {background: rgba(255,255,255,.96);}
    .block-container {max-width: 1120px; padding: 3rem 2rem 4rem;}
    h1, h2, h3 {color: #17243b; letter-spacing: -0.035em;}
    h1 {font-size: clamp(2rem, 4vw, 2.8rem); font-weight: 700; padding-bottom: .5rem;}
    h2 {font-size: 1.6rem;}
    h3 {font-size: 1.2rem;}
    [data-testid="stCaptionContainer"] {color: #64748b; line-height: 1.6;}
    .project-label {font-size: .75rem; font-weight: 700; letter-spacing: .14em;
                    color: #2457d6; margin-bottom: .5rem;}
    .project-subtitle {color: #64748b; font-size: 1.05rem; line-height: 1.7;
                       max-width: 760px; margin: 0 0 1.8rem;}
    [data-baseweb="tab-list"] {gap: 1.5rem; border-bottom: 1px solid #e2e8f0;}
    [data-baseweb="tab"] {background: white; color: #64748b; padding: .8rem .2rem;}
    [data-baseweb="tab"][aria-selected="true"] {color: #2457d6; font-weight: 600;}
    [data-baseweb="tab-highlight"] {background-color: #2457d6;}
    [data-baseweb="tab-panel"] {padding-top: 1.6rem;}
    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e2e8f0; border-radius: 12px; background: #fff;
    }
    [data-testid="stForm"] {padding: 1.5rem;}
    [data-testid="stTextArea"] textarea {
        background: #f8fafc; color: #17243b; font-size: 1rem;
        line-height: 1.7; caret-color: #2457d6;
    }
    [data-testid="stTextArea"] textarea::placeholder {color: #7b879a;}
    [data-baseweb="select"] > div {background: #f5f7fb; color: #17243b; border-color: #e2e8f0;}
    [data-testid="stBaseButton-primary"] {
        background: #2457d6; border: 1px solid #2457d6; color: white;
        border-radius: 8px; font-weight: 600; min-height: 44px; padding: .6rem 1.3rem;
    }
    [data-testid="stBaseButton-primary"]:hover {background: #1b45af; border-color: #1b45af;}
    button:focus-visible, textarea:focus-visible {outline: 2px solid #2457d6; outline-offset: 3px;}
    [data-testid="stMetric"] {background: #f5f7fb; padding: 1rem 1.2rem; border-radius: 10px;}
    [data-testid="stMetricValue"] {font-size: 1.8rem; color: #17243b;}
    [data-testid="stTable"] {border: 1px solid #e2e8f0; border-radius: 10px; overflow: auto;}
    [data-testid="stTable"] th {background: #f5f7fb; color: #334155;}
    @media (max-width: 640px) {
        .block-container {padding: 1.5rem 1rem 3rem;}
        [data-testid="stForm"] {padding: 1rem;}
        [data-baseweb="tab-list"] {gap: .75rem;}
    }
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="project-label">NATURAL LANGUAGE PROCESSING</div>', unsafe_allow_html=True)
    st.title("Semantic Paraphrase Detection")
    st.markdown('<p class="project-subtitle">Understanding whether two sentences express the same meaning using Natural Language Processing.</p>', unsafe_allow_html=True)
    analysis_tab, dashboard, about = st.tabs(["Analyze Sentences", "Model Performance", "About the Project"])
    try:
        scores = performance()
    except (OSError, ValueError, KeyError):
        scores = None
        st.warning("Saved evaluation metrics are unavailable. Restore results/baseline_metrics.json and results/sbert_metrics.json to view performance.")
    with analysis_tab:
        st.selectbox("Model", OPTIONS, index=1, key="model")
        with st.expander("Try an example"):
            st.selectbox("Example", list(EXAMPLES), key="example")
            st.button("Load Example", on_click=load_example)
            st.caption("Loading an example fills the inputs. Click Analyze Sentences to run the model.")
        with st.form("sentences"):
            left, right = st.columns(2)
            with left:
                first = st.text_area("Sentence 1", key="sentence1", height=160, placeholder="Enter your first sentence…")
            with right:
                second = st.text_area("Sentence 2", key="sentence2", height=160, placeholder="Enter your second sentence…")
            submitted = st.form_submit_button("Analyze Sentences", type="primary")
        st.caption("Probabilities are uncalibrated model estimates, not guarantees. Sentence-BERT truncates inputs longer than 256 word pieces.")
        if submitted:
            try:
                validate_inputs(first, second)
                with st.spinner("Loading saved models and analyzing your sentences…"):
                    results = analyze(first, second, st.session_state.model)
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception:
                st.error("The saved model could not run. Check that all local model files are intact and the project dependencies match the README. No download or retraining was attempted.")
            else:
                for column, (name, result) in zip(st.columns(len(results)), results.items()):
                    with column:
                        result_card(name, result)
                if len(results) == 2:
                    a, b = results.values()
                    if a["predicted_class"] != b["predicted_class"]:
                        st.warning("The models disagree. Neither prediction is automatically correct; review the meaning of both sentences.")
                    rows = {"Prediction": [r["label"] for r in results.values()],
                            "Paraphrase probability": [f"{r['paraphrase_probability']:.2%}" for r in results.values()]}
                    if scores:
                        for label, key in (("Validation accuracy", "accuracy"), ("Validation F1", "f1")):
                            rows[label] = [f"{scores[n][key]:.2%}" for n in results]
                    st.table(pd.DataFrame(rows, index=list(results)).T)
    with dashboard:
        st.header("Model Performance")
        st.write("Both saved classifiers were evaluated on the same 40,429 labeled QQP validation pairs. Paraphrase is the positive class.")
        if scores:
            table = pd.DataFrame(scores).rename(index={"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score", "macro_f1": "Macro F1-score", "roc_auc": "ROC-AUC"})
            formatted = table.map(lambda value: f"{value:.2%}")
            formatted.loc["ROC-AUC"] = table.loc["ROC-AUC"].map(lambda value: f"{value:.5f}")
            st.table(formatted)
        for filename, caption in (("final_model_comparison.png", "Validation metric comparison"),
                                  ("roc_comparison.png", "ROC curves on identical validation pairs"),
                                  ("confusion_matrix_comparison.png", "Confusion matrices: rows are actual labels, columns are predictions")):
            path = ROOT / "results/figures" / filename
            if path.is_file():
                st.image(str(path), caption=caption, width="stretch")
            else:
                st.info(f"Saved plot unavailable: {filename}")
    with about:
        st.header("About the Project")
        st.write("Semantic paraphrase detection asks whether two sentences express the same meaning. This B.Tech NLP project learns from Quora Question Pairs (QQP).")
        st.markdown("""
        - **TF-IDF baseline:** word unigrams and bigrams, symmetric lexical pair features, and Logistic Regression.
        - **Sentence-BERT:** frozen all-MiniLM-L6-v2 embeddings, symmetric pair features, a fitted scaler, and Logistic Regression.
        - **Evaluation:** a seed-42 stratified sample of 100,000 training pairs; two overlapping training rows removed before sampling; all 40,429 validation pairs retained. The official unlabeled test split supplies no metrics.
        - **Limitations:** learned patterns can miss negation, grammatical roles, and valid paraphrases. Validation is imbalanced, probabilities are uncalibrated, and individual questions are shared across splits. MiniLM may have encountered related Quora data during pretraining; exact exposure is unverified.
        """)
        st.caption("Predictions run locally with saved models. This app does not store sentences or cache prediction inputs. Text remains in the active browser session while you use the app.")


if __name__ == "__main__":
    main()
