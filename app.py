import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Configuração da página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Controle de Qualidade - Usinagem",
    page_icon="⚙️",
    layout="centered",
)

st.title("⚙️ Controle de Qualidade de Peças Usinadas")
st.markdown(
    "Insira as medidas da peça para classificá-la em **Aprovada**, "
    "**Retrabalho** ou **Refugo**."
)

# ------------------------------------------------------------------
# Geração dos dados sintéticos
# ------------------------------------------------------------------
@st.cache_data
def gerar_dados():
    X, y = make_classification(
        n_samples=900,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=3,
        class_sep=1.4,
        flip_y=0.02,
        random_state=42,
    )

    peso = 200 + (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min()) * 1000
    espessura = 5 + (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min()) * 30

    df = pd.DataFrame({
        "Peso_gramas": peso,
        "Espessura_mm": espessura,
        "Classe": y,
    })

    mapa = {0: "Aprovada", 1: "Retrabalho", 2: "Refugo"}
    df["Classe"] = df["Classe"].map(mapa)
    return df


df = gerar_dados()

# ------------------------------------------------------------------
# Treinamento do modelo SVM — CORRIGIDO
# ------------------------------------------------------------------
@st.cache_resource
def treinar_modelo(df):
    # 🔑 Conversão explícita para NumPy resolve o bug de ArrowExtensionArray
    X = np.asarray(df[["Peso_gramas", "Espessura_mm"]].values, dtype=float)
    y = np.asarray(df["Classe"].values, dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modelo = SVC(kernel="rbf", C=1.5, gamma="scale")
    modelo.fit(X_train_s, y_train)

    y_pred = modelo.predict(X_test_s)
    cm = confusion_matrix(
        y_test, y_pred, labels=["Aprovada", "Retrabalho", "Refugo"]
    )
    acc = accuracy_score(y_test, y_pred)

    return modelo, scaler, cm, acc


modelo, scaler, cm, acc = treinar_modelo(df)

# ------------------------------------------------------------------
# Interface: sliders
# ------------------------------------------------------------------
st.subheader("🔍 Inspeção da Peça")

col1, col2 = st.columns(2)
with col1:
    peso = st.slider(
        "Peso (gramas)",
        min_value=100.0,
        max_value=1400.0,
        value=700.0,
        step=1.0,
    )
with col2:
    espessura = st.slider(
        "Espessura (mm)",
        min_value=1.0,
        max_value=40.0,
        value=18.0,
        step=0.1,
    )

# ------------------------------------------------------------------
# Botão de avaliação
# ------------------------------------------------------------------
if st.button("✅ Avaliar Peça", use_container_width=True):
    entrada = np.array([[peso, espessura]], dtype=float)
    entrada_s = scaler.transform(entrada)
    predicao = modelo.predict(entrada_s)[0]

    if predicao == "Aprovada":
        st.success("🟢 **APROVADA** — Peça dentro das especificações.")
    elif predicao == "Retrabalho":
        st.warning("🟡 **RETRABALHO** — Peça necessita de ajuste.")
    else:
        st.error("🔴 **REFUGO** — Peça fora das especificações, descarte.")

# ------------------------------------------------------------------
# Matriz de confusão
# ------------------------------------------------------------------
st.divider()
st.subheader("📊 Matriz de Confusão do Modelo SVM")
st.caption(f"Acurácia no conjunto de teste: **{acc*100:.2f}%**")

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Aprovada", "Retrabalho", "Refugo"],
    yticklabels=["Aprovada", "Retrabalho", "Refugo"],
    cbar=False,
    ax=ax,
)
ax.set_xlabel("Predito")
ax.set_ylabel("Real")
plt.tight_layout()
st.pyplot(fig)

with st.expander("ℹ️ Sobre o modelo"):
    st.markdown(
        """
        - **Algoritmo:** Support Vector Machine (kernel RBF)
        - **Features de entrada:** `Peso_gramas` e `Espessura_mm`
        - **Classes:** Aprovada, Retrabalho, Refugo
        - **Dados:** gerados sinteticamente com `sklearn.datasets.make_classification`
        - **Pré-processamento:** `StandardScaler`
        - **Divisão treino/teste:** 70% / 30% (estratificada)
        """
    )
