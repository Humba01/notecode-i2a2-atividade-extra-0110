import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from fpdf import FPDF
from dotenv import load_dotenv
import google.generativeai as genai

# Carrega chave da OpenAI do .env
load_dotenv()
key=os.getenv("GOOGLE_API_KEY")

def responder_gemini(pergunta, df):
    # Converte o DataFrame para CSV e limita o tamanho
    csv_data = df.head(100).to_csv(index=False)

    prompt = f"""
    Você é um assistente de análise de dados. Use os dados CSV abaixo para responder à pergunta.
    Dados CSV:
    {csv_data}
    
    Pergunta: {pergunta}
    Responda de forma completa, clara e concisa.
    """
    # Chama a API do Gemini
    if not key:
        st.error("⚠️ A chave da API do Google Gemini não está configurada. Por favor, defina a variável de ambiente 'GOOGLE_API_KEY'.")
        return "Chave da API não configurada."
    else:
        # Configure the API key
        genai.configure(api_key=key)
        # Now you can use the library to create models
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text


# ===============================
# Funções auxiliares
# ===============================
def estatisticas_basicas(df):
    return df.describe(include="all")

def detectar_outliers(df):
    num_df = df.select_dtypes(include="number")
    modelo = IsolationForest(contamination=0.01, random_state=42)
    modelo.fit(num_df)
    df_temp = df.copy()
    df_temp["outlier"] = modelo.predict(num_df)
    return df_temp[df_temp["outlier"] == -1]

def correlacoes(df):
    return df.corr()

def gerar_histograma(df, coluna):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[coluna], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribuição da coluna {coluna}")
    return fig

def gerar_heatmap(corr):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Mapa de Correlações")
    return fig

def gerar_conclusoes(memoria):
    conclusoes = []
    for item in memoria:
        p = item["pergunta"].lower()

        if "média" in p:
            conclusoes.append("O agente calculou médias das variáveis, indicando valores típicos dos dados.")
        elif "mediana" in p:
            conclusoes.append("O agente verificou a mediana, mostrando tendência central robusta.")
        elif "desvio" in p or "variância" in p:
            conclusoes.append("O agente avaliou a dispersão, identificando variabilidade entre os dados.")
        elif "outlier" in p:
            conclusoes.append("Foram detectados outliers, que podem influenciar nas análises.")
        elif "correlação" in p:
            conclusoes.append("O agente analisou correlações e encontrou relações entre variáveis.")
        elif "distribuição" in p:
            conclusoes.append("Foram gerados gráficos de distribuição para melhor visualizar os dados.")

    if not conclusoes:
        return "Nenhuma conclusão relevante ainda foi gerada."

    return " | ".join(conclusoes)

def gerar_relatorio(memoria, conclusoes, saida="Agentes Autônomos – Relatório da Atividade Extra.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Relatório de Análise Automática", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=11)
    for item in memoria:
        pdf.multi_cell(0, 10, f"❓ Pergunta: {item['pergunta']}\n💡 Resposta:\n{item['resposta']}\n")
        pdf.ln(5)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(200, 10, "Conclusões do Agente", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, conclusoes)

    pdf.output(saida)

# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="Agente de EDA", layout="wide")
st.title("🤖 Agente Autônomo de Análise de Dados CSV")

# Inicializa memória
if "memoria" not in st.session_state:
    st.session_state.memoria = []
if "conclusoes" not in st.session_state:
    st.session_state.conclusoes = ""
# Upload do CSV
arquivo = st.file_uploader("Carregue seu arquivo CSV", type=["csv"])

if arquivo:
    df = pd.read_csv(arquivo)
    st.success(f"Arquivo carregado: {arquivo.name}")

    st.subheader("📊 Estatísticas Descritivas")
    st.write(estatisticas_basicas(df))

    # ----------------------------
    # Perguntas rápidas (pré-configuradas)
    # ----------------------------
    st.subheader("⚡ Perguntas Rápidas")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    pergunta = None  # inicializa

    with col1:
        if st.button("Qual a média dos dados?"):
            pergunta = "média"
    with col2:
        if st.button("Qual a mediana dos dados?"):
            pergunta = "mediana"
    with col3:
        if st.button("Qual o desvio padrão?"):
            pergunta = "desvio"
    with col4:
        if st.button("Existe correlação entre variáveis?"):
            pergunta = "correlação"
    with col5:
        if st.button("Detectar outliers"):
            pergunta = "outliers"
    with col6:
        if st.button("Mostrar distribuição da coluna Amount"):
            pergunta = "distribuição coluna Amount"

    # ----------------------------
    # Pergunta manual (digitada)
    # ----------------------------
    pergunta_manual = st.text_input("Digite sua pergunta:")
    if pergunta_manual:
        pergunta = pergunta_manual
# ----------------------------
    # Processar pergunta
    # ----------------------------
    if pergunta:
        pergunta_lower = pergunta.lower()
        resposta = None
        figura = None

        if "média" in pergunta_lower:
            resposta = df.mean(numeric_only=True)

        elif "mediana" in pergunta_lower:
            resposta = df.median(numeric_only=True)

        elif "desvio" in pergunta_lower or "variância" in pergunta_lower:
            resposta = df.std(numeric_only=True)

        elif "outlier" in pergunta_lower:
            resposta = detectar_outliers(df)

        elif "correlação" in pergunta_lower:
            resposta = correlacoes(df)
            figura = gerar_heatmap(resposta)

        elif "distribuição" in pergunta_lower:
            col = pergunta.split("coluna")[-1].strip()
            if col in df.columns:
                figura = gerar_histograma(df, col)
            else:
                resposta = "⚠️ Coluna não encontrada."

        else:
            # 👉 Se não reconheceu a pergunta, joga para o Gemini
            resposta = responder_gemini(pergunta, df)

        # Exibe resposta
        if resposta is not None:
            st.write(resposta)
        if figura is not None:
            st.pyplot(figura)

        # Salva na memória
        st.session_state.memoria.append({
            "pergunta": pergunta,
            "resposta": str(resposta)[:2000],
        })

        # Atualiza conclusões automáticas
        st.session_state.conclusoes = gerar_conclusoes(st.session_state.memoria)

    # ----------------------------
    # Histórico de perguntas
    # ----------------------------
    if st.session_state.memoria:
        st.subheader("📝 Histórico de Perguntas e Respostas")
        for item in st.session_state.memoria:
            st.markdown(f"**Pergunta:** {item['pergunta']}")
            st.markdown(f"**Resposta:** {item['resposta'][:500]} ...")

    # ----------------------------
    # Conclusões automáticas
    # ----------------------------
    if st.session_state.conclusoes:
        st.subheader("💡 Conclusões do Agente")
        st.write(st.session_state.conclusoes)
