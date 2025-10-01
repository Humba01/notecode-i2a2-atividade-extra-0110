import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from fpdf import FPDF
from dotenv import load_dotenv
import google.generativeai as genai
import tempfile
import pypandoc

# Carrega chave da OpenAI do .env
# load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

def responder_gemini(pergunta, df):
    data = df.head(100).to_csv(index=False)
    prompt = f"""
    Você é um assistente de análise de dados. Use os dados CSV, XLSX, XLS, ODS ou ODT abaixo para responder à pergunta.
    Dados do arquivo lido:
    {data}
    
    Pergunta: {pergunta}
    Responda de forma completa, clara e concisa. Sem retornar como resposta códigos ou tabelas, apenas texto explicativo. Ou gráficos, se necessário.
    """
    if not key:
        st.error("⚠️ A chave da API do Google Gemini não está configurada. Por favor, defina a variável de ambiente 'GOOGLE_API_KEY'.")
        return "Chave da API não configurada."
    else:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text

# ===============================
# Função para carregar múltiplos formatos
# ===============================
def carregar_arquivo(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "csv":
        return pd.read_csv(uploaded_file)
    elif ext in ["xls", "xlsx"]:
        return pd.read_excel(uploaded_file)
    elif ext == "ods":
        return pd.read_excel(uploaded_file, engine="odf")
    elif ext == "odt":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            pypandoc.convert_text(
                uploaded_file.read().decode("utf-8", errors="ignore"),
                'csv',
                format='odt',
                outputfile=tmp_csv.name,
                extra_args=['--standalone']
            )
            return pd.read_csv(tmp_csv.name)
    else:
        raise ValueError("Formato de arquivo não suportado")

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
    ax.set_title("Mapa de Correlacoes")
    return fig

def gerar_scatterplot(df, col_x, col_y):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax, alpha=0.6)
    ax.set_title(f"Dispersão entre {col_x} e {col_y}")
    return fig

def gerar_crosstab(df, col1, col2):
    return pd.crosstab(df[col1], df[col2])

def gerar_boxplot(df, coluna):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(y=df[coluna], ax=ax)
    ax.set_title(f"Boxplot da coluna {coluna}")
    return fig

def gerar_barplot(df, coluna):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=df[coluna], ax=ax)
    ax.set_title(f"Frequência dos valores da coluna {coluna}")
    return fig

def gerar_lineplot(df, col_x, col_y):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(x=df[col_x], y=df[col_y], ax=ax)
    ax.set_title(f"Evolução de {col_y} em função de {col_x}")
    return fig

def gerar_pairplot(df):
    fig = sns.pairplot(df.select_dtypes(include="number").iloc[:, :5])
    return fig

def gerar_pizza(df, coluna):
    valores = df[coluna].value_counts()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(valores, labels=valores.index, autopct='%1.1f%%')
    ax.set_title(f"Proporção dos valores da coluna {coluna}")
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
        elif "outlier" in p or "anomalia" in p:
            conclusoes.append("Foram detectados outliers, que podem influenciar nas análises.")
        elif "correlação" in p or "correlacao" in p:
            conclusoes.append("O agente analisou correlações e encontrou relações entre variáveis.")
        elif "distribuição" in p or "histograma" in p:
            conclusoes.append("Foram gerados gráficos de distribuição para melhor visualizar os dados.")
        elif "relacionadas" in p or "relações" in p or "dispersão" in p or "scatter" in p:
            conclusoes.append("Foram gerados gráficos de dispersão para explorar relações entre variáveis.")
        elif "tabela cruzada" in p or "crosstab" in p:
            conclusoes.append("Foi gerada uma tabela cruzada para analisar a relação entre variáveis categóricas.")
        elif "heatmap" in p or "mapa" in p:
            conclusoes.append("Foi gerado um mapa de calor para avaliar correlações.")
        elif "boxplot" in p or "caixa" in p:
            conclusoes.append("Foi gerado um boxplot para avaliar a distribuição dos dados.")
        elif "barras" in p or "barplot" in p or "frequência" in p:
            conclusoes.append("Foi gerado um gráfico de barras para visualizar frequências.")
        elif "linha" in p or "tendência" in p or "evolução" in p:
            conclusoes.append("Foi gerado um gráfico de linha para analisar evolução temporal.")
        elif "pairplot" in p or "matriz de dispersão" in p:
            conclusoes.append("Foi gerada uma matriz de dispersão para múltiplas variáveis.")
        elif "pizza" in p or "pie" in p or "proporção" in p:
            conclusoes.append("Foi gerado um gráfico de pizza para visualizar proporções.")
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
st.title("🤖 Agente Autônomo de Análise de Dados")

if "memoria" not in st.session_state:
    st.session_state.memoria = []
if "conclusoes" not in st.session_state:
    st.session_state.conclusoes = ""

arquivo = st.file_uploader("Carregue seu arquivo ● CSV, XLSX, XLS, ODS, ODT", 
                           type=["csv", "xlsx", "xls", "ods", "odt"])

if arquivo:
    try:
        df = carregar_arquivo(arquivo)
        st.success(f"Arquivo carregado: {arquivo.name}")
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        df = None

    if df is not None:
        st.subheader("📊 Estatísticas Descritivas")
        st.write(estatisticas_basicas(df))

    # ----------------------------
    # Perguntas rápidas (pré-configuradas)
    # ----------------------------
    st.subheader("⚡ Perguntas & Informações Rápidas")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    pergunta = None  # inicializa

    with col1:
        if st.button("Qual a média dos dados lidos?"):
            pergunta = "média"
    with col2:
        if st.button("Qual a mediana dos dados lidos?"):
            pergunta = "mediana"
    with col3:
        if st.button("Qual o desvio padrão dos dados lidos?"):
            pergunta = "desvio"
    with col4:
        if st.button("Existe correlação entre variáveis?"):
            pergunta = "correlação"
    with col5:
        if st.button("Quais são os outliers?"):
            pergunta = "outliers"
    with col6:
        if st.button("Qual a distribuição da coluna Amount?"):
            pergunta = "distribuição coluna Amount"

    # ----------------------------
    # Pergunta manual (digitada)
    # ----------------------------
    st.subheader("❔ Pergunta Manual")
    col_input, col_checkbox = st.columns([4, 1])
    st.markdown("<style>div.row-widget.stTextInput > div{align-items: center;}</style>", unsafe_allow_html=True)
    with col_input:
        pergunta_manual = st.text_input("Digite sua pergunta:")
    with col_checkbox:
        usa_graficos = st.checkbox("Usar gráficos na resposta", key="usar_graficos", value=True)

    if pergunta_manual:
        pergunta = pergunta_manual

    if pergunta:
        pergunta_lower = pergunta.lower()
        resposta = None
        figura = None
        if df is None:
            st.error("⚠️ Por favor, carregue um arquivo antes de fazer perguntas.")

        elif "média" in pergunta_lower:
            medias = df.mean(numeric_only=True)
            resposta = "Médias das colunas numéricas:\n" + medias.to_string()
        elif "mediana" in pergunta_lower:
            medianas = df.median(numeric_only=True)
            resposta = "Medianas das colunas numéricas:\n" + medianas.to_string()
        elif "desvio" in pergunta_lower or "variância" in pergunta_lower:
            desvios = df.std(numeric_only=True)
            resposta = "Desvios padrão das colunas numéricas:\n" + desvios.to_string()
        elif "outlier" in pergunta_lower or "anomalia" in pergunta_lower:
            outliers = detectar_outliers(df)
            resposta = f"Foram encontrados {len(outliers)} outliers."
            if len(outliers) > 0:
                resposta += "\nExemplos de outliers:\n" + outliers.head().to_string()
        elif "correlação" in pergunta_lower or "correlacao" in pergunta_lower:
            corr = correlacoes(df)
            resposta = "Matriz de correlações entre variáveis numéricas."
            if usa_graficos:
                figura = gerar_heatmap(corr)
        elif "distribuição" in pergunta_lower or "histograma" in pergunta_lower:
            if "amount" in df.columns.str.lower():
                coluna = df.columns[df.columns.str.lower() == "amount"][0]
            else:
                coluna = df.select_dtypes(include="number").columns[0]
            resposta = f"Histograma da coluna {coluna}."
            if usa_graficos:
                figura = gerar_histograma(df, coluna)
        elif "relacionadas" in pergunta_lower or "relações" in pergunta_lower or "dispersão" in pergunta_lower or "scatter" in pergunta_lower:
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols) >= 2:
                col_x, col_y = num_cols[:2]
                resposta = f"Gráfico de dispersão entre {col_x} e {col_y}."
                if usa_graficos:
                    figura = gerar_scatterplot(df, col_x, col_y)
                    st.session_state.graficos.append(figura)
                    st.session_state.perguntas.append(pergunta)
                    st.session_state.respostas.append(resposta)
        elif "resumo" in pergunta_lower or "sumário" in pergunta_lower:
            resposta = "Estatísticas descritivas do dataset:\n" + estatisticas_basicas(df).to_string()
        elif "tabela cruzada" in pergunta_lower or "crosstab" in pergunta_lower:
            cat_cols = df.select_dtypes(include="object").columns
            if len(cat_cols) >= 2:
                col1, col2 = cat_cols[:2]
                ctab = gerar_crosstab(df, col1, col2)
                resposta = f"Tabela cruzada entre {col1} e {col2}:\n" + ctab.to_string()
        elif "heatmap" in pergunta_lower or "mapa" in pergunta_lower:
            corr = correlacoes(df)
            resposta = "Mapa de calor das correlações."
            if usa_graficos:
                figura = gerar_heatmap(corr)
        elif "boxplot" in pergunta_lower or "caixa" in pergunta_lower:
            if "amount" in df.columns.str.lower():
                coluna = df.columns[df.columns.str.lower() == "amount"][0]
            else:
                coluna = df.select_dtypes(include="number").columns[0]
            resposta = f"Boxplot da coluna {coluna}."
            if usa_graficos:
                figura = gerar_boxplot(df, coluna)
        elif "barras" in pergunta_lower or "barplot" in pergunta_lower or "frequência" in pergunta_lower:
            cat_cols = df.select_dtypes(include="object").columns
            if len(cat_cols) >= 2:
                col1, col2 = cat_cols[:2]
                resposta = f"Gráfico de barras entre {col1} e {col2}."
                if usa_graficos:
                    figura = gerar_barplot(df, col1, col2)
        elif "linha" in pergunta_lower or "tendência" in pergunta_lower or "evolução" in pergunta_lower:
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols) >= 2:
                col_x, col_y = num_cols[:2]
                resposta = f"Gráfico de linha entre {col_x} e {col_y}."
                if usa_graficos:
                    figura = gerar_lineplot(df, col_x, col_y)
        elif "pairplot" in pergunta_lower or "matriz de dispersão" in pergunta_lower:
            resposta = "Matriz de dispersão das primeiras 5 variáveis numéricas."
            if usa_graficos:
                figura = gerar_pairplot(df)
        elif "pizza" in pergunta_lower or "pie" in pergunta_lower or "proporção" in pergunta_lower:
            cat_cols = df.select_dtypes(include="object").columns
            if len(cat_cols) >= 1:
                coluna = cat_cols[0]
                resposta = f"Gráfico de pizza da coluna {coluna}."
                if usa_graficos:
                    figura = gerar_pizza(df, coluna)
        elif "relatório" in pergunta_lower or "report" in pergunta_lower:
            resposta = "Gerando relatório completo do dataset."
            if usa_graficos:
                figura = gerar_relatorio(df)
                resposta += "\nRelatório gerado com sucesso."
        else:
            resposta = responder_gemini(pergunta, df)

        if resposta is not None:
            st.write(resposta)
        if figura is not None:
            if hasattr(figura, "savefig"):
                st.pyplot(figura)
            else:
                st.pyplot(figura.fig)

        st.session_state.memoria.append({
            "pergunta": pergunta,
            "resposta": str(resposta)[:2500],
        })

        st.session_state.conclusoes = gerar_conclusoes(st.session_state.memoria)

    if st.session_state.memoria:
        st.subheader("📝 Histórico de Perguntas e Respostas")
        for item in st.session_state.memoria:
            st.markdown(f"**Pergunta:** {item['pergunta']}")
            st.markdown(f"**Resposta:** {item['resposta'][:2500]} ...")

    if st.session_state.conclusoes:
        st.subheader("💡 Conclusões do Agente")
        st.write(st.session_state.conclusoes)
