import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from fpdf import FPDF
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
import tempfile
import pypandoc

def responder_gemini(pergunta, df, key=None):
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

def responder_openai(pergunta, df, key=None):
    data = df.head(100).to_csv(index=False)
    prompt = f"""
    Você é um assistente de análise de dados. Use os dados CSV, XLSX, XLS, ODS ou ODT abaixo para responder à pergunta.
    Dados do arquivo lido:
    {data}
    
    Pergunta: {pergunta}
    Responda de forma completa, clara e concisa. Sem retornar como resposta códigos ou tabelas, apenas texto explicativo. Ou gráficos, se necessário.
    """
    if not key:
        st.error("⚠️ A chave da API do OpenAI não está configurada. Por favor, defina a variável de ambiente 'OPENAI_API_KEY'.")
        return "Chave da API não configurada."
    else:
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um assistente de análise de dados."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content

def responder_claude(pergunta, df, key=None):
    data = df.head(100).to_csv(index=False)
    prompt = f"""
    Você é um assistente de análise de dados. Use os dados CSV, XLSX, XLS, ODS ou ODT abaixo para responder à pergunta.
    Dados do arquivo lido:
    {data}
    
    Pergunta: {pergunta}
    Responda de forma completa, clara e concisa. Sem retornar como resposta códigos ou tabelas, apenas texto explicativo. Ou gráficos, se necessário.
    """
    if not key:
        st.error("⚠️ A chave da API do Anthropic Claude não está configurada. Por favor, defina a variável de ambiente 'ANTHROPIC_API_KEY'.")
        return "Chave da API não configurada."
    else:
        client = Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-4",
            messages=[
                {"role": "system", "content": "Você é um assistente de análise de dados."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response['completion']

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
    """Retorna estatísticas descritivas do DataFrame."""
    return df.describe(include="all")


def detectar_outliers(df, contamination=0.01):
    """Detecta outliers usando Isolation Forest."""
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        raise ValueError("Não há colunas numéricas no DataFrame para detecção de outliers.")
    
    modelo = IsolationForest(contamination=contamination, random_state=42)
    modelo.fit(num_df)
    
    df_temp = df.copy()
    df_temp["outlier"] = modelo.predict(num_df)
    return df_temp[df_temp["outlier"] == -1]


def correlacoes(df):
    """Retorna matriz de correlação das colunas numéricas."""
    return df.corr()


def gerar_histograma(df, coluna):
    """Gera histograma de uma coluna numérica."""
    if coluna not in df.columns:
        raise ValueError(f"A coluna '{coluna}' não existe no DataFrame.")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[coluna], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribuição da coluna '{coluna}'")
    ax.set_xlabel(coluna)
    ax.set_ylabel("Frequência")
    plt.tight_layout()
    return fig


def gerar_heatmap(corr):
    """Gera heatmap a partir de matriz de correlação."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Mapa de Correlações")
    plt.tight_layout()
    return fig


def gerar_scatterplot(df, col_x, col_y):
    """Gera gráfico de dispersão entre duas colunas numéricas."""
    for col in [col_x, col_y]:
        if col not in df.columns:
            raise ValueError(f"A coluna '{col}' não existe no DataFrame.")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax, alpha=0.6)
    ax.set_title(f"Dispersão entre '{col_x}' e '{col_y}'")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    plt.tight_layout()
    return fig


def gerar_crosstab(df, col1, col2):
    """Gera tabela cruzada entre duas colunas."""
    for col in [col1, col2]:
        if col not in df.columns:
            raise ValueError(f"A coluna '{col}' não existe no DataFrame.")
    
    return pd.crosstab(df[col1], df[col2])


def gerar_boxplot(df, coluna):
    """Gera boxplot de uma coluna numérica."""
    if coluna not in df.columns:
        raise ValueError(f"A coluna '{coluna}' não existe no DataFrame.")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(y=df[coluna], ax=ax)
    ax.set_title(f"Boxplot da coluna '{coluna}'")
    ax.set_ylabel(coluna)
    plt.tight_layout()
    return fig


def gerar_barplot(df=None, dados=None, coluna=None, titulo="Barplot"):
    """
    Gera gráfico de barras.
    - df + coluna: conta frequências da coluna
    - dados: lista ou Series de valores
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if df is not None and coluna is not None:
        if coluna not in df.columns:
            raise ValueError(f"A coluna '{coluna}' não existe no DataFrame.")
        sns.countplot(x=df[coluna], ax=ax)
    elif dados is not None:
        sns.countplot(x=pd.Series(dados), ax=ax)
    else:
        raise ValueError("Forneça ou (df + coluna) ou (dados).")
    
    ax.set_title(titulo)
    ax.set_xlabel("")
    ax.set_ylabel("Contagem")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def gerar_lineplot(df, col_x, col_y):
    """Gera gráfico de linha entre duas colunas numéricas."""
    for col in [col_x, col_y]:
        if col not in df.columns:
            raise ValueError(f"A coluna '{col}' não existe no DataFrame.")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=df[col_x], y=df[col_y], ax=ax)
    ax.set_title(f"Evolução de '{col_y}' em função de '{col_x}'")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    plt.tight_layout()
    return fig


def gerar_pairplot(df, max_cols=5):
    """Gera pairplot das primeiras colunas numéricas (máx 5)."""
    num_df = df.select_dtypes(include="number").iloc[:, :max_cols]
    if num_df.empty:
        raise ValueError("Não há colunas numéricas para gerar pairplot.")
    
    return sns.pairplot(num_df)


def gerar_pizza(df, coluna):
    """Gera gráfico de pizza a partir de uma coluna categórica ou numérica discreta."""
    if coluna not in df.columns:
        raise ValueError(f"A coluna '{coluna}' não existe no DataFrame.")
    
    valores = df[coluna].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(valores, labels=valores.index, autopct='%1.1f%%')
    ax.set_title(f"Proporção dos valores da coluna '{coluna}'")
    plt.tight_layout()
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
if "graficos" not in st.session_state:
    st.session_state.graficos = []
if "perguntas" not in st.session_state:
    st.session_state.perguntas = []
if "respostas" not in st.session_state:
    st.session_state.respostas = []

arquivo = st.file_uploader("Carregue seu arquivo ● CSV, XLSX, XLS, ODS, ODT", 
                            type=["csv", "xlsx", "xls", "ods", "odt"])


api_key = None
# seletor entre Gemini, OpenAI e Claude
provedor = st.selectbox(
    "Selecione o modelo de linguagem",
    options=["Gemini 2.5 Flash (Google)", "GPT-4 (OpenAI)", "Claude 4 (Anthropic)"],
    index=0
)

if provedor == "GPT-4 (OpenAI)":
    api_key = st.text_input("Digite sua chave OpenAI", type="password")
elif provedor == "Gemini 2.5 Flash (Google)":
    api_key = st.text_input("Digite sua chave Google Gemini", type="password")
elif provedor == "Claude 4 (Anthropic)":
    api_key = st.text_input("Digite sua chave Claude", type="password")

if api_key:
    st.session_state["api_key"] = api_key
    st.session_state["provedor"] = provedor
    st.success(f"{provedor} pronto para uso!")
else:
    st.warning("Por favor, insira a chave do provedor selecionado.")

if provedor == "GPT-4 (OpenAI)" and not api_key:
    st.error("⚠️ Por favor, insira a chave da OpenAI.")
elif provedor == "Gemini 2.5 Flash (Google)" and not api_key:
    st.error("⚠️ Por favor, insira a chave do Google Gemini.")
elif provedor == "Claude 4 (Anthropic)" and not api_key:
    st.error("⚠️ Por favor, insira a chave do Claude.")

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
    # Pergunta ao Agente
    # ----------------------------
    st.subheader("❔ Pergunte ao Agente")
    # Definindo proporção das colunas: 4:1 → 80% e 20%
    col1, col2 = st.columns([4, 1])

    with col1:
        pergunta_manual = st.text_input("Digite sua pergunta:")
        st.checkbox("Usar gráficos na resposta", key="usar_graficos", value=True)

    with col2:
        # Usando CSS para alinhar botão à base da coluna
        st.markdown("""
            <style>
            div.stButton > button {
                vertical-align: bottom;
                height: 100%;
            }
            </style>
        """, unsafe_allow_html=True)

        if st.button("Enviar pergunta"):
            pergunta = pergunta_manual
    
    usa_graficos = st.session_state.usar_graficos

    if "usar_graficos" not in st.session_state:
        st.session_state.usar_graficos = True

    # botão para confirmar
    pergunta = None
    if pergunta_manual:
        pergunta = pergunta_manual

    if pergunta:
        pergunta_lower = pergunta.lower()
        resposta = None
        figura = None

        if df is None:
            st.error("⚠️ Por favor, carregue um arquivo antes de fazer perguntas.")

        else:
            # Funções auxiliares para seleção de colunas
            num_cols = df.select_dtypes(include="number").columns
            cat_cols = df.select_dtypes(include="object").columns

            
        if "média" in pergunta_lower:
            medias = df.mean(numeric_only=True)
            resposta = "Médias das colunas numéricas:\n" + medias.to_string()
            if usa_graficos:
                figura = gerar_barplot(dados=medias, titulo="Médias das colunas numéricas")

        elif "mediana" in pergunta_lower:
            medianas = df.median(numeric_only=True)
            resposta = "Medianas das colunas numéricas:\n" + medianas.to_string()
            if usa_graficos:
                figura = gerar_barplot(dados=medianas, titulo="Medianas das colunas numéricas")

        elif "desvio" in pergunta_lower or "variância" in pergunta_lower:
            desvios = df.std(numeric_only=True)
            resposta = "Desvios padrão das colunas numéricas:\n" + desvios.to_string()
            if usa_graficos:
                figura = gerar_barplot(dados=desvios, titulo="Desvios padrão das colunas numéricas")

        elif "outlier" in pergunta_lower or "anomalia" in pergunta_lower:
            outliers = detectar_outliers(df)
            resposta = f"Foram encontrados {len(outliers)} outliers."
            if len(outliers) > 0:
                resposta += "\nExemplos de outliers:\n" + outliers.head().to_string()
            if usa_graficos and len(num_cols) >= 2:
                figura = gerar_scatterplot(df, num_cols[0], num_cols[1])

        elif "correlação" in pergunta_lower or "correlacao" in pergunta_lower:
            corr = correlacoes(df)
            resposta = "Matriz de correlações entre variáveis numéricas."
            if usa_graficos:
                figura = gerar_heatmap(corr)

        elif "distribuição" in pergunta_lower or "histograma" in pergunta_lower:
            coluna = "Amount" if "Amount" in df.columns else num_cols[0]
            resposta = f"Histograma da coluna {coluna}."
            if usa_graficos:
                figura = gerar_histograma(df, coluna)

        elif "relacionadas" in pergunta_lower or "relações" in pergunta_lower \
            or "dispersão" in pergunta_lower or "scatter" in pergunta_lower:
            if len(num_cols) >= 2:
                resposta = f"Gráfico de dispersão entre {num_cols[0]} e {num_cols[1]}."
                if usa_graficos:
                    figura = gerar_scatterplot(df, num_cols[0], num_cols[1])

        elif "resumo" in pergunta_lower or "sumário" in pergunta_lower:
            resumo = estatisticas_basicas(df)
            resposta = "Estatísticas descritivas do dataset:\n" + resumo.to_string()
            if usa_graficos and len(num_cols) > 0:
                figura = gerar_histograma(df, num_cols[0])

        elif "tabela cruzada" in pergunta_lower or "crosstab" in pergunta_lower:
            if len(cat_cols) >= 2:
                ctab = gerar_crosstab(df, cat_cols[0], cat_cols[1])
                resposta = f"Tabela cruzada entre {cat_cols[0]} e {cat_cols[1]}:\n" + ctab.to_string()
                if usa_graficos:
                    figura = gerar_barplot(df, coluna=cat_cols[0], titulo=f"Frequência de {cat_cols[0]}")

        elif "heatmap" in pergunta_lower or "mapa" in pergunta_lower:
            corr = correlacoes(df)
            resposta = "Mapa de calor das correlações."
            if usa_graficos:
                figura = gerar_heatmap(corr)

        elif "boxplot" in pergunta_lower or "caixa" in pergunta_lower:
            coluna = "Amount" if "Amount" in df.columns else num_cols[0]
            resposta = f"Boxplot da coluna {coluna}."
            if usa_graficos:
                figura = gerar_boxplot(df, coluna)

        elif "barras" in pergunta_lower or "barplot" in pergunta_lower or "frequência" in pergunta_lower:
            if len(cat_cols) >= 1:
                resposta = f"Gráfico de barras da coluna {cat_cols[0]}."
                if usa_graficos:
                    figura = gerar_barplot(df, coluna=cat_cols[0], titulo=f"Frequência de {cat_cols[0]}")

        elif "linha" in pergunta_lower or "tendência" in pergunta_lower or "evolução" in pergunta_lower:
            if len(num_cols) >= 2:
                resposta = f"Gráfico de linha entre {num_cols[0]} e {num_cols[1]}."
                if usa_graficos:
                    figura = gerar_lineplot(df, num_cols[0], num_cols[1])

        elif "pairplot" in pergunta_lower or "matriz de dispersão" in pergunta_lower:
            resposta = "Matriz de dispersão das primeiras 5 variáveis numéricas."
            if usa_graficos:
                figura = gerar_pairplot(df)

        elif "pizza" in pergunta_lower or "pie" in pergunta_lower or "proporção" in pergunta_lower:
            if len(cat_cols) >= 1:
                resposta = f"Gráfico de pizza da coluna {cat_cols[0]}."
                if usa_graficos:
                    figura = gerar_pizza(df, cat_cols[0])

        else:
            if provedor == "GPT-4 (OpenAI)":
                resposta = responder_openai(pergunta, df, key=api_key)
            elif provedor == "Gemini 2.5 Flash (Google)":
                resposta = responder_gemini(pergunta, df, key=api_key)
            elif provedor == "Claude 4 (Anthropic)":
                resposta = responder_claude(pergunta, df, key=api_key)
            else:
                resposta = "Provedor de LLM não reconhecido."

        if resposta is not None:
            st.write(resposta)

        if figura is not None:
            st.pyplot(figura)
            st.session_state.graficos.append(figura)

        # ✅ Só salva na memória quando o usuário envia algo
        if pergunta and resposta is not None:
            # Evita adicionar duplicado na mesma execução
            if not st.session_state.memoria or st.session_state.memoria[-1]["pergunta"] != pergunta:
                st.session_state.memoria.append({
                    "pergunta": pergunta,
                    "resposta": str(resposta)[:2500],
                })
                st.session_state.conclusoes = gerar_conclusoes(st.session_state.memoria)

        # Exibe histórico e conclusões sem alterar estado
        if st.session_state.memoria:
            st.subheader("📝 Histórico de Perguntas e Respostas")
            for item in st.session_state.memoria:
                st.markdown(f"**Pergunta:** {item['pergunta']}")
                st.markdown(f"**Resposta:** {item['resposta'][:2500]} ...\n")

        if st.session_state.conclusoes:
            st.subheader("💡 Conclusões do Agente")
            st.write(st.session_state.conclusoes + "\n")
