import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
import io

# --- Configuração da Página do Streamlit ---
# st.set_page_config deve ser o primeiro comando Streamlit no seu script
st.set_page_config(
    page_title="Sistema de Matching Semear",
    page_icon="🌱",
    layout="wide"
)

# --- Cache de Modelos e Funções ---
# @st.cache_resource é usado para carregar "recursos" pesados como modelos de ML.
# O Streamlit irá carregar isso apenas uma vez e reutilizar em execuções do script,
# tornando a aplicação muito mais rápida.
@st.cache_resource
def carregar_modelos():
    """Carrega os modelos TF-IDF e SBERT apenas uma vez."""
    tfidf_vectorizer = TfidfVectorizer()
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    return tfidf_vectorizer, sbert_model

# Carrega os modelos usando a função em cache
tfidf_vectorizer, sbert_model = carregar_modelos()

# @st.cache_data é usado para armazenar dados em cache. Se a função for chamada
# com os mesmos argumentos (o mesmo conteúdo de arquivo), ela retorna o resultado
# em cache em vez de reprocessar.
@st.cache_data
def carregar_e_preparar_jovens(arquivo_jovens):
    """Carrega e renomeia as colunas do dataframe de jovens."""
    try:
        bd_jovem = pd.read_csv(arquivo_jovens, sep=",", on_bad_lines="skip")
        bd_jovem = bd_jovem.rename(columns={
            'Nome Completo:': 'Nome Completo',
            'Email:': 'Email Jovem',
            'CPF:': 'CPF Jovem',
            'Telefone:': 'Telefone Jovem',
            'Qual o seu curso de formação?': 'Curso Jovem',
            'Qual área é sua primeira opção de interesse profissional?': 'Área de Interesse 1',
            'Qual área é sua segunda opção de interesse profissional?': 'Área de Interesse 2',
            'Há alguma informação adicional que você considera importante para o seu match? Conte-nos sobre suas expectativas e o que gostaria de trabalhar durante a mentoria. Quanto mais detalhes específicos você fornecer, melhor será a chance de encontrarmos o mentor ideal para você.': 'Expectativas'
        })
        return bd_jovem
    except Exception as e:
        st.error(f"Erro ao processar o arquivo de jovens: {e}")
        return None

@st.cache_data
def carregar_e_preparar_mentores(arquivo_mentores):
    """Carrega e renomeia as colunas do dataframe de mentores."""
    try:
        bd_mentor = pd.read_csv(arquivo_mentores)
        bd_mentor = bd_mentor.rename(columns={
            'Nome': 'Nome Mentor',
            'Email': 'Email Mentor',
            'CPF': 'CPF Mentor',
            'Telefone': 'Telefone Mentor',
            'Qual o seu curso de formação?': 'Curso Mentor',
            'Quais são suas áreas de atuação?': 'Área de atuação',
            'Cargo/função atual:': 'Cargo',
            'Por favor, compartilhe uma breve biografia destacando sua trajetória e quaisquer informações relevantes que considere importantes. Este texto será encaminhado ao jovem :)': 'Bio'
        })
        return bd_mentor
    except Exception as e:
        st.error(f"Erro ao processar o arquivo de mentores: {e}")
        return None

# --- Funções de Cálculo de Similaridade (Inalteradas) ---
def calcular_similaridade_tfidf(jovens, mentores):
    jovens_interesses = (jovens['Área de Interesse 1'].astype(str) + ' ' + jovens['Área de Interesse 2'].astype(str) + ' ' + jovens['Curso Jovem'].astype(str)).fillna("")
    mentores_interesses = (mentores['Área de atuação'].astype(str) + ' ' + mentores['Curso Mentor'].astype(str) + ' ' + mentores['Cargo'].astype(str)).fillna("")
    
    # É importante usar fit_transform para os jovens e apenas transform para os mentores
    # para manter o mesmo vocabulário.
    jovens_matrix = tfidf_vectorizer.fit_transform(jovens_interesses)
    mentores_matrix = tfidf_vectorizer.transform(mentores_interesses)
    return cosine_similarity(jovens_matrix, mentores_matrix)

def calcular_similaridade_sbert(_jovens, _mentores):
    jovens_expectativas = _jovens['Expectativas'].fillna("").tolist()
    mentores_bios = _mentores['Bio'].fillna("").tolist()
    
    jovens_embeds = sbert_model.encode(jovens_expectativas, convert_to_tensor=True, show_progress_bar=True)
    mentores_embeds = sbert_model.encode(mentores_bios, convert_to_tensor=True, show_progress_bar=True)
    
    return util.pytorch_cos_sim(jovens_embeds, mentores_embeds).cpu().numpy()

def normalizar_similaridade(similaridades):
    scaler = MinMaxScaler()
    return scaler.fit_transform(similaridades)

# --- Função Principal de Pareamento (Adaptada para Streamlit) ---
def realizar_pareamento(jovens, mentores, peso_tfidf, peso_sbert, num_mentores_por_jovem, max_jovens_por_mentor, porcentagem_minima):
    st.info("Iniciando cálculo de similaridades...")

    # Cálculo de similaridade
    similaridades_tfidf = calcular_similaridade_tfidf(jovens, mentores)
    st.write("Similaridade TF-IDF calculada.")
    
    similaridades_sbert = calcular_similaridade_sbert(jovens, mentores)
    st.write("Similaridade SBERT calculada.")

    # Normalização e ponderação das similaridades
    similaridades_tfidf_norm = normalizar_similaridade(similaridades_tfidf)
    similaridades_sbert_norm = normalizar_similaridade(similaridades_sbert)
    similaridades_finais = (peso_tfidf * similaridades_tfidf_norm) + (peso_sbert * similaridades_sbert_norm)
    st.info("Similaridades normalizadas e combinadas. Iniciando pareamento...")

    pareamentos = []
    mentores_disponiveis = {mentor: 0 for mentor in mentores['Email Mentor']}
    total_jovens = len(jovens)
    
    # Barra de progresso do Streamlit
    barra_progresso = st.progress(0, text="Pareando jovens...")

    for i, jovem in jovens.iterrows():
        similaridade_jovem = list(enumerate(similaridades_finais[i]))
        similaridade_jovem.sort(key=lambda x: x[1], reverse=True)

        mentores_atribuidos = 0
        for mentor_idx, score in similaridade_jovem:
            mentor_email = mentores.iloc[mentor_idx]['Email Mentor']
            
            if (mentores_disponiveis.get(mentor_email, 0) < max_jovens_por_mentor and 
                score * 100 >= porcentagem_minima and 
                mentores_atribuidos < num_mentores_por_jovem):
                
                pareamentos.append({
                    'Nome Jovem': jovem.get('Nome Completo'),
                    'Email Jovem': jovem.get('Email Jovem'),
                    'CPF Jovem': jovem.get('CPF Jovem'),
                    'Telefone Jovem': jovem.get('Telefone Jovem'),
                    'Curso Jovem': jovem.get('Curso Jovem'),
                    'Área de Interesse 1': jovem.get('Área de Interesse 1'),
                    'Área de Interesse 2': jovem.get('Área de Interesse 2'),
                    'Expectativas': jovem.get('Expectativas'),
                    'Nome Mentor': mentores.iloc[mentor_idx].get('Nome Mentor'),
                    'Email Mentor': mentor_email,
                    'Telefone Mentor': mentores.iloc[mentor_idx].get('Telefone Mentor'),
                    'CPF Mentor': mentores.iloc[mentor_idx].get('CPF Mentor'),
                    'Curso Mentor': mentores.iloc[mentor_idx].get('Curso Mentor'),
                    'Área de atuação': mentores.iloc[mentor_idx].get('Área de atuação'),
                    'Bio': mentores.iloc[mentor_idx].get('Bio'),
                    'Similaridade TF-IDF (%)': round(similaridades_tfidf_norm[i][mentor_idx] * 100, 2),
                    'Similaridade SBERT (%)': round(similaridades_sbert_norm[i][mentor_idx] * 100, 2),
                    'Similaridade Final (%)': round(score * 100, 2)
                })
                mentores_disponiveis[mentor_email] += 1
                mentores_atribuidos += 1
                if mentores_atribuidos >= num_mentores_por_jovem:
                    break
        
        # Atualiza a barra de progresso
        barra_progresso.progress((i + 1) / total_jovens, text=f"Pareando jovens... {i+1}/{total_jovens}")

    barra_progresso.empty() # Limpa a barra de progresso ao final
    df_pareamentos = pd.DataFrame(pareamentos)
    return df_pareamentos

# --- Interface Gráfica com Streamlit ---

st.title("🌱 Sistema de Matching Semear")
st.markdown("Uma ferramenta para conectar jovens a mentores com base em seus interesses e expectativas.")

# --- Barra Lateral (Sidebar) para Controles ---
with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("1. Carregar Arquivos CSV")
    arquivo_jovens = st.file_uploader("Selecione o arquivo de dados dos JOVENS", type="csv")
    arquivo_mentores = st.file_uploader("Selecione o arquivo de dados dos MENTORES", type="csv")

    st.divider()

    st.subheader("2. Parâmetros do Algoritmo")
    
    # Pesos com slider
    peso_tfidf = st.slider("Peso da Similaridade por Interesse (TF-IDF)", 0.0, 1.0, 0.7, 0.05)
    peso_sbert = 1.0 - peso_tfidf
    st.write(f"Peso da Similaridade por Expectativa (SBERT): {peso_sbert:.2f}")

    st.divider()

    st.subheader("3. Regras de Pareamento")
    num_mentores_por_jovem = st.number_input("Opções de mentores por jovem:", min_value=1, max_value=5, value=1)
    max_jovens_por_mentor = st.number_input("Máximo de jovens por mentor:", min_value=1, max_value=10, value=3)
    porcentagem_minima = st.slider("Score mínimo para pareamento (%):", min_value=0, max_value=100, value=50)

    st.divider()
    st.info("[Documentação do Projeto](https://docs.google.com/document/d/1vCBQJTWQW5ZUxMyoVFEe58wM2FZNVlBzYLQf_HPm2zI/edit?usp=sharing)")

# --- Lógica Principal da Página ---
if arquivo_jovens and arquivo_mentores:
    bd_jovem = carregar_e_preparar_jovens(arquivo_jovens)
    bd_mentor = carregar_e_preparar_mentores(arquivo_mentores)

    if bd_jovem is not None and bd_mentor is not None:
        st.success("Arquivos carregados com sucesso!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{len(bd_jovem)} jovens encontrados:**")
            st.dataframe(bd_jovem.head())
        with col2:
            st.write(f"**{len(bd_mentor)} mentores encontrados:**")
            st.dataframe(bd_mentor.head())
        
        st.divider()

        # Botão para iniciar o pareamento
        if st.button("🚀 Realizar Pareamento", type="primary"):
            with st.spinner("Aguarde, o processo pode levar alguns minutos..."):
                df_resultados = realizar_pareamento(
                    bd_jovem, 
                    bd_mentor, 
                    peso_tfidf,
                    peso_sbert,
                    num_mentores_por_jovem,
                    max_jovens_por_mentor,
                    porcentagem_minima
                )
            
            st.success(f"🎉 Pareamento concluído! {len(df_resultados)} matches foram encontrados.")
            
            # Mostra os resultados na tela
            st.dataframe(df_resultados)

            # Prepara o arquivo para download
            # Convertendo o dataframe para CSV em memória
            output = io.BytesIO()
            # Usar to_csv com separador ';' e encoding 'utf-8-sig' para compatibilidade com Excel
            df_resultados.to_csv(output, index=False, sep=';', encoding='utf-8-sig')
            csv_data = output.getvalue()

            # Botão de download
            st.download_button(
                label="📥 Baixar Resultados em CSV",
                data=csv_data,
                file_name='matching_results.csv',
                mime='text/csv',
            )
else:
    st.warning("⬅️ Por favor, carregue os arquivos de jovens e mentores na barra lateral para começar.")