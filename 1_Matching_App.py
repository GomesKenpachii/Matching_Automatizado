# 1_Matching_App.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import re
import os
import joblib

# Imports para Pré-processamento
import unidecode
import nltk
from nltk.corpus import stopwords

# Imports para Machine Learning (Similaridade e Features)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
from scipy.sparse import hstack

# --- 1. CONFIGURAÇÃO INICIAL E SETUP ---
st.set_page_config(
    page_title="App de Matching",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def setup_nltk():
    try: stopwords.words('portuguese')
    except LookupError: nltk.download('stopwords')
setup_nltk()

# --- 2. FUNÇÕES DE PRÉ-PROCESSAMENTO E CARREGAMENTO DE MODELOS ---
stop_words_pt = set(stopwords.words('portuguese'))

def limpar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = unidecode.unidecode(texto)
    texto = re.sub(r'[^a-z\s]', '', texto)
    texto = ' '.join([palavra for palavra in texto.split() if palavra not in stop_words_pt])
    return texto

@st.cache_resource
def carregar_modelos():
    tfidf_vectorizer = TfidfVectorizer()
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    return tfidf_vectorizer, sbert_model
tfidf_vectorizer, sbert_model = carregar_modelos()

# --- 3. FUNÇÕES DE CARREGAMENTO DE DADOS ---
@st.cache_data
def carregar_e_preparar_jovens(arquivo_jovens):
    try:
        bd_jovem = pd.read_csv(arquivo_jovens, sep=",", on_bad_lines="skip")
        bd_jovem = bd_jovem.rename(columns={'Nome Completo:': 'Nome Completo', 'Email:': 'Email Jovem', 'CPF:': 'CPF Jovem', 'Telefone:': 'Telefone Jovem','Qual o seu curso de formação?': 'Curso Jovem','Qual área é sua primeira opção de interesse profissional?': 'Área de Interesse 1','Qual área é sua segunda opção de interesse profissional?': 'Área de Interesse 2','Há alguma informação adicional que você considera importante para o seu match? Conte-nos sobre suas expectativas e o que gostaria de trabalhar durante a mentoria. Quanto mais detalhes específicos você fornecer, melhor será a chance de encontrarmos o mentor ideal para você.': 'Expectativas'})
        st.write("Limpando e pré-processando textos dos jovens...")
        colunas_texto = ['Área de Interesse 1', 'Área de Interesse 2', 'Curso Jovem', 'Expectativas']
        for col in colunas_texto:
            if col in bd_jovem.columns: bd_jovem[col] = bd_jovem[col].apply(limpar_texto)
        return bd_jovem
    except Exception as e:
        st.error(f"Erro ao processar o arquivo de jovens: {e}")
        return None

@st.cache_data
def carregar_e_preparar_mentores(arquivo_mentores):
    try:
        bd_mentor = pd.read_csv(arquivo_mentores)
        bd_mentor = bd_mentor.rename(columns={'Nome': 'Nome Mentor', 'Email': 'Email Mentor', 'CPF': 'CPF Mentor', 'Telefone': 'Telefone Mentor','Qual o seu curso de formação?': 'Curso Mentor', 'Quais são suas áreas de atuação?': 'Área de atuação','Cargo/função atual:': 'Cargo','Por favor, compartilhe uma breve biografia destacando sua trajetória e quaisquer informações relevantes que considere importantes. Este texto será encaminhado ao jovem :)': 'Bio'})
        st.write("Limpando e pré-processando textos dos mentores...")
        colunas_texto = ['Curso Mentor', 'Área de atuação', 'Cargo', 'Bio']
        for col in colunas_texto:
            if col in bd_mentor.columns: bd_mentor[col] = bd_mentor[col].apply(limpar_texto)
        return bd_mentor
    except Exception as e:
        st.error(f"Erro ao processar o arquivo de mentores: {e}")
        return None

# --- 4. FUNÇÕES DE CÁLCULO DE SIMILARIDADE E MATCHING ---
def normalizar_similaridade(similaridades):
    scaler = MinMaxScaler()
    return scaler.fit_transform(similaridades)

def calcular_similaridades(jovens, mentores, peso_tfidf, peso_sbert):
    st.info("Iniciando cálculo de similaridades...")
    jovens_textos_tfidf = (jovens['Área de Interesse 1'].astype(str) + ' ' + jovens['Área de Interesse 2'].astype(str) + ' ' + jovens['Curso Jovem'].astype(str)).fillna("")
    mentores_textos_tfidf = (mentores['Área de atuação'].astype(str) + ' ' + mentores['Curso Mentor'].astype(str) + ' ' + mentores['Cargo'].astype(str)).fillna("")
    
    vetorizador_tfidf = TfidfVectorizer()
    jovens_matrix = vetorizador_tfidf.fit_transform(jovens_textos_tfidf)
    mentores_matrix = vetorizador_tfidf.transform(mentores_textos_tfidf)
    
    similaridades_tfidf = cosine_similarity(jovens_matrix, mentores_matrix)
    similaridades_tfidf_norm = normalizar_similaridade(similaridades_tfidf)

    with st.spinner("Calculando similaridade semântica (SBERT)..."):
        jovens_expectativas = jovens['Expectativas'].fillna("").tolist()
        mentores_bios = mentores['Bio'].fillna("").tolist()
        jovens_embeds = sbert_model.encode(jovens_expectativas, convert_to_tensor=True, show_progress_bar=False)
        mentores_embeds = sbert_model.encode(mentores_bios, convert_to_tensor=True, show_progress_bar=False)
        similaridades_sbert = util.pytorch_cos_sim(jovens_embeds, mentores_embeds).cpu().numpy()
        
    similaridades_sbert_norm = normalizar_similaridade(similaridades_sbert)
    
    similaridades_finais = (peso_tfidf * similaridades_tfidf_norm) + (peso_sbert * similaridades_sbert_norm)
    return similaridades_tfidf_norm, similaridades_sbert_norm, similaridades_finais

def realizar_pareamento(jovens, mentores, peso_tfidf, peso_sbert, num_mentores_por_jovem, max_jovens_por_mentor, porcentagem_minima, metodo_selecionado):
    similaridades_tfidf_norm, similaridades_sbert_norm, similaridades_finais = calcular_similaridades(jovens, mentores, peso_tfidf, peso_sbert)
    
    if metodo_selecionado == 'Modelo Preditivo (Treinado)':
        st.info("Carregando modelo preditivo e vetorizador de texto...")
        modelo = joblib.load('modelo_rf_treinado.joblib')
        vetorizador_texto = joblib.load('vetorizador_texto.joblib')

    st.info(f"Iniciando pareamento com o método: **{metodo_selecionado}**")
    pareamentos, mentores_disponiveis = [], {mentor: 0 for mentor in mentores['Email Mentor']}
    total_jovens = len(jovens)
    if total_jovens == 0: return pd.DataFrame()
        
    barra_progresso = st.progress(0, text="Pareando jovens...")
    for contador, (i, jovem) in enumerate(jovens.iterrows()):
        if metodo_selecionado == 'Modelo de Similaridade':
            similaridade_jovem = list(enumerate(similaridades_finais[i]))
            similaridade_jovem.sort(key=lambda x: x[1], reverse=True)
            pares_ordenados = [(mentor_idx, score) for mentor_idx, score in similaridade_jovem]
        else: # Lógica para o Modelo Preditivo
            dados_pares = []
            for mentor_idx in range(len(mentores)):
                mentor = mentores.iloc[mentor_idx]
                texto_combinado = ' '.join([str(jovem.get(c, '')) for c in ['Curso Jovem', 'Área de Interesse 1', 'Área de Interesse 2', 'Expectativas']]) + \
                                  ' ' + ' '.join([str(mentor.get(c, '')) for c in ['Curso Mentor', 'Área de atuação', 'Cargo', 'Bio']])
                dados_pares.append({
                    'mentor_idx': mentor_idx,
                    'Similaridade TF-IDF (%)': similaridades_tfidf_norm[i][mentor_idx] * 100,
                    'Similaridade SBERT (%)': similaridades_sbert_norm[i][mentor_idx] * 100,
                    'Similaridade Final (%)': similaridades_finais[i][mentor_idx] * 100,
                    'texto_completo_par': texto_combinado
                })
            df_pares = pd.DataFrame(dados_pares)
            features_texto_vetorizadas = vetorizador_texto.transform(df_pares['texto_completo_par'])
            features_numericas = df_pares[['Similaridade TF-IDF (%)', 'Similaridade SBERT (%)', 'Similaridade Final (%)']]
            X_pares_combinado = hstack([features_texto_vetorizadas, features_numericas])
            probabilidades = modelo.predict_proba(X_pares_combinado)
            
            df_pares['score_preditivo'] = probabilidades[:, 3]
            df_pares.sort_values(by='score_preditivo', ascending=False, inplace=True)
            pares_ordenados = [(int(row['mentor_idx']), row['Similaridade Final (%)']/100) for index, row in df_pares.iterrows()]

        mentores_atribuidos = 0
        for mentor_idx, score_similaridade in pares_ordenados:
            
            # --- MUDANÇA: O FILTRO AGORA É CONDICIONAL ---
            if metodo_selecionado == 'Modelo de Similaridade':
                if score_similaridade * 100 < porcentagem_minima:
                    continue # Pula este mentor se estiver abaixo do limiar

            mentor_email = mentores.iloc[mentor_idx]['Email Mentor']
            if mentores_disponiveis.get(mentor_email, 0) < max_jovens_por_mentor and mentores_atribuidos < num_mentores_por_jovem:
                pareamentos.append({
                    'Nome Jovem': jovem.get('Nome Completo'), 'Email Jovem': jovem.get('Email Jovem'),'Nome Mentor': mentores.iloc[mentor_idx].get('Nome Mentor'), 'Email Mentor': mentor_email,
                    'Curso Jovem': jovem.get('Curso Jovem'), 'Área de Interesse 1': jovem.get('Área de Interesse 1'), 'Área de Interesse 2': jovem.get('Área de Interesse 2'), 'Expectativas': jovem.get('Expectativas'),
                    'Curso Mentor': mentores.iloc[mentor_idx].get('Curso Mentor'), 'Área de atuação': mentores.iloc[mentor_idx].get('Área de atuação'),'Cargo': mentores.iloc[mentor_idx].get('Cargo'), 'Bio': mentores.iloc[mentor_idx].get('Bio'),
                    'Similaridade TF-IDF (%)': round(similaridades_tfidf_norm[i][mentor_idx] * 100, 2),
                    'Similaridade SBERT (%)': round(similaridades_sbert_norm[i][mentor_idx] * 100, 2),
                    'Similaridade Final (%)': round(score_similaridade * 100, 2)
                })
                mentores_disponiveis[mentor_email] += 1
                mentores_atribuidos += 1
                if mentores_atribuidos >= num_mentores_por_jovem: break

        barra_progresso.progress((contador + 1) / total_jovens, text=f"Pareando jovens... {contador+1}/{total_jovens}")
    time.sleep(0.5); barra_progresso.empty()
    return pd.DataFrame(pareamentos)

# --- 5. INTERFACE GRÁFICA (UI) ---
st.title("🌱 1. Sistema de Matching Semear")
st.markdown("Página para gerar as combinações iniciais entre jovens e mentores.")
st.info("Use o menu na barra lateral para navegar até a página de **Treinamento do Modelo**.")

with st.sidebar:
    st.header("⚙️ Configurações de Matching")
    st.subheader("1. Escolha o Método")
    MODELO_TREINADO_PATH = 'modelo_rf_treinado.joblib'
    opcoes_metodo = ['Modelo de Similaridade']
    if os.path.exists(MODELO_TREINADO_PATH):
        opcoes_metodo.append('Modelo Preditivo (Treinado)')
    
    metodo_selecionado = st.radio("Selecione o motor de matching:", options=opcoes_metodo, help="O 'Modelo Preditivo' só aparece após ser treinado na página de Treinamento.")
    st.divider()

    st.subheader("2. Carregar Arquivos CSV")
    arquivo_jovens = st.file_uploader("Selecione o arquivo de dados dos JOVENS", type="csv")
    arquivo_mentores = st.file_uploader("Selecione o arquivo de dados dos MENTORES", type="csv")
    st.divider()
    
    st.subheader("3. Regras de Pareamento")
    num_mentores_por_jovem = st.number_input("Opções de mentores por jovem:", 1, 5, 1)
    max_jovens_por_mentor = st.number_input("Máximo de jovens por mentor:", 1, 10, 1)
    
    # --- MUDANÇA: SLIDER CONDICIONAL ---
    if metodo_selecionado == 'Modelo de Similaridade':
        st.subheader("Parâmetros do Algoritmo de Similaridade")
        peso_tfidf = st.slider("Peso da Similaridade por Interesse (TF-IDF)", 0.0, 1.0, 0.7, 0.05)
        peso_sbert = round(1.0 - peso_tfidf, 2)
        st.write(f"Peso da Similaridade por Expectativa (SBERT): {peso_sbert}")
        porcentagem_minima = st.slider("Score mínimo de similaridade para pareamento (%):", 0, 100, 70)
    else:
        # Valores padrão quando o modelo preditivo é usado.
        # `porcentagem_minima` não será usada pela lógica, mas precisa ser definida.
        peso_tfidf, peso_sbert, porcentagem_minima = 0.7, 0.3, 0

if arquivo_jovens and arquivo_mentores:
    bd_jovem = carregar_e_preparar_jovens(arquivo_jovens)
    bd_mentor = carregar_e_preparar_mentores(arquivo_mentores)
    if bd_jovem is not None and bd_mentor is not None:
        st.success("Arquivos carregados e textos pré-processados!")
        if st.button("🚀 Realizar Pareamento", type="primary"):
            df_resultados = realizar_pareamento(bd_jovem, bd_mentor, peso_tfidf, peso_sbert,num_mentores_por_jovem, max_jovens_por_mentor, porcentagem_minima, metodo_selecionado)
            st.session_state['df_resultados'] = df_resultados
        
        if 'df_resultados' in st.session_state and not st.session_state['df_resultados'].empty:
            df_resultados = st.session_state['df_resultados']
            st.success(f"🎉 Pareamento concluído! {len(df_resultados)} matches foram encontrados.")
            st.dataframe(df_resultados)
            output = io.BytesIO()
            df_resultados.to_csv(output, index=False, sep=';', encoding='utf-8-sig')
            csv_data = output.getvalue()
            st.download_button(label="📥 Baixar Resultados em CSV", data=csv_data, file_name='matching_results.csv', mime='text/csv')
else:
    st.warning("⬅️ Para iniciar um novo matching, carregue os arquivos de jovens e mentores na barra lateral.")