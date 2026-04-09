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

# Imports para Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
from scipy.sparse import hstack

# --- CAMINHOS ABSOLUTOS DOS MODELOS ---
# FIX: usar caminhos absolutos para evitar erros dependentes do diretório de trabalho
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, 'modelo_rf_treinado.joblib')
VETORIZADOR_PATH = os.path.join(BASE_DIR, 'vetorizador_texto.joblib')

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
    # FIX: modelo multilingual para suporte correto ao português
    # 'paraphrase-multilingual-mpnet-base-v2' suporta 50+ idiomas incluindo PT-BR
    sbert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return sbert_model

sbert_model = carregar_modelos()

# --- 3. FUNÇÕES DE CARREGAMENTO DE DADOS ---
@st.cache_data
def carregar_e_preparar_jovens(arquivo_jovens):
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

        colunas_texto = ['Área de Interesse 1', 'Área de Interesse 2', 'Curso Jovem', 'Expectativas']
        for col in colunas_texto:
            if col in bd_jovem.columns:
                # FIX CRÍTICO: salvar texto original ANTES de limpar
                # O SBERT precisa de linguagem natural — texto limpo degrada sua qualidade
                bd_jovem[f'{col}_original'] = bd_jovem[col].fillna("")
                # Texto limpo é usado apenas pelo TF-IDF
                bd_jovem[col] = bd_jovem[col].apply(limpar_texto)

        return bd_jovem
    except Exception as e:
        st.error(f"Erro ao processar o arquivo de jovens: {e}")
        return None

@st.cache_data
def carregar_e_preparar_mentores(arquivo_mentores):
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

        colunas_texto = ['Curso Mentor', 'Área de atuação', 'Cargo', 'Bio']
        for col in colunas_texto:
            if col in bd_mentor.columns:
                # FIX CRÍTICO: salvar texto original para uso no SBERT
                bd_mentor[f'{col}_original'] = bd_mentor[col].fillna("")
                bd_mentor[col] = bd_mentor[col].apply(limpar_texto)

        return bd_mentor
    except Exception as e:
        st.error(f"Erro ao processar o arquivo de mentores: {e}")
        return None

# --- 4. FUNÇÕES DE CÁLCULO DE SIMILARIDADE E MATCHING ---
def normalizar_similaridade(similaridades):
    """
    FIX: normaliza a matriz inteira como um único vetor plano.
    A versão anterior normalizava coluna por coluna (MinMaxScaler padrão),
    o que distorcia os scores — o pior mentor de cada coluna sempre recebia 0
    e o melhor sempre 1, independente dos valores absolutos.
    """
    scaler = MinMaxScaler()
    shape_original = similaridades.shape
    flat = similaridades.reshape(-1, 1)
    scaled = scaler.fit_transform(flat)
    return scaled.reshape(shape_original)

def calcular_sobreposicao_area(jovens, mentores):
    """
    NOVA FEATURE: calcula a sobreposição direta (Jaccard) entre as palavras-chave
    das áreas de interesse do jovem e as áreas de atuação do mentor.
    Captura correspondências diretas que o modelo semântico pode perder.
    """
    n_j = len(jovens)
    n_m = len(mentores)
    scores = np.zeros((n_j, n_m))

    jovens_r = jovens.reset_index(drop=True)
    mentores_r = mentores.reset_index(drop=True)

    for i in range(n_j):
        palavras_jovem = set(
            str(jovens_r.iloc[i].get('Área de Interesse 1', '')).split() +
            str(jovens_r.iloc[i].get('Área de Interesse 2', '')).split()
        )
        palavras_jovem.discard('')

        for j in range(n_m):
            palavras_mentor = set(str(mentores_r.iloc[j].get('Área de atuação', '')).split())
            palavras_mentor.discard('')

            if palavras_jovem and palavras_mentor:
                intersecao = palavras_jovem & palavras_mentor
                uniao = palavras_jovem | palavras_mentor
                scores[i][j] = len(intersecao) / len(uniao)

    return scores

def calcular_similaridades(jovens, mentores, peso_tfidf, peso_sbert, peso_area):
    st.info("Iniciando cálculo de similaridades...")

    # TF-IDF: usa texto LIMPO (correto — TF-IDF se beneficia de normalização de texto)
    jovens_textos_tfidf = (
        jovens['Área de Interesse 1'].astype(str) + ' ' +
        jovens['Área de Interesse 2'].astype(str) + ' ' +
        jovens['Curso Jovem'].astype(str)
    ).fillna("")
    mentores_textos_tfidf = (
        mentores['Área de atuação'].astype(str) + ' ' +
        mentores['Curso Mentor'].astype(str) + ' ' +
        mentores['Cargo'].astype(str)
    ).fillna("")

    vetorizador_tfidf = TfidfVectorizer()
    jovens_matrix = vetorizador_tfidf.fit_transform(jovens_textos_tfidf)
    mentores_matrix = vetorizador_tfidf.transform(mentores_textos_tfidf)
    similaridades_tfidf = cosine_similarity(jovens_matrix, mentores_matrix)
    similaridades_tfidf_norm = normalizar_similaridade(similaridades_tfidf)

    # FIX CRÍTICO: SBERT usa texto ORIGINAL (linguagem natural completa)
    # Modelo multilingual suporta português nativamente
    with st.spinner("Calculando similaridade semântica (SBERT multilingual)..."):
        jovens_expectativas = jovens['Expectativas_original'].fillna("").tolist()
        mentores_bios = mentores['Bio_original'].fillna("").tolist()
        jovens_embeds = sbert_model.encode(jovens_expectativas, convert_to_tensor=True, show_progress_bar=False)
        mentores_embeds = sbert_model.encode(mentores_bios, convert_to_tensor=True, show_progress_bar=False)
        similaridades_sbert = util.pytorch_cos_sim(jovens_embeds, mentores_embeds).cpu().numpy()
    similaridades_sbert_norm = normalizar_similaridade(similaridades_sbert)

    # NOVA FEATURE: sobreposição direta de área (coeficiente de Jaccard)
    with st.spinner("Calculando sobreposição direta de áreas..."):
        sobreposicao_area = calcular_sobreposicao_area(jovens, mentores)
        # Jaccard já está em [0,1] — normalizamos da mesma forma para comparabilidade
        if sobreposicao_area.max() > 0:
            sobreposicao_area_norm = normalizar_similaridade(sobreposicao_area)
        else:
            sobreposicao_area_norm = sobreposicao_area  # tudo zero, normalização não muda nada

    # Normalizar pesos para garantir que somam 1
    total_peso = peso_tfidf + peso_sbert + peso_area
    if total_peso == 0:
        total_peso = 1.0
    p_tfidf = peso_tfidf / total_peso
    p_sbert = peso_sbert / total_peso
    p_area = peso_area / total_peso

    similaridades_finais = (
        p_tfidf * similaridades_tfidf_norm +
        p_sbert * similaridades_sbert_norm +
        p_area * sobreposicao_area_norm
    )

    return similaridades_tfidf_norm, similaridades_sbert_norm, sobreposicao_area, similaridades_finais

def realizar_pareamento(jovens, mentores, peso_tfidf, peso_sbert, peso_area,
                        num_mentores_por_jovem, max_jovens_por_mentor,
                        porcentagem_minima, metodo_selecionado):

    similaridades_tfidf_norm, similaridades_sbert_norm, sobreposicao_area, similaridades_finais = \
        calcular_similaridades(jovens, mentores, peso_tfidf, peso_sbert, peso_area)

    idx_classe_alvo = None
    if metodo_selecionado == 'Modelo Preditivo (Treinado)':
        st.info("Carregando modelo preditivo e vetorizador de texto...")
        modelo = joblib.load(MODELO_PATH)
        vetorizador_texto = joblib.load(VETORIZADOR_PATH)

        # FIX CRÍTICO: encontrar o índice da classe "Muito bom" dinamicamente
        # O sklearn ordena as classes — índice fixo 3 pode ser inválido
        classe_alvo = 3  # Muito bom
        if classe_alvo in modelo.classes_:
            idx_classe_alvo = list(modelo.classes_).index(classe_alvo)
        else:
            idx_classe_alvo = len(modelo.classes_) - 1
            st.warning(
                f"Classe 'Muito bom' (3) não encontrada no modelo. "
                f"Usando a classe de maior valor ({modelo.classes_[idx_classe_alvo]}) como alvo."
            )

    st.info(f"Iniciando pareamento com o método: **{metodo_selecionado}**")

    # FIX CRÍTICO: resetar índice para garantir que posição sequencial == índice do DataFrame
    # Sem isso, jovens[i] na matriz aponta para o jovem errado se o DataFrame foi filtrado
    jovens = jovens.reset_index(drop=True)
    mentores = mentores.reset_index(drop=True)

    pareamentos = []
    mentores_disponiveis = {mentor: 0 for mentor in mentores['Email Mentor']}
    total_jovens = len(jovens)
    if total_jovens == 0: return pd.DataFrame()

    barra_progresso = st.progress(0, text="Pareando jovens...")

    # FIX CRÍTICO: iterar por posição (contador) e não por índice do DataFrame (i)
    # Índice do DataFrame pode não ser sequencial após filtros/reindexação
    for contador in range(total_jovens):
        jovem = jovens.iloc[contador]

        if metodo_selecionado == 'Modelo de Similaridade':
            # Usar `contador` (posição na matriz) — não o índice do DataFrame
            similaridade_jovem = list(enumerate(similaridades_finais[contador]))
            similaridade_jovem.sort(key=lambda x: x[1], reverse=True)
            pares_ordenados = [(mentor_idx, score) for mentor_idx, score in similaridade_jovem]

        else:  # Modelo Preditivo
            dados_pares = []
            for mentor_idx in range(len(mentores)):
                mentor = mentores.iloc[mentor_idx]
                texto_combinado = (
                    ' '.join([str(jovem.get(c, '')) for c in ['Curso Jovem', 'Área de Interesse 1', 'Área de Interesse 2', 'Expectativas']]) +
                    ' ' +
                    ' '.join([str(mentor.get(c, '')) for c in ['Curso Mentor', 'Área de atuação', 'Cargo', 'Bio']])
                )
                dados_pares.append({
                    'mentor_idx': mentor_idx,
                    'Similaridade TF-IDF (%)': similaridades_tfidf_norm[contador][mentor_idx] * 100,
                    'Similaridade SBERT (%)': similaridades_sbert_norm[contador][mentor_idx] * 100,
                    'Sobreposição Área (%)': sobreposicao_area[contador][mentor_idx] * 100,
                    'Similaridade Final (%)': similaridades_finais[contador][mentor_idx] * 100,
                    'texto_completo_par': texto_combinado
                })

            df_pares = pd.DataFrame(dados_pares)
            features_texto_vetorizadas = vetorizador_texto.transform(df_pares['texto_completo_par'])
            features_numericas = df_pares[['Similaridade TF-IDF (%)', 'Similaridade SBERT (%)', 'Similaridade Final (%)']]
            X_pares_combinado = hstack([features_texto_vetorizadas, features_numericas])
            probabilidades = modelo.predict_proba(X_pares_combinado)

            # FIX CRÍTICO: usar índice dinâmico da classe alvo
            df_pares['score_preditivo'] = probabilidades[:, idx_classe_alvo]
            df_pares.sort_values(by='score_preditivo', ascending=False, inplace=True)

            # FIX: usar score_preditivo no ranking (antes usava similaridade original)
            pares_ordenados = [(int(row['mentor_idx']), row['score_preditivo']) for _, row in df_pares.iterrows()]

        mentores_atribuidos = 0
        for mentor_idx, score in pares_ordenados:
            if metodo_selecionado == 'Modelo de Similaridade':
                if score * 100 < porcentagem_minima:
                    continue

            mentor_email = mentores.iloc[mentor_idx]['Email Mentor']
            if mentores_disponiveis.get(mentor_email, 0) < max_jovens_por_mentor and mentores_atribuidos < num_mentores_por_jovem:
                pareamentos.append({
                    'Nome Jovem': jovem.get('Nome Completo'),
                    'Email Jovem': jovem.get('Email Jovem'),
                    'Nome Mentor': mentores.iloc[mentor_idx].get('Nome Mentor'),
                    'Email Mentor': mentor_email,
                    'Curso Jovem': jovem.get('Curso Jovem'),
                    'Área de Interesse 1': jovem.get('Área de Interesse 1'),
                    'Área de Interesse 2': jovem.get('Área de Interesse 2'),
                    'Expectativas': jovem.get('Expectativas'),
                    'Curso Mentor': mentores.iloc[mentor_idx].get('Curso Mentor'),
                    'Área de atuação': mentores.iloc[mentor_idx].get('Área de atuação'),
                    'Cargo': mentores.iloc[mentor_idx].get('Cargo'),
                    'Bio': mentores.iloc[mentor_idx].get('Bio'),
                    'Similaridade TF-IDF (%)': round(similaridades_tfidf_norm[contador][mentor_idx] * 100, 2),
                    'Similaridade SBERT (%)': round(similaridades_sbert_norm[contador][mentor_idx] * 100, 2),
                    'Sobreposição Área (%)': round(sobreposicao_area[contador][mentor_idx] * 100, 2),
                    'Similaridade Final (%)': round(similaridades_finais[contador][mentor_idx] * 100, 2)
                })
                mentores_disponiveis[mentor_email] += 1
                mentores_atribuidos += 1
                if mentores_atribuidos >= num_mentores_por_jovem: break

        barra_progresso.progress((contador + 1) / total_jovens, text=f"Pareando jovens... {contador+1}/{total_jovens}")

    time.sleep(0.5)
    barra_progresso.empty()
    return pd.DataFrame(pareamentos)

# --- 5. INTERFACE GRÁFICA (UI) ---
st.title("🌱 1. Sistema de Matching Semear")
st.markdown("Página para gerar as combinações iniciais entre jovens e mentores.")
st.info("Use o menu na barra lateral para navegar até a página de **Treinamento do Modelo**.")

with st.sidebar:
    st.header("⚙️ Configurações de Matching")
    st.subheader("1. Escolha o Método")

    opcoes_metodo = ['Modelo de Similaridade']
    if os.path.exists(MODELO_PATH):
        opcoes_metodo.append('Modelo Preditivo (Treinado)')

    metodo_selecionado = st.radio(
        "Selecione o motor de matching:",
        options=opcoes_metodo,
        help="O 'Modelo Preditivo' só aparece após ser treinado na página de Treinamento."
    )
    st.divider()

    st.subheader("2. Carregar Arquivos CSV")
    arquivo_jovens = st.file_uploader("Selecione o arquivo de dados dos JOVENS", type="csv")
    arquivo_mentores = st.file_uploader("Selecione o arquivo de dados dos MENTORES", type="csv")
    st.divider()

    st.subheader("3. Regras de Pareamento")
    num_mentores_por_jovem = st.number_input("Opções de mentores por jovem:", 1, 5, 1)
    max_jovens_por_mentor = st.number_input("Máximo de jovens por mentor:", 1, 10, 1)

    if metodo_selecionado == 'Modelo de Similaridade':
        st.subheader("4. Pesos do Algoritmo")
        st.caption("Os pesos são normalizados automaticamente para somar 100%.")

        peso_tfidf = st.slider(
            "Peso: Interesse/Área/Curso (TF-IDF)", 0.0, 1.0, 0.5, 0.05,
            help="Similaridade por palavras-chave nas áreas de interesse e curso."
        )
        peso_sbert = st.slider(
            "Peso: Expectativas/Bio (Semântico)", 0.0, 1.0, 0.3, 0.05,
            help="Similaridade semântica entre as expectativas do jovem e a bio do mentor."
        )
        peso_area = st.slider(
            "Peso: Sobreposição Direta de Área", 0.0, 1.0, 0.2, 0.05,
            help="Bônus para correspondência direta entre áreas de interesse e atuação (Jaccard)."
        )

        total = peso_tfidf + peso_sbert + peso_area
        if total > 0:
            st.caption(
                f"Pesos normalizados: "
                f"TF-IDF={peso_tfidf/total:.0%} | "
                f"Semântico={peso_sbert/total:.0%} | "
                f"Área={peso_area/total:.0%}"
            )

        porcentagem_minima = st.slider("Score mínimo para aceitar pareamento (%):", 0, 100, 60)
    else:
        peso_tfidf, peso_sbert, peso_area, porcentagem_minima = 0.5, 0.3, 0.2, 0

if arquivo_jovens and arquivo_mentores:
    bd_jovem = carregar_e_preparar_jovens(arquivo_jovens)
    bd_mentor = carregar_e_preparar_mentores(arquivo_mentores)
    if bd_jovem is not None and bd_mentor is not None:
        st.success("Arquivos carregados e textos pré-processados!")
        if st.button("🚀 Realizar Pareamento", type="primary"):
            df_resultados = realizar_pareamento(
                bd_jovem, bd_mentor,
                peso_tfidf, peso_sbert, peso_area,
                num_mentores_por_jovem, max_jovens_por_mentor,
                porcentagem_minima, metodo_selecionado
            )
            st.session_state['df_resultados'] = df_resultados

        if 'df_resultados' in st.session_state and not st.session_state['df_resultados'].empty:
            df_resultados = st.session_state['df_resultados']
            st.success(f"🎉 Pareamento concluído! {len(df_resultados)} matches foram encontrados.")
            st.dataframe(df_resultados)
            output = io.BytesIO()
            df_resultados.to_csv(output, index=False, sep=';', encoding='utf-8-sig')
            csv_data = output.getvalue()
            st.download_button(
                label="📥 Baixar Resultados em CSV",
                data=csv_data,
                file_name='matching_results.csv',
                mime='text/csv'
            )
else:
    st.warning("⬅️ Para iniciar um novo matching, carregue os arquivos de jovens e mentores na barra lateral.")
