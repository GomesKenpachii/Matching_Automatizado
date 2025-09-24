# pages/2_Training_App.py

import streamlit as st
import pandas as pd
import joblib

# Imports para Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# --- FUNÇÃO DE TREINAMENTO DO MODELO PREDITIVO ---
def treinar_e_avaliar_modelo(df_feedback):
    with st.spinner("Treinando o modelo de Machine Learning com textos..."):
        # PASSO 1: PREPARAÇÃO DOS DADOS
        st.subheader("1. Preparação dos Dados")
        mapa_classificacao = {'Muito Ruim': 0, 'Ruim': 1, 'Bom': 2, 'Muito bom': 3}
        df_feedback['Classificação'] = df_feedback['Classificação'].astype(str)
        df_feedback['Classificação_Num'] = df_feedback['Classificação'].map(mapa_classificacao)
        df_feedback.dropna(subset=['Classificação_Num'], inplace=True)
        df_feedback['Classificação_Num'] = df_feedback['Classificação_Num'].astype(int)

        if len(df_feedback) < 10:
            st.error("Dados insuficientes para treinamento. É necessário ter pelo menos 10 exemplos com classificação válida.")
            return

        st.write("Combinando e vetorizando informações textuais...")
        colunas_texto = ['Curso Jovem', 'Área de Interesse 1', 'Área de Interesse 2', 'Expectativas', 'Curso Mentor', 'Área de atuação', 'Cargo', 'Bio']
        df_feedback['texto_completo_par'] = df_feedback[colunas_texto].fillna('').astype(str).agg(' '.join, axis=1)

        vetorizador_texto = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        features_texto_vetorizadas = vetorizador_texto.fit_transform(df_feedback['texto_completo_par'])
        
        features_numericas = df_feedback[['Similaridade TF-IDF (%)', 'Similaridade SBERT (%)', 'Similaridade Final (%)']]
        X_combinado = hstack([features_texto_vetorizadas, features_numericas])
        y = df_feedback['Classificação_Num']
        
        if y.nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(X_combinado, y, test_size=0.25, random_state=42, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_combinado, y, test_size=0.25, random_state=42)
            
        st.write(f"Dados divididos em {len(y_train)} para treino e {len(y_test)} para teste.")
        st.write(f"Número total de features (texto + numéricas): {X_combinado.shape[1]}")

        # PASSO 2: TREINAMENTO DO MODELO
        st.subheader("2. Treinamento do Modelo")
        modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        modelo_rf.fit(X_train, y_train)
        
        joblib.dump(modelo_rf, 'modelo_rf_treinado.joblib')
        joblib.dump(vetorizador_texto, 'vetorizador_texto.joblib')
        st.success("✅ Modelo e Vetorizador de Texto foram treinados e salvos com sucesso!")
        
        # PASSO 3: AVALIAÇÃO DO MODELO
        st.subheader("3. Avaliação do Desempenho")
        y_pred = modelo_rf.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        st.metric(label="Acurácia do Modelo no conjunto de teste", value=f"{acuracia:.2%}")
        
        st.text("Relatório de Classificação Detalhado:")
        report = classification_report(y_test, y_pred, target_names=list(mapa_classificacao.keys()), output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose())

# --- UI DA PÁGINA DE TREINAMENTO ---
st.title("🧠 2. Treinamento do Modelo Preditivo")
st.markdown("Use esta página para treinar um modelo de ML com os resultados avaliados manualmente, **incluindo os textos originais**.")
st.info("O modelo e o vetorizador de texto serão salvos para uso na página de Matching.")

arquivo_feedback = st.file_uploader("Carregue o arquivo de resultados classificado (matching_results.csv)", type="csv")

if arquivo_feedback:
    df_feedback = None
    try:
        arquivo_feedback.seek(0)
        df_feedback = pd.read_csv(arquivo_feedback, sep=';')
    except Exception:
        try:
            arquivo_feedback.seek(0)
            df_feedback = pd.read_csv(arquivo_feedback, sep=',')
        except Exception as e:
            st.error(f"Não foi possível ler o arquivo CSV. Verifique o formato. Erro: {e}")
    
    if df_feedback is not None:
        st.success("Arquivo de feedback carregado com sucesso!")
        colunas_necessarias = ['Similaridade Final (%)', 'Classificação', 'Expectativas', 'Bio']
        if all(col in df_feedback.columns for col in colunas_necessarias):
            if st.button("🚀 Iniciar Treinamento do Modelo", type="primary"):
                treinar_e_avaliar_modelo(df_feedback)
        else:
            st.error(f"Arquivo inválido! Faltam colunas necessárias. Verifique se o CSV contém: {colunas_necessarias}")