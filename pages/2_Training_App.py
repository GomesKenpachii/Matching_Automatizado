# pages/2_Training_App.py

import streamlit as st
import pandas as pd
import joblib
import os

# Imports para Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# FIX: caminhos absolutos relativos à pasta raiz do projeto
# pages/2_Training_App.py está um nível abaixo da raiz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_PATH = os.path.join(BASE_DIR, 'modelo_rf_treinado.joblib')
VETORIZADOR_PATH = os.path.join(BASE_DIR, 'vetorizador_texto.joblib')

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
            st.error("Dados insuficientes para treinamento. São necessários ao menos 10 exemplos com classificação válida.")
            return

        # Mostrar distribuição das classes
        dist = df_feedback['Classificação'].value_counts()
        st.write("Distribuição das classificações no dataset de feedback:")
        st.bar_chart(dist)

        if len(df_feedback) < 50:
            st.warning(
                f"⚠️ Dataset pequeno ({len(df_feedback)} exemplos). "
                "Com menos de 50 pares avaliados, o modelo pode não generalizar bem. "
                "Continue avaliando matches e retreine para melhorar a qualidade."
            )

        st.write("Combinando e vetorizando informações textuais...")
        colunas_texto = [
            'Curso Jovem', 'Área de Interesse 1', 'Área de Interesse 2', 'Expectativas',
            'Curso Mentor', 'Área de atuação', 'Cargo', 'Bio'
        ]
        df_feedback['texto_completo_par'] = (
            df_feedback[colunas_texto].fillna('').astype(str).agg(' '.join, axis=1)
        )

        vetorizador_texto = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        features_texto_vetorizadas = vetorizador_texto.fit_transform(df_feedback['texto_completo_par'])

        # FIX: incluir Sobreposição Área se disponível (para compatibilidade com a versão nova)
        cols_numericas = ['Similaridade TF-IDF (%)', 'Similaridade SBERT (%)', 'Similaridade Final (%)']
        if 'Sobreposição Área (%)' in df_feedback.columns:
            cols_numericas.append('Sobreposição Área (%)')
            st.write("✅ Coluna 'Sobreposição Área (%)' detectada — incluída como feature numérica.")

        features_numericas = df_feedback[cols_numericas]
        X_combinado = hstack([features_texto_vetorizadas, features_numericas])
        y = df_feedback['Classificação_Num']

        st.write(f"Dados: {len(y_train if 'y_train' in dir() else y)} amostras | {X_combinado.shape[1]} features totais ({features_texto_vetorizadas.shape[1]} textuais + {len(cols_numericas)} numéricas)")

        if y.nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_combinado, y, test_size=0.25, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_combinado, y, test_size=0.25, random_state=42
            )

        st.write(f"Divisão: {len(y_train)} amostras para treino | {len(y_test)} para teste.")

        # PASSO 2: TREINAMENTO DO MODELO
        st.subheader("2. Treinamento do Modelo")
        modelo_rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'  # compensa desbalanceamento entre classes
        )
        modelo_rf.fit(X_train, y_train)

        # FIX: salvar com caminhos absolutos para evitar erros de diretório de trabalho
        joblib.dump(modelo_rf, MODELO_PATH)
        joblib.dump(vetorizador_texto, VETORIZADOR_PATH)
        st.success(f"✅ Modelo e Vetorizador treinados e salvos com sucesso!")
        st.caption(f"Modelo salvo em: `{MODELO_PATH}`")

        # PASSO 3: AVALIAÇÃO DO MODELO
        st.subheader("3. Avaliação do Desempenho")
        y_pred = modelo_rf.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        st.metric(label="Acurácia do Modelo no conjunto de teste", value=f"{acuracia:.2%}")

        # FIX: usar apenas as classes presentes no modelo para evitar erros no classification_report
        classes_no_modelo = list(modelo_rf.classes_)
        nomes_classes_presentes = [
            nome for nome, num in mapa_classificacao.items() if num in classes_no_modelo
        ]
        # Ordenar pelo valor numérico para corresponder à ordem de classes_
        nomes_classes_presentes = sorted(
            nomes_classes_presentes,
            key=lambda nome: mapa_classificacao[nome]
        )

        st.text("Relatório de Classificação Detalhado:")
        report = classification_report(
            y_test, y_pred,
            target_names=nomes_classes_presentes,
            output_dict=True,
            zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose())

        # Informação sobre o índice da classe alvo (para debug)
        classe_alvo = 3  # Muito bom
        if classe_alvo in classes_no_modelo:
            idx = classes_no_modelo.index(classe_alvo)
            st.info(
                f"ℹ️ O modelo usará o índice **{idx}** para a classe 'Muito bom' "
                f"ao ranquear sugestões no matching preditivo."
            )
        else:
            st.warning(
                "⚠️ A classe 'Muito bom' não está presente no dataset de treinamento. "
                "Adicione exemplos classificados como 'Muito bom' para melhorar o modelo."
            )

# --- UI DA PÁGINA DE TREINAMENTO ---
st.title("🧠 2. Treinamento do Modelo Preditivo")
st.markdown("Use esta página para treinar um modelo de ML com os resultados avaliados manualmente, **incluindo os textos originais**.")
st.info("O modelo e o vetorizador de texto serão salvos para uso na página de Matching.")

with st.expander("ℹ️ Como usar esta página"):
    st.markdown("""
    1. Realize um matching na página anterior e **baixe o arquivo CSV** de resultados.
    2. Abra o CSV e adicione uma coluna **`Classificação`** com os valores: `Muito Ruim`, `Ruim`, `Bom`, `Muito bom`.
    3. Carregue o arquivo classificado aqui e clique em **Iniciar Treinamento**.
    4. Volte à página de Matching e selecione o **Modelo Preditivo (Treinado)**.

    > Quanto mais pares avaliados você acumular, melhor será a qualidade do modelo.
    """)

arquivo_feedback = st.file_uploader(
    "Carregue o arquivo de resultados classificado (matching_results.csv)",
    type="csv"
)

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
        st.success(f"Arquivo carregado com sucesso! {len(df_feedback)} registros encontrados.")

        colunas_necessarias = ['Similaridade Final (%)', 'Classificação', 'Expectativas', 'Bio']
        colunas_faltando = [c for c in colunas_necessarias if c not in df_feedback.columns]

        if not colunas_faltando:
            st.dataframe(df_feedback.head(5))
            if st.button("🚀 Iniciar Treinamento do Modelo", type="primary"):
                treinar_e_avaliar_modelo(df_feedback)
        else:
            st.error(
                f"Arquivo inválido! Colunas obrigatórias faltando: **{colunas_faltando}**\n\n"
                f"Certifique-se de que o CSV possui as colunas: `{colunas_necessarias}`"
            )
