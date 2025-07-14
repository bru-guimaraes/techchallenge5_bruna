import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

def processar_features(df):
    """
    Função que faz todo o feature engineering do dataset.
    Inclui:
      - One-hot/multilabel das áreas de atuação do candidato
      - Score de match entre áreas da vaga e do candidato
      - Seleção e tratamento das demais variáveis
    """

    # --- 1. Binariza área de atuação do candidato ---
    # Transforma string "Dev, QA, Dados" em lista ['Dev', 'QA', 'Dados']
    area_lists = df['informacoes_profissionais.area_atuacao'].apply(
        lambda x: [i.strip() for i in str(x).split(',')] if pd.notnull(x) else []
    )

    # Aplica o MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    area_binarized = mlb.fit_transform(area_lists)
    area_df = pd.DataFrame(area_binarized, columns=[f"area_atuacao__{cls}" for cls in mlb.classes_], index=df.index)

    # --- 2. Calcula score de match entre áreas do candidato e da vaga ---
    def calc_skill_match(row):
        cand = set([i.strip() for i in str(row['informacoes_profissionais.area_atuacao']).split(',') if i.strip()])
        vaga = set([i.strip() for i in str(row['areas_atuacao']).split(',') if i.strip()])
        return len(cand & vaga)  # interseção: quantas áreas coincidem

    df['skill_match_score'] = df.apply(calc_skill_match, axis=1)

    # --- 3. Junta tudo num novo DataFrame ---
    # Ajuste os nomes das features conforme seu projeto!
    cols_modelo = [
        'formacao_e_idiomas.nivel_academico',
        'formacao_e_idiomas.nivel_ingles',
        'formacao_e_idiomas.nivel_espanhol',
        'nivel_academico',
        'nivel_ingles',
        'nivel_espanhol',
        'skill_match_score',
        'contratado'
    ]

    # Concatena as variáveis originais + binarizadas
    df_final = pd.concat([df[cols_modelo], area_df], axis=1)

    # --- 4. Salva o MultiLabelBinarizer treinado para usar no deploy/API ---
    joblib.dump(mlb, 'model/area_atuacao_mlb.joblib')

    return df_final

if __name__ == "__main__":
    # Lê o dataset já pré-processado
    df = pd.read_parquet('data/parquet/treino_unificado.parquet')
    df_final = processar_features(df)
    df_final.to_parquet('data/parquet/treino_features.parquet', index=False)
    print("✅ Features processadas e salvas!")
