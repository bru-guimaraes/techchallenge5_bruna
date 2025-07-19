# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from pathlib import Path

# Define o caminho robusto até o arquivo Parquet, relativo à raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
PATH_TREINO = BASE_DIR / "data" / "parquet" / "treino_unificado.parquet"
PATH_FEATURES = BASE_DIR / "data" / "parquet" / "treino_features.parquet"
PATH_ENCODER = BASE_DIR / "model" / "area_atuacao_mlb.joblib"


def processar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função que faz todo o feature engineering do dataset para treino.
    Inclui:
      - One-hot/multilabel das áreas de atuação do candidato
      - Score de match entre áreas da vaga e do candidato
      - Seleção e concatenação das demais variáveis
    """
    # --- 1. Binariza área de atuação do candidato ---
    area_lists = df['informacoes_profissionais.area_atuacao'] \
        .apply(lambda x: [i.strip() for i in str(x).split(',')] if pd.notnull(x) else [])
    mlb = MultiLabelBinarizer()
    area_binarized = mlb.fit_transform(area_lists)
    area_df = pd.DataFrame(
        area_binarized,
        columns=[f"informacoes_profissionais.area_atuacao_{cls}" for cls in mlb.classes_],
        index=df.index
    )

    # --- 2. Calcula score de match entre áreas do candidato e da vaga ---
    def calc_skill_match(row):
        cand = set([i.strip() for i in str(row['informacoes_profissionais.area_atuacao']).split(',') if i.strip()])
        vaga = set([i.strip() for i in str(row['areas_atuacao']).split(',') if i.strip()])
        return len(cand & vaga)

    df['skill_match_score'] = df.apply(calc_skill_match, axis=1)

    # --- 3. Seleção e montagem do DataFrame final ---
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
    df_final = pd.concat([df[cols_modelo], area_df], axis=1)

    # --- 4. Salva o MultiLabelBinarizer para deploy/inferência ---
    joblib.dump(mlb, PATH_ENCODER)

    return df_final


def processar_features_inference(
    req: dict,
    feature_names: list,
    mlb_path: str
) -> pd.DataFrame:
    """
    Função que recria, a partir de um dict de entrada, 
    exatamente as features usadas pelo modelo no treino.
    """
    # 1) Constroi DataFrame cru com os mesmos campos do treino
    df_raw = pd.DataFrame([{
        'informacoes_profissionais.area_atuacao': req['area_atuacao'],
        'areas_atuacao': req['areas_atuacao'],
        'formacao_e_idiomas.nivel_academico': req['nivel_academico'],
        'formacao_e_idiomas.nivel_ingles': req['nivel_ingles'],
        'formacao_e_idiomas.nivel_espanhol': req['nivel_espanhol'],
        'nivel_academico': req['nivel_academico'],
        'nivel_ingles': req['nivel_ingles'],
        'nivel_espanhol': req['nivel_espanhol']
    }])

    # 2) Tenta carregar o binarizador; se não existir, cria DataFrame vazio
    try:
        mlb = joblib.load(mlb_path)
        area_lists = df_raw['informacoes_profissionais.area_atuacao'] \
            .apply(lambda x: [i.strip() for i in x.split(',')] if pd.notnull(x) else [])
        area_binarized = mlb.transform(area_lists)
        area_df = pd.DataFrame(
            area_binarized,
            columns=[f"informacoes_profissionais.area_atuacao_{cls}" for cls in mlb.classes_],
            index=df_raw.index
        )
    except Exception:
        # fallback: nenhuma coluna de área
        area_df = pd.DataFrame(index=df_raw.index)

    # 3) Calcula skill_match_score
    def calc_skill_match(row):
        cand = set([i.strip() for i in row['informacoes_profissionais.area_atuacao'].split(',') if i.strip()])
        vaga = set([i.strip() for i in row['areas_atuacao'].split(',') if i.strip()])
        return len(cand & vaga)

    df_raw['skill_match_score'] = df_raw.apply(calc_skill_match, axis=1)

    # 4) Junta tudo e dummifica o restante
    df_feat = pd.concat([
        df_raw.drop(columns=['informacoes_profissionais.area_atuacao', 'areas_atuacao']),
        area_df
    ], axis=1)
    df_enc = pd.get_dummies(df_feat)

    # 5) Alinha ao conjunto de features do modelo
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    return df_aligned


if __name__ == "__main__":
    # Exemplo de uso no pipeline de treino
    df = pd.read_parquet(PATH_TREINO)
    df_final = processar_features(df)
    df_final.to_parquet(PATH_FEATURES, index=False)
    print("✅ Features processadas e salvas para treino!")
