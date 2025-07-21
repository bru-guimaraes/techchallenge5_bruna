#!/usr/bin/env python3
import pandas as pd
import os
import sys
import glob

def gerar_dataset_treino():
    DIR = os.path.join("data", "parquet")
    applicants_path = os.path.join(DIR, "applicants", "applicants.parquet")

    # 1) Carrega applicants
    applicants = pd.read_parquet(applicants_path)

    # 2) Concatena todos os prospects disponíveis
    prospects_files = glob.glob(os.path.join(DIR, "prospects", "*.parquet"))
    prospects = pd.concat([pd.read_parquet(f) for f in prospects_files], ignore_index=True)

    # 3) Merge left: mantém todos os applicants
    df = applicants.merge(
        prospects[["prospect_id", "codigo"]],
        left_on="infos_basicas.codigo_profissional",
        right_on="codigo",
        how="left"
    )

    # 4) Cria target
    df["contratado"] = df["prospect_id"].notna().astype(int)

    # Função auxiliar para extrair coluna como Series
    def get_series(col_name, default):
        if col_name in df.columns:
            return df[col_name].fillna(default)
        # cria uma Series do valor default com o tamanho do DataFrame
        return pd.Series([default] * len(df), name=col_name)

    # 5) Seleciona/renomeia colunas, sempre com Series
    df_treino = pd.DataFrame({
        "cliente":                get_series("infos_basicas.cliente", ""),
        "nivel_profissional":     get_series("informacoes_profissionais.nivel_profissional", ""),
        "idioma_requerido":       get_series("perfil_vaga.idioma_requerido", ""),
        "eh_sap":                 get_series("infos_basicas.vaga_sap", False),
        "area_atuacao":           get_series("informacoes_profissionais.area_atuacao", ""),
        "nivel_ingles":           get_series("formacao_e_idiomas.nivel_ingles", ""),
        "nivel_espanhol":         get_series("formacao_e_idiomas.nivel_espanhol", ""),
        "nivel_academico":        get_series("perfil_vaga.nivel_academico", ""),
        "conhecimentos_tecnicos": get_series("informacoes_profissionais.conhecimentos_tecnicos", ""),
        "contratado":             df["contratado"]  # sempre existe
    })

    # 6) Limpeza: remove duplicatas e linhas sem campos obrigatórios
    df_treino = (
        df_treino
        .drop_duplicates()
        .dropna(subset=[
            "cliente", "nivel_profissional", "idioma_requerido",
            "area_atuacao", "nivel_ingles", "nivel_espanhol", "nivel_academico"
        ])
    )

    # 7) Estatística rápida de positivos
    positives = int(df_treino["contratado"].sum())
    print(f"→ Total de casos positivos (contratado=1): {positives}")

    # 8) Salva parquet unificado
    output_path = os.path.join(DIR, "parquet_treino_unificado.parquet")
    df_treino.to_parquet(output_path, index=False)
    print(f"✅ Dataset unificado pronto para treino em: {output_path}")

if __name__ == "__main__":
    try:
        gerar_dataset_treino()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
