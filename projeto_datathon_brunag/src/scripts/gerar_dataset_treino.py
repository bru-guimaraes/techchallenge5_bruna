#!/usr/bin/env python3
import pandas as pd
import os
import sys

def gerar_dataset_treino():
    DIR = os.path.join("data", "parquet")
    applicants_path = os.path.join(DIR, "applicants", "applicants.parquet")
    prospects_path  = os.path.join(DIR, "prospects",  "prospects.parquet")

    # 1) Carrega applicants e prospects
    applicants = pd.read_parquet(applicants_path)
    prospects  = pd.read_parquet(prospects_path)

    # 2) Merge left: mantém todos os applicants (positivos e negativos)
    df = applicants.merge(
        prospects[["prospect_id", "codigo"]],
        left_on="infos_basicas.codigo_profissional",
        right_on="codigo",
        how="left"
    )

    # 3) Criar target: 1 se virou prospect, senão 0
    df["contratado"] = df["prospect_id"].notna().astype(int)

    # 4) Seleciona e renomeia colunas de entrada
    df_treino = pd.DataFrame({
        "cliente":                df.get("infos_basicas.cliente", ""),
        "nivel_profissional":     df.get("informacoes_profissionais.nivel_profissional", ""),
        "idioma_requerido":       df.get("perfil_vaga.idioma_requerido", ""),
        "eh_sap":                 df.get("infos_basicas.vaga_sap", False),
        "area_atuacao":           df.get("informacoes_profissionais.area_atuacao", ""),
        "nivel_ingles":           df.get("formacao_e_idiomas.nivel_ingles", ""),
        "nivel_espanhol":         df.get("formacao_e_idiomas.nivel_espanhol", ""),
        "nivel_academico":        df.get("perfil_vaga.nivel_academico", ""),
        "conhecimentos_tecnicos": df.get("informacoes_profissionais.conhecimentos_tecnicos", ""),
        "contratado":             df["contratado"]
    })

    # 5) Limpeza: remove duplicatas e linhas sem campos obrigatórios
    df_treino = (
        df_treino
        .drop_duplicates()
        .dropna(subset=[
            "cliente",
            "nivel_profissional",
            "idioma_requerido",
            "area_atuacao",
            "nivel_ingles",
            "nivel_espanhol",
            "nivel_academico"
        ])
    )

    # 6) Salva parquet unificado
    output_path = os.path.join(DIR, "parquet_treino_unificado.parquet")
    df_treino.to_parquet(output_path, index=False)
    print(f"✅ Dataset unificado pronto para treino em: {output_path}")

if __name__ == "__main__":
    try:
        gerar_dataset_treino()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
