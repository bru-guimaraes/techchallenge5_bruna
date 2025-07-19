import pandas as pd
import os

# Caminhos dos parquets de entrada
DIR = "data/parquet"
applicants_path = os.path.join(DIR, "applicants/applicants.parquet")
prospects_path  = os.path.join(DIR, "prospects/prospects.parquet")
vagas_path      = os.path.join(DIR, "vagas/vagas.parquet")

def gerar_dataset_treino():
    """
    Gera dataset unificado a partir dos parquets de applicants, prospects e vagas.
    Realiza merge, seleção de colunas, gera target e salva como Parquet para treino do modelo.
    """

    # 1. Carrega dados brutos dos três arquivos
    applicants = pd.read_parquet(applicants_path)
    prospects  = pd.read_parquet(prospects_path)
    vagas      = pd.read_parquet(vagas_path)

    # 2. Merge applicants + prospects pela identificação do candidato
    df = applicants.merge(
        prospects, 
        left_on="codigo_profissional", 
        right_on="codigo", 
        how="inner"
    )

    # 3. Merge com vagas para adicionar contexto da vaga
    df = df.merge(vagas, on="vaga_id", how="left")

    # 4. Seleção das colunas relevantes para modelagem (ajuste conforme necessidade)
    features = [
        'informacoes_profissionais.area_atuacao',   # área do candidato (texto separado por vírgula)
        'formacao_e_idiomas.nivel_academico',       # nível acadêmico do candidato
        'formacao_e_idiomas.nivel_ingles',          # nível de inglês do candidato
        'formacao_e_idiomas.nivel_espanhol',        # nível de espanhol do candidato
        'nivel_academico',                          # nível acadêmico exigido pela vaga
        'nivel_ingles',                             # nível de inglês exigido pela vaga
        'nivel_espanhol',                           # nível de espanhol exigido pela vaga
        'areas_atuacao',                            # áreas exigidas pela vaga (texto separado por vírgula)
        'situacao_candidado'                        # status do prospect
    ]
    df_treino = df[features].copy()

    # 5. Cria a coluna target: contratado = 1 se status contém "contratad", senão 0
    df_treino['contratado'] = (
        df_treino['situacao_candidado']
        .fillna("")
        .str.lower()
        .str.contains("contratad")
        .astype(int)
    )

    # 6. Remove duplicatas e linhas com informações essenciais faltando
    df_treino = df_treino.drop_duplicates().dropna(subset=[
        'informacoes_profissionais.area_atuacao',
        'formacao_e_idiomas.nivel_academico',
        'formacao_e_idiomas.nivel_ingles',
        'formacao_e_idiomas.nivel_espanhol'
    ])

    # 7. Salva o dataset pronto para treino
    output_path = os.path.join(DIR, "treino_unificado.parquet")
    df_treino.to_parquet(output_path, index=False)
    print(f"✅ Dataset unificado pronto para treino salvo em: {output_path}")

if __name__ == "__main__":
    gerar_dataset_treino()
