import os
import json
import pandas as pd

def carregar_dados_json(diretorio: str) -> pd.DataFrame:
    """
    Lê todos os arquivos .json em um diretório e concatena em um único DataFrame.
    """
    dados = []

    for arquivo in os.listdir(diretorio):
        if not arquivo.endswith(".json"):
            continue

        caminho_arquivo = os.path.join(diretorio, arquivo)
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                conteudo = json.load(f)
                if isinstance(conteudo, list):
                    dados.extend(conteudo)
                elif isinstance(conteudo, dict):
                    dados.append(conteudo)
        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")

    if not dados:
        print("⚠️ Nenhum dado válido encontrado.")
        return pd.DataFrame()

    df = pd.DataFrame(dados)
    print(f"✅ {len(df)} registros carregados de {diretorio}")
    return df
