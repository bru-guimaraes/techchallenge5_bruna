import os
import json
import zipfile
import pandas as pd

INPUT_DIR = "data/raw_downloads"
OUTPUT_BASE_DIR = "data/parquet"

def processar_json_para_dataframe(nome_arquivo, json_data):
    registros = []

    if "applicants" in nome_arquivo:
        for codigo, dados in json_data.items():
            registro = {"codigo_profissional": codigo}
            registro.update(dados.get("infos_basicas", {}))
            registro.update(dados.get("informacoes_pessoais", {}))
            registro.update(dados.get("informacoes_profissionais", {}))
            registros.append(registro)
        return pd.DataFrame(registros)

    elif "prospects" in nome_arquivo:
        for vaga_id, dados in json_data.items():
            for prospect in dados.get("prospects", []):
                prospect["vaga_id"] = vaga_id
                registros.append(prospect)
        return pd.DataFrame(registros)

    elif "vagas" in nome_arquivo:
        for vaga_id, dados in json_data.items():
            registro = {"vaga_id": vaga_id}
            registro.update(dados.get("informacoes_basicas", {}))
            registro.update(dados.get("perfil_vaga", {}))
            registro.update(dados.get("beneficios", {}))
            registros.append(registro)
        return pd.DataFrame(registros)

    return pd.DataFrame()  # fallback vazio

def extrair_json_dos_zips():
    for arquivo in os.listdir(INPUT_DIR):
        if arquivo.endswith(".zip"):
            zip_path = os.path.join(INPUT_DIR, arquivo)
            print(f"🧾 Processando: {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as z:
                for nome_json in z.namelist():
                    if nome_json.endswith(".json"):
                        print(f"📂 Extraindo JSON: {nome_json}")
                        with z.open(nome_json) as f:
                            try:
                                data = json.load(f)
                                df = processar_json_para_dataframe(nome_json, data)
                                if not df.empty:
                                    output_dir = os.path.join(OUTPUT_BASE_DIR, nome_json.split(".")[0])
                                    os.makedirs(output_dir, exist_ok=True)
                                    output_path = os.path.join(output_dir, f"{nome_json.split('.')[0]}.parquet")
                                    df.to_parquet(output_path, index=False)
                                    print(f"✅ Parquet salvo em: {output_path}")
                                else:
                                    print(f"⚠️ Nenhum dado processado em: {nome_json}")
                            except Exception as e:
                                print(f"❌ Erro ao processar {nome_json}: {e}")

if __name__ == "__main__":
    extrair_json_dos_zips()
