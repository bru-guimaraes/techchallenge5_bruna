import os
import shutil
import pandas as pd
from utils.extrair_json_de_zip import main as extract_main

def test_extraction(tmp_path, monkeypatch):
    # prepara um zip mínimo
    # aqui você deve criar dentro de tmp_path/data/raw_downloads/ um ZIP válido
    # com uma pasta .json para extrusion; ou simplesmente monkeypatchar o PATHs
    data_dir = tmp_path / "data"
    raw = data_dir / "raw_downloads"
    raw.mkdir(parents=True)
    # coloque ali um zip de teste ou simule o comportamento…
    # Para simplificar, vamos monkeypatchar a variável de path:
    monkeypatch.setenv("PATH_RAW", str(raw))
    monkeypatch.setenv("PATH_PARQUET_VAGAS", str(tmp_path / "out" / "vagas"))
    # executar (espera não lançar)
    extract_main()
    # checa que as pastas parquet foram criadas
    out_dir = tmp_path / "out" / "vagas"
    assert out_dir.exists()

    # opcional: leia um parquet de exemplo
    # df = pd.read_parquet(str(out_dir / "vagas.parquet"))
    # assert not df.empty
