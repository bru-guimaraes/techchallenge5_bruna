#!/usr/bin/env python3
import json
import requests
import sys

# Endpoint da API
URL = "http://localhost:8000/predict"

# Payload que deveria ser NEGATIVO
payload = {
    "cliente": "Itaú",
    "nivel_profissional": "Júnior",
    "idioma_requerido": "Inglês",
    "eh_sap": False,
    "area_atuacao": "Suporte",
    "nivel_ingles": "baixo",
    "nivel_espanhol": "baixo",
    "nivel_academico": "medio",
    "conhecimentos_tecnicos": "Suporte básico"
}

print("=== PAYLOAD NEGATIVO ===")
print(json.dumps(payload, indent=2, ensure_ascii=False))

try:
    resp = requests.post(URL, json=payload, timeout=5)
    resp.raise_for_status()
except Exception as e:
    print(f"\nErro ao chamar {URL}: {e}")
    sys.exit(1)

data = resp.json()
print("\n=== RESPOSTA ===")
print(json.dumps(data, indent=2, ensure_ascii=False))

# Validação
if data.get("previsao") != 0:
    print("\n❌ Teste falhou: esperava previsao=0")
    sys.exit(1)
else:
    print("\n✅ Teste passou: previsao=0 (NEGATIVE) como esperado")
    sys.exit(0)
