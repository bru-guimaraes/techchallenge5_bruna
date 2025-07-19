import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from utils.carregar_dados_para_treino import carregar_dados_para_treino

# Carrega os dados
df = carregar_dados_para_treino()

# Cria coluna auxiliar: lista com a área
df['areas_atuacao'] = df['area_atuacao'].apply(lambda x: [x])

# Binariza a área de atuação
mlb = MultiLabelBinarizer()
area_atuacao_encoded = mlb.fit_transform(df['areas_atuacao'])

# Salva o binarizador
os.makedirs("model", exist_ok=True)
joblib.dump(mlb, "model/area_atuacao_mlb.joblib")

# Seleciona colunas
X_base = df.drop(columns=["contratado", "area_atuacao", "areas_atuacao", "codigo_profissional"])
X_dummies = pd.get_dummies(X_base)

# Junta com área binarizada
X_final = pd.concat([X_dummies.reset_index(drop=True), pd.DataFrame(area_atuacao_encoded, columns=mlb.classes_)], axis=1)
y = df["contratado"]

# Salva nomes das features
with open("model/features.json", "w", encoding="utf-8") as f:
    json.dump(list(X_final.columns), f, ensure_ascii=False, indent=2)

# Treina o modelo
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Salva o modelo
joblib.dump(modelo, "model/modelo_classificador.joblib")

print("✅ Treinamento finalizado e arquivos salvos na pasta 'model'")
