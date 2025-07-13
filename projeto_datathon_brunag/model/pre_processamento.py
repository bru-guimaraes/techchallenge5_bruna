from unidecode import unidecode
import pandas as pd

# Dicionários de valores aceitos (expanda conforme necessário)
VALID_INGLES = {'baixo', 'medio', 'alto'}
VALID_ESPANHOL = {'baixo', 'medio', 'alto'}
# Expanda se quiser mais aliases para nivel_academico
ACADEMIC_MAP = {
    'medio': 'medio',
    'ensino medio': 'medio',
    'superior': 'superior',
    'ensino superior': 'superior',
    'pos': 'pos',
    'pós': 'pos',
    'pos graduacao': 'pos',
    'pós graduacao': 'pos',
    'pos-graduacao': 'pos',
    'pós-graduação': 'pos',
    'mestrado': 'mestrado',
    'doutorado': 'doutorado',
}

def normalize_value(val):
    if not isinstance(val, str):
        return val
    # Remove acentos, põe minúsculo, tira espaços extras
    return unidecode(val).strip().lower()

def preprocess_input(data: PredictRequest, feature_names):
    d = {k: normalize_value(v) for k, v in data.dict().items()}

    # Normaliza nivel_ingles
    if d['nivel_ingles'] not in VALID_INGLES:
        raise ValueError(f"nivel_ingles inválido: '{d['nivel_ingles']}'. Valores aceitos: {VALID_INGLES}")
    # Normaliza nivel_espanhol
    if d['nivel_espanhol'] not in VALID_ESPANHOL:
        raise ValueError(f"nivel_espanhol inválido: '{d['nivel_espanhol']}'. Valores aceitos: {VALID_ESPANHOL}")
    # Normaliza nivel_academico com alias/flexibilidade
    if d['nivel_academico'] not in ACADEMIC_MAP:
        raise ValueError(
            f"nivel_academico inválido: '{d['nivel_academico']}'. "
            f"Valores aceitos: {list(ACADEMIC_MAP.keys())}"
        )
    d['nivel_academico'] = ACADEMIC_MAP[d['nivel_academico']]

    # area_atuacao pode ser flexível, mas mantenha como está ou adicione lógica parecida se quiser mapeamento

    df = pd.DataFrame([d])
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    return df_aligned
