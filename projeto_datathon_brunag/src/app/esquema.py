"""
app/esquema.py

Esquemas Pydantic para validação de entrada.
"""

from pydantic import BaseModel

class DadosEntrada(BaseModel):
    """Modelo de entrada para predição de candidatos."""

    cliente: str
    nivel_profissional: str
    idioma_requerido: str
    eh_sap: bool
    area_atuacao: str
    nivel_ingles: str
    nivel_espanhol: str
    formacao: str
    conhecimentos_tecnicos: str
