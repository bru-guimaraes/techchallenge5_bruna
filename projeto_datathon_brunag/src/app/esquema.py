from enum import Enum
from pydantic import BaseModel, Field

class NivelAcademico(str, Enum):
    medio = 'medio'
    superior = 'superior'
    pos = 'pos'
    mestrado = 'mestrado'
    doutorado = 'doutorado'

class DadosEntrada(BaseModel):
    cliente: str = Field(..., example="Bradesco", description="Nome do cliente ou empresa.")
    nivel_profissional: str = Field(..., example="Sênior", description="Nível profissional do candidato.")
    idioma_requerido: str = Field(..., example="Inglês", description="Idioma principal requerido para a vaga.")
    eh_sap: bool = Field(..., example=True, description="Se a vaga exige SAP (True/False).")
    area_atuacao: str = Field(..., example="Dados", description="Área de atuação principal da vaga.")
    nivel_ingles: str = Field(..., example="alto", description="Nível de inglês exigido (baixo/medio/alto).")
    nivel_espanhol: str = Field(..., example="baixo", description="Nível de espanhol exigido (baixo/medio/alto).")
    nivel_academico: NivelAcademico = Field(
        ..., example=NivelAcademico.pos,
        description="Nível acadêmico exigido: medio, superior, pos, mestrado ou doutorado."
    )
    conhecimentos_tecnicos: str = Field(
        ..., example="Python, SQL, Machine Learning",
        description="Principais conhecimentos técnicos exigidos pela vaga."
    )

    class Config:
        schema_extra = {
            "example": {
                "cliente": "Bradesco",
                "nivel_profissional": "Sênior",
                "idioma_requerido": "Inglês",
                "eh_sap": True,
                "area_atuacao": "Dados",
                "nivel_ingles": "alto",
                "nivel_espanhol": "baixo",
                "nivel_academico": "pos",
                "conhecimentos_tecnicos": "Python, SQL, Machine Learning"
            }
        }
