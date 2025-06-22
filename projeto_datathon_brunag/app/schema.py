from pydantic import BaseModel

class InputData(BaseModel):
    cliente: str
    nivel_profissional: str
    idioma_requerido: str
    eh_sap: bool
    area_atuacao: str
    nivel_ingles: str
    nivel_espanhol: str
    formacao: str
    conhecimentos_tecnicos: str
