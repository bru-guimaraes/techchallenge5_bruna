from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Exemplo de modelo de entrada
class EntradaExemplo(BaseModel):
    nome: str
    idade: int

# Rota raiz
@app.get("/")
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}

# Rota de exemplo com POST
@app.post("/exemplo")
def exemplo_endpoint(dados: EntradaExemplo):
    return {
        "mensagem": f"Olá, {dados.nome}! Você tem {dados.idade} anos."
    }
