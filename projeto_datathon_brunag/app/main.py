from fastapi import FastAPI
from app.schema import InputData
from app.predict import fazer_previsao

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    resultado = fazer_previsao(data.dict())
    return {"contratado": resultado}
