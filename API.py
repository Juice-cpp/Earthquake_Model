from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

modelo = joblib.load("model.joblib")


@app.get("/")
def inicio():
    return {"mensagem": "API funcionando"}


@app.post("/predict")
def predict(dados: dict):
    valores = np.array(dados["features"]).reshape(1, -1)
    resultado = modelo.predict(valores)
    return {"predicao": resultado[0]}