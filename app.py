import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Clasificador de Asuntos",
    description="API para clasificar expedientes institucionales con porcentaje de confianza"
)

# Cargar el modelo guardado al arrancar el servicio
modelo = joblib.load("clasificador_asuntos.pkl")

class PeticionLote(BaseModel):
    asuntos: List[str]

class ResultadoPrediccion(BaseModel):
    asunto: str
    categoria: str
    confianza: float

@app.get("/")
def home():
    return {"status": "ok", "mensaje": "API del clasificador funcionando correctamente"}

@app.post("/predecir", response_model=List[ResultadoPrediccion])
def predecir_asuntos(datos: PeticionLote):
    asuntos = datos.asuntos
    if not asuntos:
        return []

    # 1. Obtener la categoría ganadora
    predicciones = modelo.predict(asuntos)

    # 2. Obtener probabilidades y calcular el porcentaje más alto por fila
    probabilidades = modelo.predict_proba(asuntos)
    confianzas = np.max(probabilidades, axis=1) * 100

    # 3. Formatear la respuesta JSON
    salida = []
    for asunto, cat, conf in zip(asuntos, predicciones, confianzas):
        salida.append({
            "asunto": asunto,
            "categoria": str(cat),
            "confianza": round(float(conf), 2)
        })

    return salida
