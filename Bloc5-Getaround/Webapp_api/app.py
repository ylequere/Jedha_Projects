# -*- coding: utf-8 -*-

import uvicorn
import pandas as pd 
from pydantic import BaseModel
from fastapi import FastAPI, Response, status
from typing import Union
from pickle import load

description = """
GetAround is the Airbnb for cars. You can rent cars from any person for a few hours to a few days!
The goal of this Getaround API is to predict the rental value of your car following criteria that you will provide.

## Machine-Learning-Prediction
 
* `/predict` provide your car details to get an estimation of your rental car price.

"""

tags_metadata = [
    {
        "name": "Machine-Learning-Prediction",
        "description": "Endpoints that provides an estimation of a car rental price"
    }
]

app = FastAPI(
    title="🚗 Getaround API 💰",
    description=description,
    version="0.1",
    contact={
        "name": "Getaround API - Yann Le Quéré",
        "url": "https://github.com/ylequere",
    },
    openapi_tags=tags_metadata
)

@app.get("/")
async def index():
    message = "This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"
    return message

class PredictionFeatures(BaseModel):
    model_key: str
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

@app.post("/predict", tags=["Machine-Learning-Prediction"])
async def predict(predictionFeatures: PredictionFeatures, response: Response):
    model = load(open('./regressor.dmp', 'rb'))
    preprocessor = load(open('./preprocessor.dmp', 'rb'))
    
    df = pd.DataFrame(dict(predictionFeatures), index=[0])

    # Transforming input values into features for regressor model
    try:
        X = preprocessor.transform(df)
    except ValueError as err:
        response.status_code = status.HTTP_406_NOT_ACCEPTABLE
        return {"Error":err.__str__()}
    
    prediction = model.predict(X)    

    response = {"prediction": round(prediction.tolist()[0], 2)}
    return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, access_log=True)