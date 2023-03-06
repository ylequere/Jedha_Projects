# -*- coding: utf-8 -*-
import requests 
import json

def test_prediction():

    payload = {
        "model_key": "Toyota"
        ,"mileage": 19633
        ,"engine_power": 110
        ,"fuel": "diesel"
        ,"paint_color": "grey"
        ,"car_type": "van"
        ,"private_parking_available": False
        ,"has_gps": True
        ,"has_air_conditioning": False
        ,"automatic_car": False
        ,"has_getaround_connect": False
        ,"has_speed_regulator": False
        ,"winter_tires": True
    }

    r = requests.post(
        "https://getaround-api-ylequere.herokuapp.com/predict",
        data=json.dumps(payload)
    )

    response = r
    print(response)
    print(r.text)

def test_prediction_bad_input():

    payload = {
        "model_key": "Tayoto"
        ,"mileage": 19633
        ,"engine_power": 110
        ,"fuel": "diesel"
        ,"paint_color": "grey"
        ,"car_type": "van"
        ,"private_parking_available": False
        ,"has_gps": True
        ,"has_air_conditioning": False
        ,"automatic_car": False
        ,"has_getaround_connect": False
        ,"has_speed_regulator": False
        ,"winter_tires": True
    }

    r = requests.post(
        "https://getaround-api-ylequere.herokuapp.com/predict",
        data=json.dumps(payload)
    )

    response = r
    print(response)
    print(r.text)


test_prediction()
test_prediction_bad_input()