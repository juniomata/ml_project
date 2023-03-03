from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as util
import data_preparation as data_pipeline
import preprocess as preprocessing

config_data = util.load_config()

ohe_gender = util.pickle_load(config_data["ohe_gender_path"])
ohe_married = util.pickle_load(config_data["ohe_married_path"])
ohe_offer = util.pickle_load(config_data["ohe_offer_path"])
ohe_phone_service = util.pickle_load(config_data["ohe_phone_service_path"])
ohe_multiple_lines = util.pickle_load(config_data["ohe_multiple_lines_path"])
ohe_internet_service = util.pickle_load(config_data["ohe_internet_service_path"])
ohe_internet_type = util.pickle_load(config_data["ohe_internet_type_path"])
ohe_online_security = util.pickle_load(config_data["ohe_online_security_path"])
ohe_online_backup = util.pickle_load(config_data["ohe_online_backup_path"])
ohe_device_protection_plan = util.pickle_load(config_data["ohe_device_protection_plan_path"])
ohe_premium_tech_support = util.pickle_load(config_data["ohe_premium_tech_support_path"])
ohe_streaming_tv = util.pickle_load(config_data["ohe_streaming_tv_path"])
ohe_streaming_movies = util.pickle_load(config_data["ohe_streaming_movies_path"])
ohe_streaming_music = util.pickle_load(config_data["ohe_streaming_music_path"])
ohe_unlimited_data = util.pickle_load(config_data["ohe_unlimited_data_path"])
ohe_contract = util.pickle_load(config_data["ohe_contract_path"])
ohe_paperless_billing = util.pickle_load(config_data["ohe_paperless_billing_path"])
ohe_payment_method = util.pickle_load(config_data["ohe_payment_method_path"])

le_encoder = util.pickle_load(config_data["le_encoder_path"])
model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    gender : str
    age : int
    married : str
    number_of_referrals : int
    tenure_in_months : int
    offer : str
    phone_service : str
    avg_monthly_long_distance_charges : float
    multiple_lines : str
    internet_service : str
    internet_type : str
    avg_monthly_gb_download : float
    online_security : str
    online_backup : str
    device_protection_plan : str
    premium_tech_support : str
    streaming_tv : str
    streaming_movies : str
    streaming_music : str
    unlimited_data : str
    contract : str
    paperless_billing : str
    payment_method : str
    monthly_charge : float
    total_charges: float
    total_long_distance_charges : float
    total_revenue : float

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)

    # Convert dtype
    data = pd.concat(
        [
            data[config_data["predictors"][3:-6]],
            data[config_data["predictors"][:3]].astype(int),
            data[config_data["predictors"][-6:]].astype(float)
        ],
        axis = 1
    )

    # Check range data
    try:
        data_pipeline.check_data(data, config_data, True)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # Encoding categorical data
    data = preprocessing.ohe_transform(data, "gender", ohe_gender) 
    data = preprocessing.ohe_transform(data, "married", ohe_married) 
    data = preprocessing.ohe_transform(data, "offer", ohe_offer) 
    data = preprocessing.ohe_transform(data, "phone_service", ohe_phone_service) 
    data = preprocessing.ohe_transform(data, "multiple_lines", ohe_multiple_lines) 
    data = preprocessing.ohe_transform(data, "internet_service", ohe_internet_service) 
    data = preprocessing.ohe_transform(data, "internet_type", ohe_internet_type) 
    data = preprocessing.ohe_transform(data, "online_security", ohe_online_security) 
    data = preprocessing.ohe_transform(data, "online_backup", ohe_online_backup) 
    data = preprocessing.ohe_transform(data, "device_protection_plan", ohe_device_protection_plan) 
    data = preprocessing.ohe_transform(data, "premium_tech_support", ohe_premium_tech_support) 
    data = preprocessing.ohe_transform(data, "streaming_tv", ohe_streaming_tv) 
    data = preprocessing.ohe_transform(data, "streaming_movies", ohe_streaming_movies) 
    data = preprocessing.ohe_transform(data, "streaming_music", ohe_streaming_music) 
    data = preprocessing.ohe_transform(data, "unlimited_data", ohe_unlimited_data) 
    data = preprocessing.ohe_transform(data, "contract", ohe_contract) 
    data = preprocessing.ohe_transform(data, "paperless_billing", ohe_paperless_billing) 
    data = preprocessing.ohe_transform(data, "payment_method", ohe_payment_method) 

    # Predict data
    print(model_data["model_data"]["model_object"])
    y_pred = model_data["model_data"]["model_object"].predict(data)

    # Inverse tranform
    y_pred = list(le_encoder.inverse_transform(y_pred))[0] 

    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)