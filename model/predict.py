import pickle
import pandas as pd
from typing import List, Dict

_model = None
EXPECTED_COLUMNS = ['claim_number','age_of_driver','gender',
                    'marital_status','safty_rating','annual_income',
                    'high_education_ind','address_change_ind','living_status',
                    'zip_code','claim_date','claim_day_of_week','accident_site',
                    'past_num_of_claims','witness_present_ind','liab_prct',
                    'channel','policy_report_filed_ind','claim_est_payout','age_of_vehicle',
                    'vehicle_category','vehicle_price','vehicle_color','vehicle_weight']

def get_model():
    global _model
    if _model is None:
        _model = pickle.load(open('model/artifacts/model_v1.sav','rb'))
    return _model

def validate_schema(df:pd.DataFrame):
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Invalid schema. Expected {EXPECTED_COLUMNS}")
    
def predict(data:List[Dict], proba:bool=False):
    df = pd.DataFrame(data)
    validate_schema(df)
    model = get_model()
    return model.predict_proba(df)[:,1] if proba else model.predict(df) 
