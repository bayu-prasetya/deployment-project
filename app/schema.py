from typing import Optional, List
from pydantic import BaseModel

class InsuranceClaim(BaseModel):
    claim_number: int
    age_of_driver: int
    gender: str
    marital_status: Optional[float]
    safty_rating: int
    annual_income: int
    high_education_ind: int
    address_change_ind: int
    living_status: str
    zip_code: int
    claim_date: str
    claim_day_of_week: str
    accident_site: str
    past_num_of_claims: int
    witness_present_ind: Optional[float]
    liab_prct: int
    channel: str
    policy_report_filed_ind: int
    claim_est_payout: float
    age_of_vehicle: int
    vehicle_category: str
    vehicle_price: float
    vehicle_color: str
    vehicle_weight: float

class PredictionRequest(BaseModel):
    records: List[InsuranceClaim]