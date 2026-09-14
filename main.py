import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from train import mae
import pathlib

app=FastAPI()

path=pathlib.Path(__file__).parent.resolve()
model=joblib.load(path / "house_model.joblib")
features=joblib.load(path / "housefeatures.joblib")

#input Schema
class HouseFeature(BaseModel):
    MedInc: float = Field(gt=0, description="Media Income of Neighbourhood")
    HouseAge: float = Field(gt=0, description="Average Age of house in the block")
    AveRooms: float = Field(gt=0, description="Average number of rooms in house")
    AveBedrms: float = Field(gt=0, description="Average number of Bedrooms in house")
    Population: float = Field(gt=0, description="Total population of the area")
    AveOccup: float = Field(gt=0, description="Average number of people in a house")
    Latitude: float = Field(ge=32, le=42, description="Latitude of House")
    Longitude: float = Field(ge=-144, le=-118, description="Longitude of the house")

#home
@app.get("/")
def home():
    return{
        "message":"California house prediction api",
        "status":"running",
        "endpoint":"Send POST request to / predict"
    }

@app.get("/health")
def health():
    return{
        "status":"running",
        "model":"RandomForestRegressor",
        "features":features,
        "avg_error":mae
    }

@app.post("/predict")
def predict(house: HouseFeature):
    try:
      df = pd.DataFrame([{
        "MedInc": house.MedInc,
        "HouseAge": house.HouseAge,
        "AveRooms": house.AveRooms,
        "AveBedrms": house.AveBedrms,
        "Population": house.Population,
        "AveOccup": house.AveOccup,
        "Latitude": house.Latitude,
        "Longitude": house.Longitude,
      }])

      df = df[features]
      prediction = model.predict(df)[0]
      price_USD=prediction*100000

      return {"predicted_price": f"${price_USD:,.0f}",
            "Predicted Price sort":f"{prediction:,.0f} hundread thousand",
            "fidance_range":f"${(price_USD - mae*100000):,.0f} to {(price_USD + mae*100000):,.0f} "
            }
    except Exception as e:  
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction failed: {str(e)}"
            )

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)