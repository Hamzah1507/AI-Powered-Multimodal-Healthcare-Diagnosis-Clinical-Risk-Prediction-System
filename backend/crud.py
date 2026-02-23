from database import predictions_collection
from models import PredictionRecord
from datetime import datetime

async def save_prediction(data: dict):
    data["timestamp"] = datetime.utcnow()
    result = await predictions_collection.insert_one(data)
    return str(result.inserted_id)

async def get_all_predictions():
    predictions = []
    async for record in predictions_collection.find().sort("timestamp", -1):
        record["_id"] = str(record["_id"])
        predictions.append(record)
    return predictions

async def get_patient_history(patient_id: str):
    predictions = []
    async for record in predictions_collection.find({"patient_id": patient_id}).sort("timestamp", -1):
        record["_id"] = str(record["_id"])
        predictions.append(record)
    return predictions