from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import warnings
import transformers
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

from predict import predict

app = FastAPI(title="AI Healthcare Diagnosis API")

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Healthcare Diagnosis API is running! ✅"}

@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(...),
    pregnancies: float = Form(...),
    glucose: float = Form(...),
    blood_pressure: float = Form(...),
    skin_thickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    diabetes_pedigree: float = Form(...),
    age: float = Form(...),
    symptoms: str = Form(...)
):
    # Read image
    image_bytes = await image.read()

    # Vitals list
    vitals = [
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi,
        diabetes_pedigree, age
    ]

    # Get prediction
    result = predict(image_bytes, vitals, symptoms)

    return {
        "status": "success",
        "diagnosis": result["diagnosis"],
        "risk_score": result["risk_score"],
        "probabilities": result["probabilities"]
    }