from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import warnings
import transformers
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

from predict import predict_xray, predict_vitals, predict_brain
from gradcam import generate_gradcam_xray, generate_gradcam_brain

app = FastAPI(title="AI Healthcare Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Healthcare Diagnosis API is running! ✅"}

@app.post("/predict-xray")
async def predict_xray_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = predict_xray(image_bytes)
    return {"status": "success", **result}

@app.post("/predict-brain")
async def predict_brain_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = predict_brain(image_bytes)
    if result.get("error"):
        return {"status": "error", "message": result["message"]}
    return {"status": "success", **result}

@app.post("/predict-vitals")
async def predict_vitals_endpoint(
    pregnancies: float = Form(...),
    glucose: float = Form(...),
    blood_pressure: float = Form(...),
    skin_thickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    diabetes_pedigree: float = Form(...),
    age: float = Form(...)
):
    vitals = [pregnancies, glucose, blood_pressure,
              skin_thickness, insulin, bmi,
              diabetes_pedigree, age]
    result = predict_vitals(vitals)
    return {"status": "success", **result}

@app.post("/gradcam-xray")
async def gradcam_xray_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = generate_gradcam_xray(image_bytes)
    return {"status": "success", **result}

@app.post("/gradcam-brain")
async def gradcam_brain_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = generate_gradcam_brain(image_bytes)
    return {"status": "success", **result}