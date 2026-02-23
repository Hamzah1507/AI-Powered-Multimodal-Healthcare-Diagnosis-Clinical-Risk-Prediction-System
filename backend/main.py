from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from auth import router as auth_router
from fastapi.responses import Response
import warnings
import transformers
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

from predict import predict_xray, predict_vitals, predict_brain
from gradcam import generate_gradcam_xray, generate_gradcam_brain
from report import generate_report
from database import predictions_collection
from crud import save_prediction, get_all_predictions, get_patient_history

app = FastAPI(title="AI Healthcare Diagnosis API")
app.include_router(auth_router, prefix)

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

@app.post("/generate-report")
async def generate_report_endpoint(
    image: UploadFile = File(...),
    module: str = Form(...),
    patient_name: str = Form(''),
    patient_id: str = Form(''),
    patient_age: str = Form(''),
    patient_gender: str = Form('Male'),
    xray_diagnosis: str = Form(''),
    xray_risk_score: int = Form(0),
    xray_prob_normal: float = Form(0),
    xray_prob_pneumonia: float = Form(0),
    vitals_diagnosis: str = Form(''),
    vitals_risk_score: int = Form(0),
    vitals_prob_no_diabetes: float = Form(0),
    vitals_prob_diabetes: float = Form(0),
    brain_diagnosis: str = Form(''),
    brain_risk_score: int = Form(0),
    brain_prob_glioma: float = Form(0),
    brain_prob_meningioma: float = Form(0),
    brain_prob_no_tumor: float = Form(0),
    brain_prob_pituitary: float = Form(0),
    heatmap: str = Form('')
):
    image_bytes = await image.read()
    patient = {
        'name': patient_name, 'id': patient_id,
        'age': patient_age, 'gender': patient_gender
    }

    xray_result = None
    vitals_result = None
    brain_result = None

    if module == 'xray' and xray_diagnosis:
        xray_result = {
            'diagnosis': xray_diagnosis,
            'risk_score': xray_risk_score,
            'probabilities': {
                'Normal': xray_prob_normal,
                'Pneumonia': xray_prob_pneumonia
            }
        }
        vitals_result = {
            'diagnosis': vitals_diagnosis,
            'risk_score': vitals_risk_score,
            'probabilities': {
                'No Diabetes': vitals_prob_no_diabetes,
                'Diabetes': vitals_prob_diabetes
            }
        }

    if module == 'brain' and brain_diagnosis:
        brain_result = {
            'diagnosis': brain_diagnosis,
            'risk_score': brain_risk_score,
            'probabilities': {
                'Glioma': brain_prob_glioma,
                'Meningioma': brain_prob_meningioma,
                'No Tumor': brain_prob_no_tumor,
                'Pituitary': brain_prob_pituitary
            }
        }

    pdf_bytes = generate_report(
        patient=patient,
        module=module,
        xray_result=xray_result,
        vitals_result=vitals_result,
        brain_result=brain_result,
        heatmap_b64=heatmap if heatmap else None,
        original_image_bytes=image_bytes
    )

    patient_name_clean = patient_name.replace(' ', '_') or 'Patient'
    filename = f"MediAI_Report_{patient_name_clean}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ─── Database & History Endpoints ───────────────────────────────────────────

@app.post("/save-prediction")
async def save_prediction_endpoint(data: dict):
    id = await save_prediction(data)
    return {"status": "success", "id": id}

@app.get("/history")
async def get_history():
    records = await get_all_predictions()
    return {"status": "success", "data": records}

@app.get("/history/{patient_id}")
async def get_patient_history_endpoint(patient_id: str):
    records = await get_patient_history(patient_id)
    return {"status": "success", "data": records}