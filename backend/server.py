import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Disable TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for security if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
MODEL_PATH = r"C:\Users\KIM\Desktop\Project\Maize Disease Classification\models\model_1.keras"
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise RuntimeError("Failed to load model. Check the file path and model format.")

# Define the class names
CLASS_NAMES = ['blight' , 'common_rust', 'gray_spot', 'healthy' ]

# Disease-specific recommendations
RECOMMENDATIONS = {
    "blight": (
        "Blight detected:\nRemove infected leaves and destroy crop debris.\n"
        "Apply copper-based fungicides and avoid overhead irrigation.\n"
        "Ensure adequate spacing to reduce humidity and improve air circulation."
    ),
    "common_rust": (
        "Common Rust detected:\nUse resistant maize varieties where possible.\n"
        "Apply fungicides during early rust development stages.\n"
        "Rotate crops and remove volunteer maize plants to reduce pathogen carryover."
    ),
    "gray_spot": (
        "Gray spot detected:\nPractice crop rotation and remove infected residure.\n"
        "Avoid close planting to reduce leaf wetness.\n"
        "Use fungicides like strobilurins or trialozes when necessary."
    ),
    "healthy": (
        "Your maize leaf is healthy:\nKeep practicing good agricultural hygiene.\n"
        "Conduct regular inspections, and avoid unnecessary chemical use."
    )
}
@app.get("/")
async def root():
    return {"message": "Welcome to the Maize Disease Classification API! Check /docs for usage."}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")  # Ensure RGB format
        image = image.resize((224, 224))  # Resize to model input size
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image file: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)  # Expand dimensions for model input
        
        # Predict using the model
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        recommendation = RECOMMENDATIONS.get(predicted_class, "No recommendation available.").split('\n')
        
        return {"class": predicted_class, "confidence": confidence, "recommendation": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    import sys
    sys.argv = ["uvicorn", "backend.server:app", "--host", "127.0.0.1", "--port", "8080", "--reload"]
    uvicorn.main()
