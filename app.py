from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import cv2
import easyocr
import numpy as np

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

reader = easyocr.Reader(['en'], gpu=True)

# Set Referrer-Policy header
@app.middleware("http")
async def add_referrer_policy_header(request, call_next):
    response = await call_next(request)
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

@app.post("/process_image/")
async def process_image(file: UploadFile):
    try:
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        result = reader.readtext(image)
        
        detected_text = []
        for detection in result:
            x_min, y_min = [int(cord) for cord in detection[0][0]]
            x_max, y_max = [int(cord) for cord in detection[0][2]]
            text = detection[1]
            detected_text.append({"text": text, "coordinates": [(x_min, y_min), (x_max, y_max)]})
        
        response_data = {"success": True, "detected_text": detected_text}
        return JSONResponse(content=jsonable_encoder(response_data))
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
