import cv2
import easyocr
from fastapi import FastAPI, UploadFile, File
from typing import List
import numpy as np
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import base64

app = FastAPI()


reader = easyocr.Reader(['en'], gpu=True)


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

        
        _, img_encoded = cv2.imencode(".png", image)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")

        
        response_data = {"success": True, "detected_text": detected_text, "image_base64": img_base64}
        return JSONResponse(content=jsonable_encoder(response_data))
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
