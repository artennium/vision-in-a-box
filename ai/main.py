from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()
model = YOLO("yolov8n.pt")  # โหลดโมเดลขนาดเล็กที่รันเร็ว

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # อ่านไฟล์ภาพที่รับมา
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # รัน YOLO Inference
    results = model(image)
    
    predictions = []
    for r in results:
        for box in r.boxes:
            predictions.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            })
            
    return {"predictions": predictions}