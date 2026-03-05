from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()
model = YOLO("yolov8n.pt")  # จะโหลดโมเดลขนาดเล็กที่สุดมาใช้

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # อ่านไฟล์รูปที่ส่งมา
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # สั่ง YOLO ประมวลผล
    results = model(img)
    
    # ดึงค่า JSON ออกมา (Class, Confidence, Box)
    predictions = []
    for r in results:
        for box in r.boxes:
            predictions.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })
            
    return {"predictions": predictions}
