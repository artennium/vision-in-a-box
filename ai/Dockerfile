FROM python:3.9-slim

# ลง Library พื้นฐานที่ OpenCV เรียกใช้
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. ลงตัวที่ขนาดเล็กและลงง่ายก่อน (Docker จะ Cache บรรทัดนี้ไว้)
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# 2. ลงตัวที่เป็น CPU-only เพื่อลดขนาดไฟล์จาก 1GB เหลือแค่ประมาณ 100-200MB 
# (YOLO ปกติจะแถม Driver การ์ดจอมาด้วยซึ่งใหญ่มาก การใช้ --index-url นี้จะช่วยได้มหาศาล)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. ลง Ultralytics (YOLO) เป็นขั้นตอนสุดท้าย
RUN pip install --no-cache-dir ultralytics

COPY main.py .

# รัน Backend บน Port 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
