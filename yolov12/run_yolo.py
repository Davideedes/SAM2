from ultralytics import YOLO

# 1) Load your trained weights
model = YOLO("yolov12/pothole_yolov12_google_colab_training/weights/best.pt")  # or the path printed in your run

# 2) Run on images, a folder, or a video
results = model.predict(
    source="pipeline/resources/input_pictures",
    imgsz=1024,         
    conf=0.20,         
    iou=0.60,          
    device="mps", #change to 0 for gpu or 1 for cpu
    save=True,        
    save_txt=True      
)