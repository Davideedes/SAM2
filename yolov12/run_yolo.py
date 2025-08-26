from ultralytics import YOLO

# 1) Load your trained weights
model = YOLO("train_yolo_detect/weights/best.pt")  # or the path printed in your run

# 2) Run on images, a folder, or a video
results = model.predict(
    source="/Users/jaenix/Documents/SAM2/pipeline/resources/sequence_to_test_1",
    imgsz=1024,         
    conf=0.20,         
    iou=0.60,          
    device="mps",          
    save=True,        
    save_txt=True      
)