import cv2

input_path = r"demo\data\gallery\01_dog.mp4"
output_path = r"demo\data\gallery\01_dog_short.mp4"
max_frames = 20

cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

count = 0
while count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    count += 1

cap.release()
out.release()
print(f"Fertig! {count} Frames gespeichert nach {output_path}")