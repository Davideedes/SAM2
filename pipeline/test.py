import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile, os, shutil
from sam2.sam2_video_predictor import SAM2VideoPredictor

from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import requests
import copy
import torch
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  

# 1. Bilder als Frames laden und auf kleinste Auflösung bringen
frame_paths = [
    os.path.join("testbilder", "Schlagloch1.jpeg"),
    os.path.join("testbilder", "Schlagloch2.jpeg"),
    os.path.join("testbilder", "Schlagloch5.jpeg"),
    os.path.join("testbilder", "Schlagloch3.jpeg"),
    os.path.join("testbilder", "Schlagloch4.jpeg"),
    os.path.join("testbilder", "Schlagloch6.jpeg"),
]

# Lade alle Bilder und ermittle die kleinste Auflösung
imgs = [Image.open(p).convert("RGB") for p in frame_paths]
min_width = min(img.width for img in imgs)
min_height = min(img.height for img in imgs)
target_size = (min_width, min_height)

# Skaliere alle Bilder auf die kleinste Auflösung
frames = [np.array(img.resize(target_size, Image.LANCZOS)) for img in imgs]
n_frames = len(frames)
print(f"{n_frames} Frames geladen, skaliert auf {target_size}")

# 2. SAM2 Video Predictor laden
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")

# 3. Frames als JPEGs in temporären Ordner speichern
tmpdir = tempfile.mkdtemp()
for idx, frame in enumerate(frames):
    Image.fromarray(frame).save(os.path.join(tmpdir, f"{idx:05d}.jpg"))
inference_state = predictor.init_state(video_path=tmpdir)

# 4. Hilfsfunktionen für Visualisierung
def show_mask(mask, ax, color=None):
    if color is None:
        color = np.array([1.0, 0.0, 0.0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[np.array(labels) == 1]
    neg_points = coords[np.array(labels) == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='#a020f0', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

# 5. Interaktive Punktwahl für die ersten drei Bilder
clicked_points_per_frame = []
clicked_labels_per_frame = []

for frame_idx in range(4):
    clicked_points = []
    clicked_labels = []

    def on_click(event):
        if event.inaxes is not None:
            x, y = int(event.xdata), int(event.ydata)
            if event.button == 1:
                clicked_points.append([x, y])
                clicked_labels.append(1)
            elif event.button == 3:
                clicked_points.append([x, y])
                clicked_labels.append(0)
            else:
                return
            ax.clear()
            ax.imshow(frames[frame_idx])
            if clicked_points:
                input_points = np.array(clicked_points, dtype=np.float32)
                input_labels = np.array(clicked_labels, dtype=np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    points=input_points,
                    labels=input_labels,
                )
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax)
                show_points(input_points, input_labels, ax)
            ax.set_title(f"Bild {frame_idx+1}: {len(clicked_points)} Punkt(e) gesetzt (grün=+, lila=-)\nDrücke Enter für nächstes Bild")
            plt.axis('off')
            fig.canvas.draw()

    def on_key(event):
        if event.key == "enter":
            plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frames[frame_idx])
    ax.set_title(f"Klicke ins Bild {frame_idx+1}, um Punkte zu setzen (Enter = Weiter)")
    plt.axis('off')
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    clicked_points_per_frame.append(clicked_points)
    clicked_labels_per_frame.append(clicked_labels)

# 6. Nach Interaktion: Masken für die ersten drei Frames setzen
for frame_idx in range(3):
    if clicked_points_per_frame[frame_idx]:
        points = np.array(clicked_points_per_frame[frame_idx], dtype=np.float32)
        labels = np.array(clicked_labels_per_frame[frame_idx], dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
        )

print("Propagiere Maske auf das 4. Bild ...")
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# 7. Ergebnisse anzeigen
for idx in range(n_frames):
    plt.figure(figsize=(8, 6))
    plt.imshow(frames[idx])
    if 1 in video_segments.get(idx, {}):
        show_mask(video_segments[idx][1], plt.gca())
    plt.title(f"Bild {idx+1}/{n_frames}")
    plt.axis('off')
    plt.show()

shutil.rmtree(tmpdir)



# 8. Masks als kontextreiche Bounding-Box-Crops speichern  -----------------
out_dir = os.path.join("testbilder", "masken_ctx")
os.makedirs(out_dir, exist_ok=True)

context_ratio = 0.25   # 100 % extra Rand (= Boxfläche verdoppeln)
min_side      = 128   # garantiert mind. 256 px Breite/Höhe

for idx in range(n_frames):
    mask = video_segments.get(idx, {}).get(1)
    if mask is None:
        continue

    # --- 2-D Maske (Vereinigung aller Hypothesen) ------------------------
    if mask.ndim == 3:
        mask2d = np.any(mask, axis=0)
    else:
        mask2d = mask
    if not mask2d.any():
        continue

    # --- Bounding Box ----------------------------------------------------
    ys, xs = np.where(mask2d)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    box_w, box_h = x_max - x_min, y_max - y_min

    # --- Kontext + Mindestgröße -----------------------------------------
    # 1) prozentual erweitern
    pad_w = int(box_w * context_ratio / 2)   # /2: links + rechts
    pad_h = int(box_h * context_ratio / 2)   # /2: oben + unten

    # 2) Mindest-Seitenlänge absichern
    box_w_ext = box_w + 2 * pad_w
    box_h_ext = box_h + 2 * pad_h
    if box_w_ext < min_side:
        extra = (min_side - box_w_ext) // 2
        pad_w += extra
    if box_h_ext < min_side:
        extra = (min_side - box_h_ext) // 2
        pad_h += extra

    H, W = mask2d.shape
    x_min = max(0, x_min - pad_w)
    y_min = max(0, y_min - pad_h)
    x_max = min(W - 1, x_max + pad_w)
    y_max = min(H - 1, y_max + pad_h)

    # --- Crop & speichern ------------------------------------------------
    crop = frames[idx][y_min:y_max + 1, x_min:x_max + 1]
    out_path = os.path.join(out_dir, f"frame_{idx:05d}.png")
    Image.fromarray(crop).save(out_path)
    print(f"Gespeichert: {out_path}  ({crop.shape[1]}×{crop.shape[0]})")



model_id = 'microsoft/Florence-2-base'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)
image_path = r"./testbilder/masken_ctx/frame_00001.png"
image = Image.open(image_path).convert("RGB")
#5 von 6 kriegt er 

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer

def convert_to_od_format(data):  
    """  
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.  
  
    Parameters:  
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.  
  
    Returns:  
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.  
    """  
    # Extract bounding boxes and labels  
    bboxes = data.get('bboxes', [])  
    labels = data.get('bboxes_labels', [])  
      
    # Construct the output format  
    od_results = {  
        'bboxes': bboxes,  
        'labels': labels  
    }  
      
    return od_results  

def plot_bbox(image, data):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.show()  






task_prompt = '<OPEN_VOCABULARY_DETECTION>'
results = run_example(task_prompt, text_input="pothole")
print(results)
bbox_results  = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])

plot_bbox(image, bbox_results)





# task_prompt = '<DENSE_REGION_CAPTION>'
# results = run_example(task_prompt)
# print(results)


# plot_bbox(image, results['<DENSE_REGION_CAPTION>'])




# task_prompt = '<REGION_TO_DESCRIPTION>'
# results = run_example(task_prompt, text_input="<loc_52><loc_332><loc_932><loc_774>")
# print(results)





