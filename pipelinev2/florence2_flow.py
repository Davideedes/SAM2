# Florence-2 : Crops ➜ BBoxe ➜ Overlay-PNG & JSON
from pathlib import Path, PurePath
import json, torch, matplotlib.pyplot as plt, matplotlib.patches as patches
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

__all__=["run_florence_on_folder"]

model_id="microsoft/Florence-2-base"
device="cuda" if torch.cuda.is_available() else "cpu"
model=None; processor=None   # lazy-load

def _load_model():
    global model,processor
    if model is None:
        model=AutoModelForCausalLM.from_pretrained(
            model_id,trust_remote_code=True,torch_dtype="auto").eval().to(device)
        processor=AutoProcessor.from_pretrained(model_id,trust_remote_code=True)

def _detect_pothole(img):
    _load_model()
    task="<OPEN_VOCABULARY_DETECTION>"
    inp=processor(text=task+" pothole", images=img,
                  return_tensors="pt").to(device,torch.float16)
    with torch.inference_mode():
        ids=model.generate(input_ids=inp["input_ids"],
                           pixel_values=inp["pixel_values"],
                           max_new_tokens=256,num_beams=3)
    txt=processor.batch_decode(ids,skip_special_tokens=False)[0]
    res=processor.post_process_generation(txt,task,img.size)
    return res.get(task,{})

def _plot_and_save(img,boxes,labels,out_png):
    fig,ax=plt.subplots(); ax.imshow(img); ax.axis("off")
    for b,l in zip(boxes,labels):
        x1,y1,x2,y2=b
        ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                                       ec="r",fc="none",lw=2))
        ax.text(x1,y1,l,color="w",bbox=dict(fc="r",alpha=.5,lw=0))
    fig.savefig(out_png,bbox_inches="tight",dpi=150); plt.close(fig)

def run_florence_on_folder(crop_folder:str):
    crop_dir=Path(crop_folder); out_dir=crop_dir/"florence_out"
    out_dir.mkdir(exist_ok=True)
    for p in sorted(crop_dir.glob("frame_*.png")):
        img=Image.open(p).convert("RGB")
        det=_detect_pothole(img)
        if not det.get("bboxes"): continue
        out_png=out_dir/(p.stem+"_pothole.png")
        out_json=out_dir/(p.stem+"_pothole.json")
        _plot_and_save(img,det["bboxes"],det["bboxes_labels"],out_png)
        json.dump(det,out_json.open("w"),indent=2)
        print(f"{p.name}: {len(det['bboxes'])} Treffer ➜ {out_png}")

