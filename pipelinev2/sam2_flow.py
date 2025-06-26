# SAM-2 : Bilder ➜ Masken ➜ kontextreiche Crops
import os, shutil, tempfile, numpy as np, torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor

__all__ = ["run_sam2_pipeline"]

def load_and_resize(folder:str)->list[np.ndarray]:
    imgs = [Image.open(os.path.join(folder,f)).convert("RGB")
            for f in sorted(os.listdir(folder))
            if f.lower().endswith((".jpg",".jpeg",".png"))]
    if not imgs: raise RuntimeError("Keine Bilder gefunden!")
    target = (min(i.width  for i in imgs),
              min(i.height for i in imgs))
    return [np.array(i.resize(target, Image.LANCZOS)) for i in imgs]

def interactive_seg(frames, predictor, n_click_frames=4, obj_id=1):
    """Returns dict{frame_idx:mask}  after user clicks."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    tmp = {"pts":[],"lbl":[]}
    inf = predictor.init_state(video_path=_write_tmp(frames))
    out = {}

    def _onclick(ev,idx):
        if ev.inaxes is None: return
        tmp["pts"].append([ev.xdata, ev.ydata])
        tmp["lbl"].append(1 if ev.button==1 else 0)
        _redraw(idx)

    def _redraw(i):
        ax.cla(); ax.imshow(frames[i]); ax.axis("off")
        if tmp["pts"]:
            pts, lbl = np.array(tmp["pts"],np.float32), np.array(tmp["lbl"])
            _,_,log = predictor.add_new_points_or_box(
                inf, i, obj_id, pts, lbl)
        mask2D = (log[0] > 0).cpu().squeeze(0).numpy()   # (H,W)
        ax.imshow(mask2D, alpha=.4, cmap="Reds")
        ax.scatter(pts[:,0],pts[:,1],c=["g"if l else "purple"for l in lbl])
        fig.canvas.draw_idle()

    def _accept(ev,i):
        if tmp["pts"]:
            predictor.add_new_points_or_box(
                inf,i,obj_id,np.array(tmp["pts"],np.float32),
                np.array(tmp["lbl"]))
        plt.close()

    # --- click-loop ----------------------------------------------
    for i in range(n_click_frames):
        import matplotlib.pyplot as plt
        tmp["pts"],tmp["lbl"]=[],[]
        fig,ax=plt.subplots(figsize=(8,6)); _redraw(i)
        fig.canvas.mpl_connect("button_press_event",
                               lambda e,idx=i:_onclick(e,idx))
        btn_ax=fig.add_axes([0.8,0.01,0.18,0.05])
        Button(btn_ax,"Weiter").on_clicked(lambda e,idx=i:_accept(e,idx))
        plt.show()

    # --- propagate ------------------------------------------------
    for fi,ids,logs in predictor.propagate_in_video(inf):
        out[fi]={oid:(logs[k]>0).cpu().numpy()
                 for k,oid in enumerate(ids)}
    return out

def save_crops(frames,masks,out_dir,ctx=0.25,min_side=128,obj_id=1):
    os.makedirs(out_dir,exist_ok=True)
    for fi,frame in enumerate(frames):
        mask=masks.get(fi,{}).get(obj_id)
        if mask is None: continue
        m2 = np.any(mask,0) if mask.ndim==3 else mask
        if not m2.any(): continue
        ys,xs=np.where(m2); xmin,xmax=xs.min(),xs.max()
        ymin,ymax=ys.min(),ys.max()
        w,h=xmax-xmin,ymax-ymin
        padw,padh=int(w*ctx/2),int(h*ctx/2)
        w_ext,h_ext=w+2*padw,h+2*padh
        if w_ext<min_side: padw+=(min_side-w_ext)//2
        if h_ext<min_side: padh+=(min_side-h_ext)//2
        H,W=m2.shape
        xmin,ymin=max(0,xmin-padw),max(0,ymin-padh)
        xmax,ymax=min(W-1,xmax+padw),min(H-1,ymax+padh)
        crop=frame[ymin:ymax+1,xmin:xmax+1]
        Image.fromarray(crop).save(
            os.path.join(out_dir,f"frame_{fi:05d}.png"))
    print(f"{len(os.listdir(out_dir))} Crops gespeichert ➜ {out_dir}")

def _write_tmp(frames):
    tmp=tempfile.mkdtemp()
    for i,f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(tmp,f"{i:05d}.jpg"))
    return tmp

def run_sam2_pipeline(img_folder:str, crop_folder:str,
                      n_click_frames:int=4)->None:
    frames = load_and_resize(img_folder)
    predictor=SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
    masks = interactive_seg(frames,predictor,n_click_frames)
    save_crops(frames,masks,crop_folder)
