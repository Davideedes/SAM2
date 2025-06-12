fetch_mpy_images — README

A tiny Python helper that pulls Mapillary still-images plus full JSON metadata to your disk.
Point it at one of:
	•	a single image key
	•	an entire sequence key
	•	or a bounding box

…and it neatly drops everything into timestamped folders under downloads/.

⸻

1 · What it actually does

Step	If you give…	It will…
1	Image key	Grab that one image and its metadata.
2	Sequence key	Page through the Graph API (/images?sequence_ids=) and fetch every frame (500 per request).
3	BBOX	Hit /images?bbox= and download up to limit images (default = 1) inside the extent.

Every file pair lands as

downloads/
└── image_<KEY> /  ⬅ single image
    ├─ <id>.jpg
    └─ <id>.json

or

downloads/
└── sequence_<KEY> / ⬅ many images
    ├─ <id1>.jpg
    ├─ <id1>.json
    ├─ <id2>.jpg
    └─ …

or

downloads/
└── bbox_<lon1>_<lat1>_<lon2>_<lat2> /
    └─ …

During the run you’ll see a friendly ▸ progress read-out and a ✅ summary with resolution, date, and lat/long.

⸻

2 · Prerequisites

Package	Tested with
Python	3.9+
requests	≥ 2.32
configparser	stdlib
(optional) shutil, pathlib, datetime, json	stdlib

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install requests


⸻

3 · Configuration (INI)

Create fetch_mpy_images.ini in the same folder (or a generic config.ini).
All keys live under one section:

[mapillary]
access_token =  ML...your-long-token  
# ONE of the next three:
image_key    =  1026761414815487
sequence_key =  800335321065526
bbox         =  9.9400,53.5960,9.9500,53.6000  ; minLon,minLat,maxLon,maxLat

limit        =  5           ; only for bbox queries
output_root  =  downloads   ; where to store sub-folders

If no token is set, the script will prompt you once interactively.

⸻

4 · Running

python fetch_mpy_images.py

The script:
	1.	Picks up the INI.
	2.	Figures out which query type you configured.
	3.	Creates a folder like downloads/sequence_<KEY>/.
	4.	Streams JPEGs + writes the exact Graph-API JSON beside each image.

⸻

5 · Troubleshooting

Message	Why / Fix
No config.ini…	The INI is missing or in the wrong folder.
❌  No bbox configured…	You left all three query options blank.
No imagery found	Wrong key or the bbox really is empty – check with the Mapillary web viewer.
requests.exceptions.*	Network hiccup; run again, Mapillary is usually resilient.


⸻

6 · Extending / hacking
	•	Command-line flags – wrap argparse around IMAGE_KEY, SEQUENCE_KEY, etc., to override the INI on the fly.
	•	Original resolution – swap thumb_original_url for url in FIELDS to fetch the full-res images (be prepared for big downloads).
	•	Parallel downloads – drop concurrent.futures.ThreadPoolExecutor around the request loop for a speed boost.
	•	EXIF embedding – write the lat/long into the JPEGs with piexif so they geo-tag in viewers.

⸻
