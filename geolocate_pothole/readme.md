Geolocate Pothole — README

A minimal Python utility that pin-points a pothole in real-world (WGS-84) coordinates from a single annotated street-level image and its photogrammetric metadata.

⸻

1. What the script does

geolocate_pothole.py follows the classic photogrammetry pipeline:

Step	Purpose	Key objects
1. Read inputs	Loads camera/photo metadata (lokstedt.json) and polygon annotation (annotations.xml).	json, lxml
2. Pixel → image plane	Finds the centroid of the pothole polygon and converts it to normalised pixel coordinates.	numpy
3. Image plane → world ray	Re-builds the camera intrinsics ( f in pixels ) and rotation, then projects the pixel as a 3-D ray in the local ENU frame; converts that to ECEF.	cv2, pyproj
4. Ray ∩ ground plane	Intersects the ray with a flat-road plane placed computed_altitude m below the lens.	trivial math
5. World point → GPS	Converts the intersection back to latitude, longitude and prints camera pose, pothole position, distance and bearing.	pyproj.Geod


⸻

2. Prerequisites

Package	Tested version
Python 3.9 +	
numpy	≥ 1.25
opencv-python	≥ 4.10
pyproj	≥ 3.6
lxml	≥ 5.2

Install them quickly:

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install numpy opencv-python pyproj lxml

3. Input files

File	Format / required keys
lokstedt.json	Mapillary/OpenSfM photo metadata export containing:• width, height (pixels)• camera_parameters → [f_ratio, k1, k2] (only f_ratio used)• computed_rotation (Rodrigues × 3)• computed_geometry.coordinates → [lon, lat] (deg)• altitude (ellipsoidal, m)• computed_altitude (camera height above ground, m)
annotations.xml	Pascal-VOC-like XML with at least one<polygon label="pothole" points="x1,y1;x2,y2;…"/> element in image pixel coords. Only the first pothole polygon is used; extend the script for multiples.

Place both files in the same folder as geolocate_pothole.py, or tweak the hard-coded filenames near the top of the script.

⸻

4. Running the script

python geolocate_pothole.py

Sample console output:

Camera  @ 53.598201 N, 9.945312 E   heading 127.4° true
Pothole @ 53.598123 N, 9.945587 E  (±0.2 m)
Distance camera → pothole: 21.36 m, bearing 126.9° true

What those numbers mean
	•	Camera @ – the photo centre (from metadata) and its true-north heading (derived from the optical axis).
	•	Pothole @ – geodetic position of the polygon centroid, with an uncertainty of ±10 % of camera height (rule-of-thumb).
	•	Distance / bearing – great-circle distance and forward azimuth from camera to pothole.

⸻

5. Customising / extending
	•	Multiple potholes – loop over every <polygon label="pothole"> element, computing each centroid in turn.
	•	Oblique ground – replace the flat plane assumption with a DEM or road mesh and intersect the ray by your favourite method (e.g. trimesh ray-casting).
	•	CLI arguments – wrap the hard-coded filenames in argparse to pass json/xml paths on the command-line.
	•	Error bars – propagate pixel-level annotation uncertainty and camera pose covariance if available.

⸻

6. Troubleshooting

Error message	Likely cause / fix
Ray is ~parallel to the ground plane!	Pixel lies very near the horizon. Pick a different pixel or ensure correct camera rotation.
KeyError: … while loading JSON	Check that lokstedt.json has all required keys (see Input files).
Unexpected location / heading	Verify that computed_rotation is CAM-to-world Rodrigues as exported by OpenSfM.
