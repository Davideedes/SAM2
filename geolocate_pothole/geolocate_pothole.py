import json, math, numpy as np, cv2
from pyproj import CRS, Transformer, Geod
from lxml import etree as ET

# -------------------------------------------------------------
# helper: convert a world‑space vector to true‑north heading (°)
# -------------------------------------------------------------
def cam_heading(wvec, east, north):
    e = wvec.dot(east)
    n = wvec.dot(north)
    return (math.degrees(math.atan2(e, n)) + 360) % 360

# ──────────────────────────────────────────────
# 1 ▸ read metadata & annotation
# ──────────────────────────────────────────────
meta = json.load(open("lokstedt.json"))
root = ET.parse("annotations.xml").getroot()
poly_txt = root.xpath('.//polygon[@label=\"pothole\"]')[0].get("points")

# image size
W, H = meta["width"], meta["height"]

# ──────────────────────────────────────────────
# 2 ▸ pixel position = centroid of the polygon
# ──────────────────────────────────────────────
pts = np.array([[float(x), float(y)] for x, y in
                (p.split(",") for p in poly_txt.split(";"))])
u, v = pts.mean(axis=0)

# ──────────────────────────────────────────────
# 3a ▸ intrinsics and rotation (camera → world)
# ──────────────────────────────────────────────
f_ratio, k1, k2 = meta["camera_parameters"]
f_px  = f_ratio * max(W, H)
fx = fy = f_px
cx, cy = W / 2, H / 2

rvec = np.asarray(meta["computed_rotation"], dtype=float)
R_wc, _ = cv2.Rodrigues(rvec)      # world → camera
R_cw    = R_wc.T                   # camera → world

# normalised pixel coordinates (+X right, +Y down, +Z forward)
x_n = (u - cx) / fx
y_n = (v - cy) / fy
d_cam = np.array([x_n, y_n, 1.0])
d_cam /= np.linalg.norm(d_cam)

# local ENU basis at the camera centre
lon, lat = meta["computed_geometry"]["coordinates"]
lon_rad, lat_rad = map(math.radians, (lon, lat))
east  = np.array([-math.sin(lon_rad),  math.cos(lon_rad), 0.0])
north = np.array([-math.sin(lat_rad)*math.cos(lon_rad),
                  -math.sin(lat_rad)*math.sin(lon_rad),
                   math.cos(lat_rad)])
up = np.cross(east, north)
up /= np.linalg.norm(up)
# ------------------------------------------------------------------
# 3b ▸ convert camera‑frame vectors (ENU) to the global ECEF frame
# ------------------------------------------------------------------
# OpenSfM / Mapillary define the *world* axes as local ENU, not ECEF.
# Build the 3×3 matrix that converts ENU‑coords → ECEF‑coords
enu2ecef = np.column_stack((east, north, up))  # columns: E, N, U

# direction of the pixel‑ray in ENU
d_enu = R_cw @ d_cam
d_enu /= np.linalg.norm(d_enu)

# …and in ECEF (used for intersection and heading)
d_w = enu2ecef @ d_enu
d_w /= np.linalg.norm(d_w)

# camera optical axis (+Z) expressed in ENU then ECEF
z_enu = R_cw @ np.array([0, 0, 1])
z_ecef = enu2ecef @ z_enu
heading_deg = cam_heading(z_ecef, east, north)

# ──────────────────────────────────────────────
# 4 ▸ intersect the ray with the ground plane
# ──────────────────────────────────────────────
alt_cam = meta["altitude"]  # ellipsoidal altitude of camera (m)
ecef   = CRS.from_epsg(4978)
wgs84  = CRS.from_epsg(4979)
llh2ecef = Transformer.from_crs(wgs84, ecef, always_xy=True)
ecef2llh = Transformer.from_crs(ecef, wgs84, always_xy=True)

# camera centre in ECEF
ox, oy, oz = llh2ecef.transform(lon, lat, alt_cam)
o = np.array([ox, oy, oz])

# assume flat road – road surface is cam_height below the lens
cam_height = meta["computed_altitude"]  # camera height above ground (m)
dot = d_w.dot(up)
if abs(dot) < 1e-6:
    raise ValueError("Ray is ~parallel to the ground plane!")

t = -cam_height / dot
Px, Py, Pz = o + t * d_w

# back to lat/lon/alt
lon_p, lat_p, alt_p = ecef2llh.transform(Px, Py, Pz)

# ──────────────────────────────────────────────
# 5 ▸ output nicely
# ──────────────────────────────────────────────
print(f"Camera  @ {lat:.6f} N, {lon:.6f} E   heading {heading_deg:.1f}° true")
print(f"Pothole @ {lat_p:.6f} N, {lon_p:.6f} E  (±{abs(cam_height)*0.1:.1f} m)")

geod = Geod(ellps='WGS84')
fwd_az, back_az, dist_m = geod.inv(lon, lat, lon_p, lat_p)
print(f"Distance camera → pothole: {dist_m:.2f} m, bearing {fwd_az:.1f}° true")