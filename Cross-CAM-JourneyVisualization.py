import os
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from PIL import Image
import streamlit as st
import time
from collections import deque
from torchvision.models import resnet50, ResNet50_Weights
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
import pydeck as pdk
import base64
import logging

# Configure logging to file
logging.basicConfig(
    filename='debug.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(layout="wide")

# Initialize journey log and alert images in session state
if "journey_log" not in st.session_state:
    st.session_state.journey_log = []  # Each event: {alert_id, timestamp, video_id, lat, lon}
if "alert_images" not in st.session_state:
    st.session_state.alert_images = {}  # Mapping from alert_id to base64-encoded image string

# -----------------------------
# Helper: Convert image to base64 string
# -----------------------------
def image_to_base64(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{jpg_as_text}"

# -----------------------------
# Model Loading (cached for performance)
# -----------------------------
@st.cache_resource
def load_models():
    detector = YOLO('yolo11n.pt')  # Use your YOLO model
    resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50_model.eval()
    # Remove final classification layer to obtain embeddings.
    embedding_model = torch.nn.Sequential(*list(resnet50_model.children())[:-1])
    embedding_model.eval()
    return detector, embedding_model

detector, embedding_model = load_models()

# Preprocessing for ResNet50
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Constants for geometry
KNOWN_HEIGHT = 1.7       # meters
FOCAL_LENGTH = 800       # pixels
HORIZONTAL_FOV = 60      # degrees

# -----------------------------
# Expose parameters in UI with descriptions
# -----------------------------
st.sidebar.markdown("### Re‑ID Tuning Parameters")
st.sidebar.write("**Time Window (seconds):** How long an alert remains active for matching.")
TIME_WINDOW_SECONDS = st.sidebar.slider("Time Window (seconds)", 1, 600, 180, 1)

st.sidebar.write("**Similarity Threshold:** The cosine similarity needed to consider two embeddings a match (lower is more lenient).")
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.70, 1.00, 0.80, 0.01)

st.sidebar.write("**Temporal Tolerance (seconds):** If a new detection occurs within this time of the last detection for an alert, the matching threshold is lowered.")
temporal_tolerance = st.sidebar.number_input("Temporal Tolerance (seconds)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

st.sidebar.write("**Minimum Detection Box Area (pixels):** Detections with a bounding box area smaller than this value are ignored (to remove unclear, pixelated detections).")
min_box_area = st.sidebar.slider("Minimum Detection Box Area (pixels)", 50, 10000, 300, step=50)

# -----------------------------
# Expose Camera Inclusion Selection
# -----------------------------
st.sidebar.markdown("### Cameras to Include in Journey Log")
st.sidebar.write("Select which cameras should contribute detections to the journey log.")
include_cameras = st.sidebar.multiselect("Cameras to include", options=[1, 2, 3, 4, 5], default=[1, 2])

# -----------------------------
# Expose Camera Locations
# -----------------------------
st.sidebar.markdown("### Camera Locations")
st.sidebar.write("These are the geographic coordinates for each camera.")
lat1 = st.sidebar.number_input("Camera 1 Latitude", value=31.3500, format="%.6f")
lon1 = st.sidebar.number_input("Camera 1 Longitude", value=34.3200, format="%.6f")
lat2 = st.sidebar.number_input("Camera 2 Latitude", value=31.3520, format="%.6f")
lon2 = st.sidebar.number_input("Camera 2 Longitude", value=34.3220, format="%.6f")
cam3_lat = st.sidebar.number_input("Camera 3 Latitude", value=31.3510, format="%.6f")
cam3_lon = st.sidebar.number_input("Camera 3 Longitude", value=34.3210, format="%.6f")
cam4_lat = st.sidebar.number_input("Camera 4 Latitude", value=31.3525, format="%.6f")
cam4_lon = st.sidebar.number_input("Camera 4 Longitude", value=34.3230, format="%.6f")
cam5_lat = st.sidebar.number_input("Camera 5 Latitude", value=31.3505, format="%.6f")
cam5_lon = st.sidebar.number_input("Camera 5 Longitude", value=34.3205, format="%.6f")

camera_coords = {
    1: (lat1, lon1),
    2: (lat2, lon2),
    3: (cam3_lat, cam3_lon),
    4: (cam4_lat, cam4_lon),
    5: (cam5_lat, cam5_lon)
}

# -----------------------------
# Default video paths from local folder "videos"
# -----------------------------
default_video_paths = {
    1: "videos/manwalk1-par1.mp4",
    2: "videos/manwalk1-par2.mp4",
    3: "videos/manwalk2.mp4",
    4: "videos/roie1.mp4",
    5: "videos/roie1.mp4"
}

# -----------------------------
# Define Area-of-Interest (AOI) for Camera 1 (in pixel coordinates)
# -----------------------------
aoi_polygon = Polygon([(100, 100), (500, 100), (500, 400), (100, 400)])
aoi_points = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.int32)

# -----------------------------
# Global variables for tracking
# -----------------------------
past_detections = deque()  # (timestamp, embedding, video_id)
active_alerts = {}         # {alert_id: {"embedding": ref_emb, "last_seen": timestamp, "videos": set(), "count": int}}
alert_id_counter = 1       # Unique counter

def estimate_distance(box):
    _, y1, _, y2 = box
    height_in_pixels = y2 - y1
    if height_in_pixels > 0:
        return round((KNOWN_HEIGHT * FOCAL_LENGTH) / height_in_pixels, 2)
    return None

def estimate_horizontal_offset(box, frame_width, distance):
    x1, _, x2, _ = box
    x_center = (x1 + x2) / 2
    x_offset_pixels = x_center - (frame_width / 2)
    if distance is not None:
        fov_radians = np.radians(HORIZONTAL_FOV)
        x_offset_meters = x_offset_pixels * (2 * np.tan(fov_radians / 2) * distance) / frame_width
        return round(x_offset_meters, 2)
    return None

def compute_real_world_location(camera_lat, camera_lon, distance, offset):
    if distance is None or offset is None:
        return None, None
    total_movement_meters = np.sqrt(distance**2 + offset**2)
    bearing = np.degrees(np.arctan2(offset, distance))
    new_location = geodesic(meters=total_movement_meters).destination((camera_lat, camera_lon), bearing)
    return round(new_location.latitude, 6), round(new_location.longitude, 6)

def extract_embedding_and_crop(cropped_image):
    image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)
    input_tensor = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        embedding = embedding_model(input_tensor)
    embedding = embedding.squeeze().numpy()
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

def assign_alert_id(emb, current_time, video_id, threshold):
    global active_alerts, alert_id_counter
    matching_ids = []
    for alert_id, data in active_alerts.items():
        similarity = 1 - cosine(emb, data["embedding"])
        if current_time - data["last_seen"] < temporal_tolerance:
            effective_threshold = max(threshold - 0.1, 0.5)
        else:
            effective_threshold = threshold + 0.05 if video_id not in data["videos"] else threshold
        logging.debug(f"Comparing embedding with alert_id {alert_id}: similarity = {similarity:.3f} (effective threshold = {effective_threshold:.3f})")
        if similarity > effective_threshold:
            matching_ids.append(alert_id)
    if not matching_ids:
        new_id = alert_id_counter
        alert_id_counter += 1
        active_alerts[new_id] = {"embedding": emb, "last_seen": current_time, "videos": {video_id}, "count": 1}
        return new_id
    if len(matching_ids) == 1:
        chosen_id = matching_ids[0]
        count = active_alerts[chosen_id].get("count", 1)
        old_emb = active_alerts[chosen_id]["embedding"]
        new_emb = (old_emb * count + emb) / (count + 1)
        active_alerts[chosen_id]["embedding"] = new_emb
        active_alerts[chosen_id]["last_seen"] = current_time
        active_alerts[chosen_id]["videos"].add(video_id)
        active_alerts[chosen_id]["count"] = count + 1
        return chosen_id
    chosen_id = min(matching_ids)
    total_count = 0
    merged_embeddings = []
    merged_videos = set()
    for aid in matching_ids:
        count = active_alerts[aid].get("count", 1)
        merged_embeddings.append(active_alerts[aid]["embedding"] * count)
        total_count += count
        merged_videos.update(active_alerts[aid]["videos"])
    merged_embeddings.append(emb)
    total_count += 1
    merged_embedding = np.sum(merged_embeddings, axis=0) / total_count
    active_alerts[chosen_id]["embedding"] = merged_embedding
    active_alerts[chosen_id]["last_seen"] = current_time
    active_alerts[chosen_id]["videos"].update(merged_videos)
    active_alerts[chosen_id]["videos"].add(video_id)
    active_alerts[chosen_id]["count"] = total_count
    for aid in matching_ids:
        if aid != chosen_id and aid in active_alerts:
            del active_alerts[aid]
    return chosen_id

def clean_active_alerts(current_time, max_age=TIME_WINDOW_SECONDS):
    global active_alerts
    active_alerts = {aid: data for aid, data in active_alerts.items() 
                     if current_time - data["last_seen"] <= max_age}

def detect_frame(frame, video_id, camera_lat, camera_lon, conf_threshold=0.5):
    global past_detections
    detections = []
    results = detector(frame)
    frame_width = frame.shape[1]
    current_time = time.time()
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, score, cls in zip(boxes, scores, classes):
            if score < conf_threshold:
                continue
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = box.astype(int)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < min_box_area:
                continue
            if video_id == 1:
                centroid = Point((x1 + x2) / 2, (y1 + y2) / 2)
                if not aoi_polygon.contains(centroid):
                    continue
            crop = frame[y1:y2, x1:x2]
            emb = extract_embedding_and_crop(crop)
            if emb is None or np.isnan(emb).any() or np.linalg.norm(emb) == 0:
                continue
            distance = estimate_distance(box)
            x_offset_meters = estimate_horizontal_offset(box, frame_width, distance) if distance else None
            real_lat, real_lon = compute_real_world_location(camera_lat, camera_lon, distance, x_offset_meters)
            alert_id = assign_alert_id(emb, current_time, video_id, threshold=similarity_threshold)
            detection = {
                "box": [x1, y1, x2, y2],
                "embedding": emb,
                "class": "person",
                "distance": distance,
                "x_offset_meters": x_offset_meters,
                "real_world_location": (real_lat, real_lon),
                "alert_id": alert_id,
                "timestamp": current_time,
                "video_id": video_id
            }
            detections.append(detection)
            if video_id in include_cameras:
                st.session_state.journey_log.append({
                    "alert_id": str(alert_id),
                    "timestamp": current_time,
                    "video_id": str(video_id),
                    "lat": real_lat,
                    "lon": real_lon
                })
            if alert_id not in st.session_state.alert_images:
                st.session_state.alert_images[alert_id] = image_to_base64(crop)
            past_detections.append((current_time, emb, video_id))
    past_detections = deque([d for d in past_detections if current_time - d[0] <= TIME_WINDOW_SECONDS])
    clean_active_alerts(current_time, max_age=TIME_WINDOW_SECONDS)
    return detections

def draw_detections(frame, detections, video_id=None):
    if video_id == 1:
        cv2.polylines(frame, [aoi_points], isClosed=True, color=(0,255,255), thickness=2)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = (0,255,0)
        alert_id = det.get("alert_id")
        if alert_id is not None:
            color = (0,0,255)
            alert_text = f"ALERT_ID: {alert_id}"
            cv2.putText(frame, alert_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if det["distance"] is not None:
            cv2.putText(frame, f"Dist: {det['distance']}m", (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        if det["x_offset_meters"] is not None:
            cv2.putText(frame, f"Offset: {det['x_offset_meters']}m", (x1, y2+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        if det["real_world_location"][0] is not None:
            cv2.putText(frame, f"Loc: {det['real_world_location']}", (x1, y1-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# -----------------------------
# Create a PathLayer for tracking the person over time.
# -----------------------------
def create_path_data(journey_df, selected_ids, color_dict):
    path_data = []
    for aid in selected_ids:
        alert_points = journey_df[journey_df["alert_id"] == aid].sort_values("timestamp")
        if not alert_points.empty:
            path = alert_points[["longitude", "latitude"]].values.tolist()
            path_data.append({
                "alert_id": str(aid),
                "path": path,
                "color": list(color_dict.get(aid, [255, 0, 0, 200]))
            })
    return path_data

path_layer = None  # Will be created later in the visualization tab

# -----------------------------
# Streamlit Tabs: Live Tracking & Journey Visualization
# -----------------------------
tab1, tab2 = st.tabs(["Live Tracking", "Journey Visualization"])

with tab1:
    st.header("Live Tracking")
    st.write("For each camera, select whether to use its video. If no video is uploaded, the default video from the 'videos' folder will be loaded if available.")
    
    use_cam1 = st.checkbox("Use Camera 1", value=True, key="use_cam1")
    file_cam1 = st.file_uploader("Camera 1 Video", type=["mp4", "avi", "mov"], key="v1")
    st.write("Default for Camera 1:", default_video_paths[1])
    
    use_cam2 = st.checkbox("Use Camera 2", value=True, key="use_cam2")
    file_cam2 = st.file_uploader("Camera 2 Video", type=["mp4", "avi", "mov"], key="v2")
    st.write("Default for Camera 2:", default_video_paths[2])
    
    use_cam3 = st.checkbox("Use Camera 3", value=True, key="use_cam3")
    file_cam3 = st.file_uploader("Camera 3 Video (Optional)", type=["mp4", "avi", "mov"], key="v3")
    st.write("Default for Camera 3:", default_video_paths[3])
    
    use_cam4 = st.checkbox("Use Camera 4", value=True, key="use_cam4")
    file_cam4 = st.file_uploader("Camera 4 Video (Optional)", type=["mp4", "avi", "mov"], key="v4")
    st.write("Default for Camera 4:", default_video_paths[4])
    
    use_cam5 = st.checkbox("Use Camera 5", value=True, key="use_cam5")
    file_cam5 = st.file_uploader("Camera 5 Video (Optional)", type=["mp4", "avi", "mov"], key="v5")
    st.write("Default for Camera 5:", default_video_paths[5])
    
    start_button = st.button("Start Live Stream")
    
    active_cameras = {}
    if start_button:
        for cam_id, (uploader, use_video) in enumerate(
            [(file_cam1, use_cam1), (file_cam2, use_cam2), (file_cam3, use_cam3), (file_cam4, use_cam4), (file_cam5, use_cam5)],
            start=1):
            if use_video:
                if uploader is not None:
                    video_bytes = uploader.read()
                    filename = f"video{cam_id}.mp4"
                    with open(filename, "wb") as f:
                        f.write(video_bytes)
                    cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        active_cameras[cam_id] = cap
                else:
                    default_path = default_video_paths.get(cam_id, "")
                    if default_path and os.path.exists(default_path):
                        cap = cv2.VideoCapture(default_path, cv2.CAP_FFMPEG)
                        if cap.isOpened():
                            active_cameras[cam_id] = cap
        
        if not active_cameras:
            st.error("No videos available!")
        else:
            columns = st.columns(len(active_cameras))
            placeholders = {cam_id: col.empty() for cam_id, col in zip(active_cameras.keys(), columns)}
            while any(cap.isOpened() for cap in active_cameras.values()):
                for cam_id, cap in list(active_cameras.items()):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        st.warning(f"Video {cam_id} has ended.")
                        cap.release()
                        del active_cameras[cam_id]
                        continue
                    cam_lat, cam_lon = camera_coords.get(cam_id, (None, None))
                    detections = detect_frame(frame, cam_id, cam_lat, cam_lon)
                    updated_frame = draw_detections(frame, detections, video_id=cam_id)
                    placeholders[cam_id].image(updated_frame, channels="RGB", use_container_width=True)
            st.write("Live stream ended.")

with tab2:
    st.header("Journey Visualization")
    st.write("This tab displays the journey log (with timestamps, camera IDs, and images) on a map along with an alert gallery (photo book).")
    if st.session_state.journey_log:
        df = pd.DataFrame(st.session_state.journey_log)
        # Convert alert_id to string
        df["alert_id"] = df["alert_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
        st.subheader("Journey Data Table")
        st.dataframe(df)
        
        # Build summary DataFrame per alert_id (with start time, average location, and aggregated camera IDs)
        summary_df = df.groupby("alert_id").agg({
            "timestamp": "min",
            "latitude": "mean",
            "longitude": "mean",
            "video_id": lambda x: ','.join(sorted(set(map(str, x))))
        }).reset_index()
        summary_df["alert_id"] = summary_df["alert_id"].astype(str)
        summary_df["timestamp_str"] = summary_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        unique_ids = sorted(summary_df["alert_id"].unique())
        
        st.subheader("Alert Gallery (Photo Book)")
        selected_alert_ids = []
        cols = st.columns(3)
        for i, row in summary_df.iterrows():
            col = cols[i % 3]
            # Checkbox label includes alert id, start time, and camera info.
            if col.checkbox(f"Alert {row['alert_id']}<br/>{row['timestamp_str']}<br/>Cam: {row['video_id']}", key=f"alert_{row['alert_id']}"):
                selected_alert_ids.append(row["alert_id"])
            # Check if an image exists before displaying it.
            img = st.session_state.alert_images.get(row["alert_id"], None)
            if img:
                col.image(img, caption=f"Alert {row['alert_id']}", width=100)
            else:
                col.write("No image")
        
        if not selected_alert_ids:
            selected_ids = unique_ids
        else:
            selected_ids = selected_alert_ids
        
        st.write("Selected Alert IDs:", selected_ids)
        
        # Filter journey data based on selected alerts.
        journey_df = df[df["alert_id"].isin(selected_ids)].sort_values("timestamp")
        st.subheader("Line Chart of Selected Journey(s)")
        st.line_chart(journey_df.set_index("timestamp")[["latitude", "longitude"]])
        
        # Create a color mapping for different alert_ids.
        color_palette = [
            [255, 0, 0, 200],
            [0, 255, 0, 200],
            [0, 0, 255, 200],
            [255, 255, 0, 200],
            [255, 0, 255, 200],
            [0, 255, 255, 200],
            [128, 0, 0, 200],
            [0, 128, 0, 200],
            [0, 0, 128, 200]
        ]
        color_dict = {aid: color_palette[i % len(color_palette)] for i, aid in enumerate(unique_ids)}
        journey_df["color"] = journey_df["alert_id"].apply(lambda aid: list(color_dict.get(aid, [255, 0, 0, 200])))
        
        summary_df["color"] = summary_df["alert_id"].apply(lambda aid: list(color_dict.get(aid, [255, 0, 0, 200])))
        summary_df["image"] = summary_df["alert_id"].apply(lambda aid: st.session_state.alert_images.get(aid, ""))
        summary_df = summary_df[summary_df["alert_id"].isin(selected_ids)]
        
        # Create a PathLayer for each alert track.
        def create_path_data(journey_df, selected_ids, color_dict):
            path_data = []
            for aid in selected_ids:
                alert_points = journey_df[journey_df["alert_id"] == aid].sort_values("timestamp")
                if not alert_points.empty:
                    path = alert_points[["longitude", "latitude"]].values.tolist()
                    path_data.append({
                        "alert_id": str(aid),
                        "path": path,
                        "color": list(color_dict.get(aid, [255, 0, 0, 200]))
                    })
            return path_data
        
        path_data = create_path_data(journey_df, selected_ids, color_dict)
        path_layer = pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color="color",
            width_scale=20,
            width_min_pixels=2,
            pickable=True,
        )
        
        # PyDeck layers for journey events, camera locations, and summary text.
        journey_layer = pdk.Layer(
            "ScatterplotLayer",
            data=journey_df,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius=50,
            radius_min_pixels=3,
            radius_max_pixels=5,
            pickable=True,
        )
        
        cam_data = []
        for cam_id, (lat, lon) in camera_coords.items():
            offset = 0.0005 * (cam_id - 1)
            cam_data.append({"name": f"Camera {cam_id}", "latitude": lat, "longitude": lon + offset})
        cam_df = pd.DataFrame(cam_data)
        camera_layer = pdk.Layer(
            "ScatterplotLayer",
            data=cam_df,
            get_position=["longitude", "latitude"],
            get_fill_color="[0, 0, 255, 255]",
            get_radius=100,
            radius_min_pixels=5,
            radius_max_pixels=8,
            pickable=True,
        )
        text_layer = pdk.Layer(
            "TextLayer",
            data=cam_df,
            pickable=False,
            get_position=["longitude", "latitude"],
            get_text="name",
            get_color="[0, 0, 0, 255]",
            get_size=16,
            get_alignment_baseline="'bottom'",
        )
        summary_text_layer = pdk.Layer(
            "TextLayer",
            data=summary_df,
            pickable=True,
            get_position=["longitude", "latitude"],
            get_text="alert_id + ': ' + timestamp_str + ' (Cam: ' + video_id + ')'",
            get_color="color",
            get_size=18,
            get_alignment_baseline="'center'",
        )
        
        tooltip_html = """
        <div style="background: rgba(0,0,0,0.7); padding:5px; border-radius:5px; color:white;">
            <b>Alert ID:</b> {alert_id}<br/>
            <b>Time:</b> {timestamp}<br/>
            <b>Camera(s):</b> {video_id}<br/>
            <img src="{image}" width="100"/>
        </div>
        """
        
        view_state = pdk.ViewState(
            latitude=31.35,
            longitude=34.32,
            zoom=14,
            pitch=0,
        )
        
        deck = pdk.Deck(
            layers=[journey_layer, path_layer, camera_layer, text_layer, summary_text_layer],
            initial_view_state=view_state,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={"html": tooltip_html, "style": {"color": "white"}}
        )
        st.pydeck_chart(deck)
    else:
        st.write("No journey data yet. Run live tracking to record events.")