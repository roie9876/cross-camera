import cv2
import torch
import torchvision.transforms as T
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from PIL import Image
import streamlit as st
import time
from collections import deque
from torchvision.models import resnet50, ResNet50_Weights
from geopy.distance import geodesic

st.set_page_config(layout="wide")

# -----------------------------
# Model Loading (cached for performance)
# -----------------------------
@st.cache_resource
def load_models():
    detector = YOLO('yolo11n.pt')  # Use your YOLO model

    resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50_model.eval()

    embedding_model = torch.nn.Sequential(*list(resnet50_model.children())[:-1])  # Remove last layer
    embedding_model.eval()
    return detector, embedding_model

detector, embedding_model = load_models()

# Preprocessing for ResNet50
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Constants
KNOWN_HEIGHT = 1.7       # Avg person height (meters)
FOCAL_LENGTH = 800       # Camera focal length
HORIZONTAL_FOV = 60      # Camera horizontal field of view (degrees)

# UI: Time window selection for matching (in seconds)
TIME_WINDOW_SECONDS = st.slider("Set Time Window for Cross-Camera Comparison (Seconds)", 1, 600, 180, 1)

# UI: Expose the similarity threshold in the sidebar
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.70, 1.00, 0.80, 0.01)

# Camera location inputs
st.sidebar.subheader("Enter Camera Locations")
lat1 = st.sidebar.number_input("Camera 1 Latitude", value=32.0853, format="%.6f")
lon1 = st.sidebar.number_input("Camera 1 Longitude", value=34.7818, format="%.6f")
lat2 = st.sidebar.number_input("Camera 2 Latitude", value=32.0857, format="%.6f")
lon2 = st.sidebar.number_input("Camera 2 Longitude", value=34.7822, format="%.6f")

# Global variables for persistent tracking
past_detections = deque()  # Each entry: (timestamp, embedding, video_id)
# active_alerts stores active identities with reference embedding, last seen time, and set of video ids
active_alerts = {}         # Format: {alert_id: {"embedding": ref_emb, "last_seen": timestamp, "videos": set()}}
alert_id_counter = 1       # Unique counter for new ALERT_IDs

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
    """Computes real-world coordinates using camera location, estimated distance and horizontal offset."""
    if distance is None or offset is None:
        return None, None
    total_movement_meters = np.sqrt(distance**2 + offset**2)
    bearing = np.degrees(np.arctan2(offset, distance))  # Angle relative to the camera
    new_location = geodesic(meters=total_movement_meters).destination((camera_lat, camera_lon), bearing)
    return round(new_location.latitude, 6), round(new_location.longitude, 6)

def extract_embedding_and_crop(cropped_image):
    """Extracts a normalized feature embedding from an image crop."""
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
    """
    Compare the new embedding with active identities.
    If a match is found (cosine similarity exceeds an effective threshold), update and return that ALERT_ID.
    Otherwise, create a new ALERT_ID.
    When merging from a different camera, require a higher threshold (threshold + 0.05) to avoid false matches.
    """
    global active_alerts, alert_id_counter
    best_match_id = None
    best_similarity = -1
    for alert_id, data in active_alerts.items():
        ref_emb = data["embedding"]
        similarity = 1 - cosine(emb, ref_emb)
        # If the current detection is from a new camera relative to this identity, use a higher threshold
        effective_threshold = threshold + 0.05 if video_id not in data["videos"] else threshold
        # Debug: print similarity values and effective threshold
        print(f"Comparing embedding with alert_id {alert_id}: similarity = {similarity:.3f} (effective threshold = {effective_threshold:.3f})")
        if similarity > effective_threshold and similarity > best_similarity:
            best_similarity = similarity
            best_match_id = alert_id
    if best_match_id is not None:
        # Update the reference embedding (simple averaging for smoothing)
        active_alerts[best_match_id]["embedding"] = (active_alerts[best_match_id]["embedding"] + emb) / 2
        active_alerts[best_match_id]["last_seen"] = current_time
        active_alerts[best_match_id]["videos"].add(video_id)
        return best_match_id
    else:
        new_id = alert_id_counter
        alert_id_counter += 1
        active_alerts[new_id] = {"embedding": emb, "last_seen": current_time, "videos": {video_id}}
        return new_id

def clean_active_alerts(current_time, max_age=TIME_WINDOW_SECONDS):
    """Remove identities that haven't been seen within the max_age window."""
    global active_alerts
    active_alerts = {alert_id: data for alert_id, data in active_alerts.items()
                     if current_time - data["last_seen"] <= max_age}

def detect_frame(frame, video_id, camera_lat, camera_lon, conf_threshold=0.5):
    """
    Detect objects, estimate distance, and assign persistent ALERT_IDs
    to persons detected in the frame.
    """
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
            if int(cls) != 0:  # Only process persons
                continue

            x1, y1, x2, y2 = box.astype(int)
            crop = frame[y1:y2, x1:x2]
            emb = extract_embedding_and_crop(crop)
            if emb is None or np.isnan(emb).any() or np.linalg.norm(emb) == 0:
                continue

            # Embedding is normalized in extract_embedding_and_crop
            distance = estimate_distance(box)
            x_offset_meters = estimate_horizontal_offset(box, frame_width, distance) if distance else None
            real_lat, real_lon = compute_real_world_location(camera_lat, camera_lon, distance, x_offset_meters)

            # Get a persistent ALERT_ID for this detection (includes the video_id)
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
            past_detections.append((current_time, emb, video_id))

    # Clean up old detections and active identities
    past_detections = deque([d for d in past_detections if current_time - d[0] <= TIME_WINDOW_SECONDS])
    clean_active_alerts(current_time, max_age=TIME_WINDOW_SECONDS)

    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes, additional info, and ALERT_IDs on the frame.
       Only display the ALERT_ID if the identity has been seen in both videos.
    """
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = (0, 255, 0)  # Default green for detections
        display_alert = False
        alert_id = det.get("alert_id")
        # Only display the alert if this identity has been seen in more than one video
        if alert_id is not None and alert_id in active_alerts:
            if len(active_alerts[alert_id]["videos"]) > 1:
                display_alert = True
        if display_alert:
            color = (0, 0, 255)  # Red for cross-camera matched persons
            alert_text = f"ALERT_ID: {alert_id}"
            cv2.putText(frame, alert_text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if det["distance"] is not None:
            cv2.putText(frame, f"Dist: {det['distance']}m", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if det["x_offset_meters"] is not None:
            cv2.putText(frame, f"Offset: {det['x_offset_meters']}m", (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        if det["real_world_location"][0] is not None:
            cv2.putText(frame, f"Loc: {det['real_world_location']}", (x1, y1 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    # Convert from BGR to RGB before display
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# -----------------------------
# Streamlit UI and Video Processing
# -----------------------------
st.title("Live Cross-Camera Tracking")
video1_file = st.file_uploader("Upload Video 1", type=["mp4", "avi", "mov"])
video2_file = st.file_uploader("Upload Video 2", type=["mp4", "avi", "mov"])
start_button = st.button("Start Live Stream")

if start_button and video1_file and video2_file:
    # Save uploaded files to disk (cv2.VideoCapture requires a file path)
    with open("video1.mp4", "wb") as f1:
        f1.write(video1_file.read())
    with open("video2.mp4", "wb") as f2:
        f2.write(video2_file.read())

    # Open videos with FFmpeg backend (if available)
    cap1 = cv2.VideoCapture("video1.mp4", cv2.CAP_FFMPEG)
    cap2 = cv2.VideoCapture("video2.mp4", cv2.CAP_FFMPEG)

    if not cap1.isOpened():
        st.error("Error: Unable to open Video 1.")
    if not cap2.isOpened():
        st.error("Error: Unable to open Video 2.")

    col1, col2 = st.columns(2)
    frame_placeholder1 = col1.empty()
    frame_placeholder2 = col2.empty()

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Stop if one of the videos has finished
        if not ret1 or frame1 is None:
            st.warning("Video 1 has ended.")
            break
        if not ret2 or frame2 is None:
            st.warning("Video 2 has ended.")
            break

        detections1 = detect_frame(frame1, 1, lat1, lon1)
        detections2 = detect_frame(frame2, 2, lat2, lon2)

        frame_placeholder1.image(draw_detections(frame1, detections1),
                                 channels="RGB", use_container_width=True)
        frame_placeholder2.image(draw_detections(frame2, detections2),
                                 channels="RGB", use_container_width=True)

    cap1.release()
    cap2.release()
    st.write("Live stream ended.")