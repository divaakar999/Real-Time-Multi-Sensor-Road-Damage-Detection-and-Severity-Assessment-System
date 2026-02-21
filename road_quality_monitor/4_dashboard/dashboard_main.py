"""
=======================================================================
STEP 4C - DASHBOARD: Main Streamlit Web Application
=======================================================================
Launch with:
    streamlit run 4_dashboard/app.py

Features:
  • 🎥  Live video feed with YOLOv8 bounding boxes
  • 🗺️  Interactive Folium map with color-coded markers
  • 📊  Real-time severity statistics with animated charts
  • 📄  Downloadable PDF and CSV damage reports
  • 🔴🟡🟢  Severity color coding throughout
  • 📸  Detection gallery / history

=======================================================================
"""

import sys
import json
import time
import threading
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

# ── Page configuration (MUST BE FIRST) ────────────────────────────────
try:
    st.set_page_config(
        page_title = "AI Road Quality Monitor",
        page_icon  = "🛣️",
        layout     = "wide",
        initial_sidebar_state = "expanded",
    )
except Exception:
    pass

# ── NumPy 2.x Safety Check ───────────────────────────────────────────
if np.__version__.startswith("2."):
    st.error(f"❌ **NumPy 2.x detected ({np.__version__})**")
    st.markdown(f"""
        **Critical Compatibility Issue:** Your current environment is running NumPy {np.__version__}. 
        This version is incompatible with the pre-compiled YOLOv8 and OpenCV binaries, 
        resulting in a **Black Screen** during detection.
        
        **Immediate Resolution:**
        1. Click the **three dots** ⋮ (top right) → **Settings** → **Reboot App**.
        2. Verify that the root `requirements.txt` contains `numpy<2.0.0`.
    """)
    st.stop()

# ── Safety Imports ─────────────────────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ── Add project root to path ───────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "3_detection"))
sys.path.insert(0, str(ROOT / "4_dashboard"))

# ── AI Model Cache ────────────────────────────────────────────────────
@st.cache_resource
def get_yolo_model(path):
    from ultralytics import YOLO
    return YOLO(str(path))

# ── Damage Classes ─────────────────────────────────────────────────────
ROAD_DAMAGE_CLASSES = {
    "pothole", "crack", "wear",
    "alligator crack", "longitudinal crack",
    "transverse crack", "depression", "raveling",
    "rutting", "bleeding", "damage",
}

# ── WebRTC Video Processor (Guarded) ───────────────────────────────────
if WEBRTC_AVAILABLE:
    class RoadDamageProcessor:
        def __init__(self, model, clf, gps, conf, iou, road_classes):
            self.model = model
            self.clf   = clf
            self.gps   = gps
            self.conf  = conf
            self.iou   = iou
            self.road_classes = road_classes
            self.new_detections = []

        def recv(self, frame):
            try:
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                
                # Run prediction
                results = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False)
                annotated = img.copy()
                SEVERITY_BGR = {"HIGH": (0, 0, 220), "MEDIUM": (0, 165, 255), "LOW": (50, 200, 50)}
                
                for res in results:
                    if not res.boxes: continue
                    for box in res.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cid = int(box.cls[0])
                        cf  = float(box.conf[0])
                        cn  = self.model.names.get(cid, f"class_{cid}")
                        
                        if cn.lower() in self.road_classes:
                            sev   = self.clf.classify((x1, y1, x2, y2), (h, w), cn, cf)
                            color = SEVERITY_BGR.get(sev, (255, 255, 255))
                            label = f"{cn} [{sev}] {cf:.2f}"
                            
                            # Log detection
                            self.new_detections.append({
                                "class": cn, "confidence": cf, "severity": sev,
                                "bbox": [x1, y1, x2, y2], "gps": self.gps.get_current(),
                                "timestamp": datetime.now().isoformat()
                            })
                        else:
                            color = (180, 180, 180)
                            label = f"{cn} {cf:.2f}"

                        if CV2_AVAILABLE:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated, label, (x1, max(y1-6, 14)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            except Exception as e:
                # Log to terminal for debugging
                print(f"ERROR in WebRTC recv: {e}")
                return frame
else:
    class RoadDamageProcessor:
        def __init__(self, *args, **kwargs):
            pass
        def recv(self, frame):
            return frame

# ── WebRTC Video Processor (Guarded) ───────────────────────────────────

# ── Startup Status ─────────────────────────────────────────────────────
startup_status = st.empty()
startup_status.info("🚀 System starting up...")

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global dark theme refinements ─────────────────────────────── */
:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2128;
    --border:        #30363d;
    --accent-blue:   #58a6ff;
    --accent-green:  #3fb950;
    --accent-orange: #d29922;
    --accent-red:    #f85149;
    --text-primary:  #e6edf3;
    --text-muted:    #8b949e;
}

/* Metric cards */
.metric-card {
    background:    var(--bg-card);
    border:        1px solid var(--border);
    border-radius: 12px;
    padding:       20px;
    text-align:    center;
    transition:    transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform:   translateY(-3px);
    box-shadow:  0 8px 24px rgba(0,0,0,0.4);
}
.metric-number {
    font-size:   2.8rem;
    font-weight: 800;
    line-height: 1;
    margin:      4px 0;
}
.metric-label {
    font-size:   0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color:       var(--text-muted);
}

/* Severity badges */
.badge-high   { background:#f85149; color:white; padding:3px 10px;
                border-radius:12px; font-size:0.75rem; font-weight:700; }
.badge-medium { background:#d29922; color:white; padding:3px 10px;
                border-radius:12px; font-size:0.75rem; font-weight:700; }
.badge-low    { background:#3fb950; color:white; padding:3px 10px;
                border-radius:12px; font-size:0.75rem; font-weight:700; }

/* Section headers */
.section-header {
    font-size:     1.3rem;
    font-weight:   700;
    color:         var(--text-primary);
    border-bottom: 2px solid var(--accent-blue);
    padding-bottom: 6px;
    margin:        16px 0 12px 0;
}

/* Live feed border */
.live-feed img {
    border: 3px solid var(--accent-blue);
    border-radius: 8px;
}

/* Detection log table */
.det-row-high   { background: rgba(248, 81, 73,  0.12); }
.det-row-medium { background: rgba(210, 153, 34, 0.12); }
.det-row-low    { background: rgba(63,  185, 80, 0.12); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

/* Status pill */
.status-live {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(63,185,80,0.15); border: 1px solid #3fb950;
    color: #3fb950; border-radius: 20px; padding: 4px 14px;
    font-size: 0.8rem; font-weight: 600;
    animation: pulse 2s infinite;
}
.status-stopped {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(139,148,158,0.15); border: 1px solid #8b949e;
    color: #8b949e; border-radius: 20px; padding: 4px 14px;
    font-size: 0.8rem; font-weight: 600;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "detections":        [],
        "is_detecting":      False,
        "frame_placeholder": None,
        "detection_count":   0,
        "high_count":        0,
        "medium_count":      0,
        "low_count":         0,
        "model_loaded":      False,
        "model":             None,
        "video_source":      0,
        "fps":               0.0,
        "last_frame":        None,
        "demo_mode":         False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ══════════════════════════════════════════════════════════════════════
#  HELPER: Load detections from JSON file
# ══════════════════════════════════════════════════════════════════════
def load_detections_from_file() -> list:
    """Load existing detections.json (from a previous detection session)."""
    det_path = ROOT / "detections.json"
    if det_path.exists():
        with open(det_path) as f:
            return json.load(f)
    return []


def generate_demo_detections(n: int = 30) -> list:
    """
    Generate realistic demo detections for testing the dashboard
    without a trained model or camera.

    Returns n synthetic detections around Bangalore, India.
    """
    import random, math
    classes  = ["pothole", "crack", "wear"]
    sevs     = ["HIGH", "HIGH", "MEDIUM", "MEDIUM", "MEDIUM", "LOW"]
    base_lat = 12.9716
    base_lon = 77.5946
    dets     = []

    for i in range(n):
        cls  = random.choice(classes)
        sev  = random.choice(sevs)
        conf = random.uniform(0.45, 0.97)
        angle= random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, 0.005)

        dets.append({
            "frame":      i * 5,
            "class":      cls,
            "confidence": round(float(conf), 3),
            "severity":   sev,
            "bbox":       [random.randint(100, 400), random.randint(100, 300),
                           random.randint(400, 640), random.randint(300, 480)],
            "gps": {
                "lat":       round(float(base_lat + dist * math.cos(angle)), 6),
                "lon":       round(float(base_lon + dist * math.sin(angle)), 6),
                "alt":       920.0,
                "speed":     round(float(random.uniform(20, 50)), 1),
                "source":    "simulated",
            },
            "timestamp": (datetime.now() - timedelta(seconds=n - i)).isoformat(),
        })

    return dets


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size:2.5rem;">🛣️</div>
        <div style="font-size:1.1rem; font-weight:800; color:#58a6ff;">Road Quality</div>
        <div style="font-size:0.85rem; color:#8b949e;">AI Monitoring System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Mode selector ────────────────────────────────────────────────
    st.markdown("### ⚙️ Configuration")

    mode = st.radio(
        "Dashboard Mode",
        ["🎥  Live Detection", "📂  Load Previous Session", "🎭  Demo Mode"],
        index=2,    # Default to demo mode (safe without model)
    )

    st.markdown("---")

    # ── Detection Settings ───────────────────────────────────────────
    st.markdown("### 🔧 Detection Settings")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value = 0.10, max_value = 0.90,
        value     = 0.40, step      = 0.05,
        help      = "Minimum confidence to show a detection"
    )

    iou_threshold = st.slider(
        "NMS IoU Threshold",
        min_value = 0.30, max_value = 0.80,
        value     = 0.45, step      = 0.05,
        help      = "Intersection over Union threshold for NMS"
    )

    st.markdown("---")

    # ── Map Settings ─────────────────────────────────────────────────
    st.markdown("### 🗺️  Map Settings")
    show_heatmap  = st.toggle("Heatmap Layer",  value=True)
    show_route    = st.toggle("GPS Route",       value=True)
    map_tile      = st.selectbox(
        "Map Style",
        ["CartoDB dark_matter", "CartoDB positron", "OpenStreetMap", "Stamen Terrain"],
        index=0,
    )

    st.markdown("---")

    # ── Video source ─────────────────────────────────────────────────
    st.markdown("### 📷 Video Source")
    src_type = st.radio("Source Type", ["Webcam (Local/Server)", "Browser Camera (Mobile)", "Video File", "RTSP Stream"], index=1)
    
    if src_type == "Browser Camera (Mobile)":
        video_source = "webrtc"
    elif src_type == "Webcam (Local/Server)":
        cam_idx = st.number_input("Camera Index", 0, 10, 0)
        video_source = cam_idx
    elif src_type == "Video File":
        uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        video_source = uploaded
    else:
        video_source = st.text_input("RTSP URL", "rtsp://192.168.1.100:554/stream")

    st.markdown("---")

    # ── About ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.75rem; color:#8b949e; text-align:center;">
        <strong>Final Year Project</strong><br/>
        AI-Based Road Quality Monitoring<br/>
        YOLOv8 + OpenCV + Streamlit<br/>
        <br/>
        v1.0.0 · Feb 2026
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════

# ── Page Header ───────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("""
    <h1 style="font-size:2rem; font-weight:900; margin:0; color:#e6edf3;">
        🛣️ AI Road Quality Monitor
    </h1>
    <p style="color:#8b949e; margin:4px 0 0 2px; font-size:0.9rem;">
        Real-time pothole & crack detection powered by YOLOv8
    </p>
    """, unsafe_allow_html=True)

with col_status:
    if st.session_state["is_detecting"]:
        st.markdown('<div class="status-live">⬤ LIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-stopped">⬤ STOPPED</div>', unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── Load detections based on mode ─────────────────────────────────────
try:
    if "🎭" in mode:
        if not st.session_state["detections"]:
            st.session_state["detections"] = generate_demo_detections(40)
        st.info("🎭 **Demo Mode** — Showing 40 synthetic detections. Train your model and switch to Live Detection.")

    elif "📂" in mode:
        loaded = load_detections_from_file()
        if loaded:
            st.session_state["detections"] = loaded
            st.success(f"✅ Loaded {len(loaded)} detections from previous session.")
        else:
            st.warning("⚠️  No previous detections found. Run the real-time detector first.")
            st.code("python 3_detection/realtime_detection.py --source 0")
except Exception as e:
    st.error(f"Error loading detections: {e}")

detections = st.session_state.get("detections", [])


# ══════════════════════════════════════════════════════════════════════
#  KPI METRIC CARDS
# ══════════════════════════════════════════════════════════════════════
total  = len(detections)
high   = sum(1 for d in detections if d.get("severity") == "HIGH")
medium = sum(1 for d in detections if d.get("severity") == "MEDIUM")
low    = sum(1 for d in detections if d.get("severity") == "LOW")
avg_conf = np.mean([d.get("confidence", 0) for d in detections]) if detections else 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-top:3px solid #58a6ff;">
        <div class="metric-label">Total Detected</div>
        <div class="metric-number" style="color:#58a6ff;">{total}</div>
        <div class="metric-label">defects</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-top:3px solid #f85149;">
        <div class="metric-label">High Severity</div>
        <div class="metric-number" style="color:#f85149;">{high}</div>
        <div class="metric-label">{high/max(total,1)*100:.0f}% of total</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-top:3px solid #d29922;">
        <div class="metric-label">Medium Severity</div>
        <div class="metric-number" style="color:#d29922;">{medium}</div>
        <div class="metric-label">{medium/max(total,1)*100:.0f}% of total</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card" style="border-top:3px solid #3fb950;">
        <div class="metric-label">Low Severity</div>
        <div class="metric-number" style="color:#3fb950;">{low}</div>
        <div class="metric-label">{low/max(total,1)*100:.0f}% of total</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card" style="border-top:3px solid #a5d6ff;">
        <div class="metric-label">Avg Confidence</div>
        <div class="metric-number" style="color:#a5d6ff;">{avg_conf:.0%}</div>
        <div class="metric-label">model accuracy</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  MAIN CONTENT: VIDEO FEED + MAP (side by side)
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Feed & Map", "📊 Analytics", "📋 Detection Log", "📁 Reports"])

with tab1:
    left_col, right_col = st.columns([1, 1])

    # ── LEFT: Video Feed ─────────────────────────────────────────────
    with left_col:
        st.markdown('<div class="section-header">📹 Video Feed</div>', unsafe_allow_html=True)

        frame_placeholder = st.empty()

        # ── Control buttons ──────────────────────────────────────────
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            start_btn = st.button(
                "▶ Start Detection",
                type      = "primary",
                width = "stretch",
                disabled  = st.session_state["is_detecting"],
            )
        with btn_col2:
            stop_btn = st.button(
                "⏹ Stop",
                width = "stretch",
                disabled  = not st.session_state["is_detecting"],
            )
        with btn_col3:
            clear_btn = st.button("🗑 Clear Data", width="stretch")

        if clear_btn:
            st.session_state["detections"] = []
            st.rerun()

        # ── Live detection loop ──────────────────────────────────────
        if start_btn:
            st.session_state["is_detecting"] = True

        if stop_btn:
            st.session_state["is_detecting"] = False

        # ── Show demo frame or live feed ─────────────────────────────
        if st.session_state["is_detecting"] and "🎥" in mode:
            # ── BRANCH: WebRTC (Mobile) vs Local ───────────────────────────
            if video_source == "webrtc":
                if not WEBRTC_AVAILABLE:
                    st.error("⚠️ Browser Camera (WebRTC) is not available. Please check dependencies.")
                    st.session_state["is_detecting"] = False
                else:
                    st.info("📱 **Browser Camera Active** — Detection is running on your device's stream.")
                    weights = ROOT / "weights" / "best.pt"
                    # Pre-load to avoid timeout in component factory
                    with st.spinner("🧠 Loading AI Model..."):
                        model_inst = get_yolo_model(weights)
                        clf_inst   = SeverityClassifier()
                        gps_inst   = GPSTagger()
                    
                    ctx = webrtc_streamer(
                        key="road-monitor-webrtc-v2", # Changed key for fresh reload
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration=RTCConfiguration({
                            "iceServers": [
                                {"urls": ["stun:stun.l.google.com:19302"]},
                                {"urls": ["stun:stun1.l.google.com:19302"]},
                                {"urls": ["stun:stun2.l.google.com:19302"]},
                                {"urls": ["stun:stun3.l.google.com:19302"]},
                                {"urls": ["stun:stun4.l.google.com:19302"]},
                            ]
                        }),
                        media_stream_constraints={"video": True, "audio": False},
                        video_processor_factory=lambda: RoadDamageProcessor(
                            model_inst, clf_inst, gps_inst,
                            conf_threshold, iou_threshold, ROAD_DAMAGE_CLASSES
                        ),
                        async_processing=True,
                    )
                    # Sync detections back to main state
                    if ctx and ctx.video_processor:
                        new_ones = list(ctx.video_processor.new_detections)
                        if new_ones:
                            st.session_state["detections"].extend(new_ones)
                            ctx.video_processor.new_detections = []
                            # Pulse limit
                            if len(st.session_state["detections"]) > 500:
                                st.session_state["detections"] = st.session_state["detections"][-500:]
            else:
                # ── BRANCH: Local Hardware / File ───────────────────────────
                # NOTE: The loop logic is at the very bottom of the script
                # to prevent blocking UI rendering.
                st.info("💡 **Ready to Start** — Live feed will appear below.")
                st.spinner("🔄 Initializing camera/stream...")

        else:
            # ── Static demo frame (IDLE) ────────────────────────────────
            if CV2_AVAILABLE:
                demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                demo_frame[:] = (30, 35, 45) # Dark blue-grey
                cv2.putText(demo_frame, "Road Monitor - IDLE",
                            (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (88, 166, 255), 2)
                cv2.putText(demo_frame, "Click 'Start Detection' to activate feed",
                            (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 148, 158), 1)

                # Draw sample bounding boxes for illustration
                sample_boxes = [
                    ((80,  180, 200, 280), "pothole [HIGH]",   (0, 0, 220)),
                    ((300, 100, 500, 200), "crack [MEDIUM]",   (0, 165, 255)),
                ]
                for (x1,y1,x2,y2), label, color in sample_boxes:
                    cv2.rectangle(demo_frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(demo_frame, label, (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                frame_placeholder.image(
                    cv2.cvtColor(demo_frame, cv2.COLOR_BGR2RGB),
                    channels            = "RGB",
                    width = "stretch",
                    caption             = "🎬 Dashboard active — Ready to start monitoring"
                )
            else:
                st.warning("⚠️ **OpenCV not available.** Visualization restricted.")
                st.info("The system is running in headless/restricted mode. Live video processing requires OpenCV.")
                frame_placeholder.info("📽️ Waiting for input source...")

    # ── RIGHT: Map ────────────────────────────────────────────────────
    with right_col:
        st.markdown('<div class="section-header">🗺️ Damage Location Map</div>', unsafe_allow_html=True)

        if detections:
            try:
                from map_component import build_damage_map
                import streamlit_folium as stf

                fmap = build_damage_map(
                    detections   = detections,
                    show_heatmap = show_heatmap,
                    show_route   = show_route,
                    tile_style   = map_tile,
                )
                stf.folium_static(fmap, width=660, height=460)

            except ImportError:
                # Fallback if streamlit-folium not installed
                st.warning("Install `streamlit-folium` for map: `pip install streamlit-folium`")
                lats = [d["gps"]["lat"] for d in detections if "gps" in d]
                lons = [d["gps"]["lon"] for d in detections if "gps" in d]
                df   = pd.DataFrame({"lat": lats, "lon": lons})
                st.map(df)
        else:
            st.info("📍 No detections yet. Start detection or switch to Demo Mode to see location data.")
            # Show blank world map region
            blank_df = pd.DataFrame({"lat": [12.9716], "lon": [77.5946]})
            st.map(blank_df)


with tab2:
    # ══════════════════════════════════════════════════════════════════
    #  ANALYTICS TAB
    # ══════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📊 Detection Analytics</div>', unsafe_allow_html=True)

    if not PLOTLY_AVAILABLE:
        st.error("Plotly is not installed. Please check requirements.txt.")
    elif not detections:
        st.info("No detections yet to analyze.")
    else:
        df = pd.DataFrame(detections)

        chart1, chart2 = st.columns(2)

        with chart1:
            # ── Severity Distribution Donut ────────────────────────────
            sev_counts = df["severity"].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            color_map = {"HIGH": "#f85149", "MEDIUM": "#d29922", "LOW": "#3fb950"}
            fig_pie = px.pie(
                sev_counts, values="count", names="severity",
                title="Severity Distribution",
                color="severity", color_discrete_map=color_map,
                hole=0.5,
            )
            fig_pie.update_layout(
                paper_bgcolor="#1c2128", plot_bgcolor="#1c2128",
                font_color="#e6edf3", title_font_size=14,
                legend={"orientation": "v", "x": 1.0, "y": 0.5},
                margin={"l": 0, "r": 0, "t": 40, "b": 0},
            )
            fig_pie.update_traces(textinfo="percent+value", textfont_color="white")
            st.plotly_chart(fig_pie, width="stretch")

        with chart2:
            # ── Damage Class Bar Chart ─────────────────────────────────
            cls_counts = df["class"].value_counts().reset_index()
            cls_counts.columns = ["class", "count"]
            fig_bar = px.bar(
                cls_counts, x="class", y="count",
                title="Detections by Damage Type",
                color="class",
                color_discrete_sequence=["#f85149", "#ffa94d", "#3fb950"],
                text="count",
            )
            fig_bar.update_layout(
                paper_bgcolor="#1c2128", plot_bgcolor="#1c2128",
                font_color="#e6edf3", title_font_size=14,
                showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(gridcolor="#30363d"),
                yaxis=dict(gridcolor="#30363d"),
            )
            fig_bar.update_traces(textposition="outside", textfont_color="#e6edf3")
            st.plotly_chart(fig_bar, width="stretch")

        chart3, chart4 = st.columns(2)

        with chart3:
            # ── Confidence Distribution ─────────────────────────────────
            fig_hist = px.histogram(
                df, x="confidence", nbins=20,
                title="Detection Confidence Distribution",
                color_discrete_sequence=["#58a6ff"],
            )
            fig_hist.add_vline(
                x=df["confidence"].mean(), line_dash="dash",
                line_color="#ffa94d",
                annotation_text=f"Avg: {df['confidence'].mean():.2f}",
                annotation_font_color="#ffa94d",
            )
            fig_hist.update_layout(
                paper_bgcolor="#1c2128", plot_bgcolor="#1c2128",
                font_color="#e6edf3", title_font_size=14,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(gridcolor="#30363d"),
                yaxis=dict(gridcolor="#30363d", title="Count"),
            )
            st.plotly_chart(fig_hist, width="stretch")

        with chart4:
            # ── Severity × Class Heatmap ────────────────────────────────
            cross = pd.crosstab(df["class"], df["severity"])
            for col in ["HIGH", "MEDIUM", "LOW"]:
                if col not in cross.columns:
                    cross[col] = 0
            cross = cross[["HIGH", "MEDIUM", "LOW"]]

            fig_heat = go.Figure(data=go.Heatmap(
                z         = cross.values,
                x         = cross.columns.tolist(),
                y         = cross.index.tolist(),
                colorscale= [[0, "#1c2128"], [0.5, "#d29922"], [1, "#f85149"]],
                text      = cross.values,
                texttemplate = "%{text}",
                textfont  = {"size": 14, "color": "white"},
            ))
            fig_heat.update_layout(
                title         = "Class × Severity Heatmap",
                paper_bgcolor = "#1c2128", plot_bgcolor="#1c2128",
                font_color    = "#e6edf3", title_font_size=14,
                margin        = dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_heat, width="stretch")

        # ── Time-series line chart ─────────────────────────────────────
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            st.markdown("##### 📈 Detections Over Time")
            df_t = df.copy()
            df_t["timestamp"] = pd.to_datetime(df_t["timestamp"], errors="coerce")
            df_t = df_t.dropna(subset=["timestamp"])
            if not df_t.empty:
                df_t["minute"] = df_t["timestamp"].dt.floor("1min")
                timeline = df_t.groupby(["minute","severity"]).size().reset_index(name="count")
                fig_line = px.line(
                    timeline, x="minute", y="count", color="severity",
                    color_discrete_map={"HIGH":"#f85149","MEDIUM":"#d29922","LOW":"#3fb950"},
                    title="Detection Frequency Over Time",
                    markers=True,
                )
                fig_line.update_layout(
                    paper_bgcolor="#1c2128", plot_bgcolor="#1c2128",
                    font_color="#e6edf3", title_font_size=14,
                    xaxis=dict(gridcolor="#30363d"),
                    yaxis=dict(gridcolor="#30363d"),
                    legend_title="Severity",
                )
                st.plotly_chart(fig_line, width="stretch")


with tab3:
    # ══════════════════════════════════════════════════════════════════
    #  DETECTION LOG TAB
    # ══════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📋 Detection Log</div>', unsafe_allow_html=True)

    if not detections:
        st.info("No detections yet.")
    else:
        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            sev_filter = st.multiselect(
                "Filter Severity", ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"]
            )
        with f2:
            cls_filter = st.multiselect(
                "Filter Class", ["pothole", "crack", "wear"],
                default=["pothole", "crack", "wear"]
            )
        with f3:
            min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

        # Filter detections
        filtered = [
            d for d in detections
            if d.get("severity") in sev_filter
            and d.get("class") in cls_filter
            and d.get("confidence", 0) >= min_conf
        ]

        st.caption(f"Showing {len(filtered)} of {len(detections)} detections")

        # Build display DataFrame
        rows = []
        reversed_filtered = list(filtered)[::-1]
        for d in reversed_filtered[:100]:   # Last 100
            gps = d.get("gps", {})
            sev = d.get("severity", "LOW")
            sev_badge = {
                "HIGH":   '🔴 HIGH',
                "MEDIUM": '🟡 MEDIUM',
                "LOW":    '🟢 LOW',
            }.get(sev, sev)
            rows.append({
                "Class":       d.get("class","?").title(),
                "Severity":    sev_badge,
                "Confidence":  f"{d.get('confidence',0):.1%}",
                "Latitude":    f"{gps.get('lat',0):.6f}",
                "Longitude":   f"{gps.get('lon',0):.6f}",
                "Speed km/h":  f"{gps.get('speed',0):.0f}",
                "Time":        d.get("timestamp","")[:19],
            })

        st.dataframe(
            pd.DataFrame(rows),
            width = "stretch",
            height              = 400,
        )


with tab4:
    # ══════════════════════════════════════════════════════════════════
    #  REPORTS TAB
    # ══════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📁 Generate Reports</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)

    with r1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem; margin-bottom:8px;">📄</div>
            <div style="font-size:1.1rem; font-weight:700; margin-bottom:4px;">PDF Report</div>
            <div style="color:#8b949e; font-size:0.85rem;">
                Professional inspection report with KPI cards, detection table,
                damage breakdown by class, and maintenance recommendations.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        route_name = st.text_input("Route / Road Name", "MG Road, Bangalore")
        inspector  = st.text_input("Inspector Name",    "AI Monitoring System")

        if st.button("📄 Generate PDF Report", width="stretch", type="primary"):
            if not detections:
                st.warning("No detections to report yet.")
            else:
                with st.spinner("Generating PDF..."):
                    try:
                        from report_generator import generate_pdf_report
                        pdf_bytes = generate_pdf_report(
                            detections  = detections,
                            route_name  = route_name,
                            inspector   = inspector,
                        )
                        ts = datetime.now().strftime("%Y%m%d_%H%M")
                        st.download_button(
                            label        = "⬇️ Download PDF Report",
                            data         = pdf_bytes,
                            file_name    = f"road_report_{ts}.pdf",
                            mime         = "application/pdf",
                            width = "stretch",
                        )
                        st.success("✅ PDF ready for download!")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        st.info("Install fpdf2: `pip install fpdf2`")

    with r2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem; margin-bottom:8px;">📊</div>
            <div style="font-size:1.1rem; font-weight:700; margin-bottom:4px;">CSV Report</div>
            <div style="color:#8b949e; font-size:0.85rem;">
                Machine-readable spreadsheet with all detection details, GPS coordinates,
                severity classifications, and repair priorities.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        if st.button("📊 Generate CSV Report", width="stretch"):
            if not detections:
                st.warning("No detections to export yet.")
            else:
                with st.spinner("Generating CSV..."):
                    try:
                        from report_generator import generate_csv_report
                        csv_bytes = generate_csv_report(detections)
                        ts = datetime.now().strftime("%Y%m%d_%H%M")
                        st.download_button(
                            label        = "⬇️ Download CSV",
                            data         = csv_bytes,
                            file_name    = f"road_detections_{ts}.csv",
                            mime         = "text/csv",
                            width = "stretch",
                        )
                        st.success("✅ CSV ready for download!")
                    except Exception as e:
                        st.error(f"CSV generation failed: {e}")

        st.markdown("<br/><br/>", unsafe_allow_html=True)

        st.markdown("##### 📌 Schedule Summary")
        priority_data = {
            "🔴 Urgent (< 48h)":    high,
            "🟡 Soon (1–4 weeks)":  medium,
            "🟢 Routine":           low,
        }
        for label, count in priority_data.items():
            col_l, col_r = st.columns([3, 1])
            col_l.write(label)
            col_r.write(f"**{count}** locations")

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8b949e; font-size:0.8rem; padding:10px 0;">
    🛣️ <strong>AI Road Quality Monitoring System</strong> · Final Year Project ·
    Built with YOLOv8, OpenCV, Streamlit & Folium
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  DETECTION BACKGROUND LOOP (Non-blocking UI Rendering)
# ══════════════════════════════════════════════════════════════════════
if st.session_state["is_detecting"] and "🎥" in mode:
    try:
        from ultralytics import YOLO
        from severity_classifier import SeverityClassifier
        from gps_tagger import GPSTagger

        weights = ROOT / "weights" / "best.pt"
        if not weights.exists():
            st.error("⚠️ No trained model found at `weights/best.pt`")
            st.session_state["is_detecting"] = False
        else:

            if video_source == "webrtc":
                # WebRTC handles its own loop in a child process/thread
                pass 
            else:
                cap_src = int(video_source) if str(video_source).isdigit() else video_source
                cap = cv2.VideoCapture(cap_src)
                
                if not cap.isOpened():
                    st.error(f"Cannot open video source: {video_source}")
                    st.session_state["is_detecting"] = False
                else:
                    model = get_yolo_model(weights)
                    clf = SeverityClassifier()
                    gps = GPSTagger()
                    
                    SEVERITY_BGR = {"HIGH": (0, 0, 220), "MEDIUM": (0, 165, 255), "LOW": (50, 200, 50)}
                    
                    while st.session_state["is_detecting"]:
                        ret, frame = cap.read()
                        if not ret: break

                        results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                        annotated = frame.copy()
                        new_dets = []

                        for res in results:
                            for box in (res.boxes or []):
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                cid = int(box.cls[0]); cf = float(box.conf[0])
                                cn = model.names.get(cid, f"class_{cid}")
                                
                                if cn.lower() in ROAD_DAMAGE_CLASSES:
                                    sev = clf.classify((x1, y1, x2, y2), frame.shape[:2], cn, cf)
                                    color = SEVERITY_BGR.get(sev, (255, 255, 255))
                                    label = f"{cn} [{sev}] {cf:.2f}"
                                    new_dets.append({
                                        "class": cn, "confidence": cf, "severity": sev,
                                        "bbox": [x1, y1, x2, y2], "gps": gps.get_current(),
                                        "timestamp": datetime.now().isoformat()
                                    })
                                else:
                                    color = (180, 180, 180); label = f"{cn} {cf:.2f}"

                                if CV2_AVAILABLE:
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(annotated, label, (x1, max(y1-6, 14)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        st.session_state["detections"].extend(new_dets)
                        if CV2_AVAILABLE:
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(annotated_rgb, channels="RGB", width="stretch")
                        
                        time.sleep(0.01)
                    cap.release()
    except Exception as e:
        st.error(f"Detection error: {e}")
