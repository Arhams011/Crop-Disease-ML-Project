import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="wide")

@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, scaler, class_names

def preprocess_image(image, target_size=(256, 256)):
    img_resized = cv2.resize(image, target_size)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_resized = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([15, 30, 30]), np.array([35, 255, 255]))
    mask_dark = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([85, 255, 150]))
    mask = cv2.bitwise_or(mask_green, mask_yellow)
    mask = cv2.bitwise_or(mask, mask_dark)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask_filtered = np.zeros_like(mask)
        cv2.drawContours(mask_filtered, [largest], -1, 255, -1)
        mask = mask_filtered
    segmented = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    return img_resized, hsv, mask, segmented

def extract_color_features(hsv_image, mask, bins=32):
    features = []
    for i in range(3):
        hist = cv2.calcHist([hsv_image], [i], mask, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)

def extract_glcm_features(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    gray_masked = img_as_ubyte(gray_masked / 255.0) if gray_masked.max() > 1 else img_as_ubyte(gray_masked)
    try:
        glcm = graycomatrix(gray_masked, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        return np.array([graycoprops(glcm, p).mean() for p in ["contrast", "correlation", "energy", "homogeneity", "dissimilarity"]])
    except:
        return np.array([0, 0, 0, 0, 0])

def extract_shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.array([0, 0, 0, 0, 0])
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect = w / h if h > 0 else 0
    circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solid = area / hull_area if hull_area > 0 else 0
    img_area = mask.shape[0] * mask.shape[1]
    return np.array([area/img_area, perim/(2*(mask.shape[0]+mask.shape[1])), aspect, circ, solid])

def extract_all_features(image):
    orig, hsv, mask, seg = preprocess_image(image)
    return np.concatenate([extract_color_features(hsv, mask), extract_glcm_features(seg, mask), extract_shape_features(mask)])

def calculate_severity_score(image):
    orig, hsv, mask, _ = preprocess_image(image)
    total_leaf = np.sum(mask > 0)
    if total_leaf == 0:
        return 0, "No Leaf", "gray", None
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    diseased = cv2.inRange(hsv, np.array([5, 40, 50]), np.array([25, 255, 255]))
    necrotic = cv2.inRange(hsv, np.array([5, 20, 0]), np.array([25, 255, 80]))
    yellow = cv2.inRange(hsv, np.array([20, 50, 120]), np.array([35, 255, 255]))
    shadow = np.zeros_like(mask)
    shadow[(v < 60) & (s < 40)] = 255
    total_diseased = cv2.bitwise_or(diseased, necrotic)
    total_diseased = cv2.bitwise_or(total_diseased, yellow)
    total_diseased = cv2.bitwise_and(total_diseased, cv2.bitwise_not(shadow))
    total_diseased = cv2.bitwise_and(total_diseased, mask)
    total_diseased = cv2.morphologyEx(total_diseased, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    score = (np.sum(total_diseased > 0) / total_leaf) * 100
    if score < 5:
        return score, "Healthy/Minimal", "green", total_diseased
    elif score < 15:
        return score, "Mild", "yellow", total_diseased
    elif score < 35:
        return score, "Moderate", "orange", total_diseased
    else:
        return score, "Severe", "red", total_diseased

def format_class_name(name):
    parts = name.split("___")
    if len(parts) == 2:
        return f"{parts[0].replace(chr(95), chr(32))} - {parts[1].replace(chr(95), chr(32))}"
    return name.replace("_", " ")

def get_treatment(disease, severity):
    d = disease.lower()
    treatments = {
        "early_blight": {
            "desc": "Fungal disease (Alternaria) causing dark spots with concentric rings",
            "organic": ["Remove infected leaves", "Copper fungicide (Bordeaux)", "Neem oil spray", "Baking soda solution"],
            "chemical": ["Chlorothalonil", "Mancozeb", "Azoxystrobin"],
            "prevention": ["Crop rotation (2-3 years)", "Drip irrigation", "Mulching", "Remove plant debris"]
        },
        "late_blight": {
            "desc": "Serious fungal disease (Phytophthora) - spreads rapidly in cool, wet weather",
            "organic": ["BURN infected plants", "Copper fungicide weekly", "Bacillus subtilis", "Remove volunteers"],
            "chemical": ["Metalaxyl (Ridomil)", "Chlorothalonil", "Mancozeb + Metalaxyl"],
            "prevention": ["Disease-free seeds", "No overhead watering", "Good air circulation", "Destroy infected material"]
        },
        "healthy": {
            "desc": "No disease detected - plant appears healthy!",
            "organic": ["Continue regular care", "Monitor weekly", "Organic compost"],
            "chemical": ["No treatment needed"],
            "prevention": ["Good air circulation", "Water at base", "Regular inspection", "Clean garden"]
        }
    }
    for key in treatments:
        if key.replace("_", "") in d.replace("_", ""):
            return treatments[key], "🟢 HEALTHY" if "healthy" in d else ("🔴 CRITICAL" if severity == "Severe" else "🟠 WARNING" if severity == "Moderate" else "🟡 CAUTION")
    return {"desc": "Disease detected", "organic": ["Remove affected leaves", "Copper fungicide"], "chemical": ["Consult expert"], "prevention": ["Crop rotation"]}, "ℹ️ MONITOR"

def main():
    st.title("🌿 Crop Disease Detection")
    st.markdown("### Tomato & Potato | Healthy, Early Blight, Late Blight")
    try:
        model, scaler, class_names = load_model()
    except Exception as e:
        st.error(f"Error: {e}")
        return
    uploaded = st.file_uploader("📤 Upload leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded", use_container_width=True)
        if st.button("🔬 Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                features = extract_all_features(img_array)
                pred_idx = model.predict(scaler.transform(features.reshape(1, -1)))[0]
                pred = class_names[pred_idx]
                score, sev_class, sev_color, diseased_mask = calculate_severity_score(img_array)
                orig, _, _, _ = preprocess_image(img_array)
                st.markdown("---")
                st.subheader("📊 Results")
                c1, c2 = st.columns([2, 1])
                with c1:
                    if "healthy" in pred.lower():
                        st.success(f"### ✅ {format_class_name(pred)}")
                        st.balloons()
                    else:
                        st.error(f"### ⚠️ {format_class_name(pred)}")
                with c2:
                    color_map = {"green": st.success, "yellow": st.warning, "orange": st.warning, "red": st.error}
                    color_map.get(sev_color, st.info)(f"### {sev_class}")
                    st.metric("Severity", f"{score:.1f}%")
                st.markdown("---")
                st.subheader("📈 Visualization")
                v1, v2 = st.columns(2)
                with v1:
                    st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Processed", use_container_width=True)
                with v2:
                    if diseased_mask is not None and np.sum(diseased_mask) > 0:
                        overlay = orig.copy()
                        overlay[diseased_mask > 0] = [0, 0, 255]
                        st.image(cv2.cvtColor(cv2.addWeighted(orig, 0.6, overlay, 0.4, 0), cv2.COLOR_BGR2RGB), caption="Affected (Red)", use_container_width=True)
                    else:
                        st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="No disease visible", use_container_width=True)
                st.markdown("---")
                st.subheader("💊 Treatment")
                treat, urgency = get_treatment(pred, sev_class)
                st.info(f"**{urgency}** - {treat[chr(100)+chr(101)+chr(115)+chr(99)]}")
                t1, t2, t3 = st.columns(3)
                with t1:
                    st.markdown("#### 🌱 Organic")
                    for i in treat["organic"]:
                        st.markdown(f"• {i}")
                with t2:
                    st.markdown("#### 🧪 Chemical")
                    for i in treat["chemical"]:
                        st.markdown(f"• {i}")
                with t3:
                    st.markdown("#### 🛡️ Prevention")
                    for i in treat["prevention"]:
                        st.markdown(f"• {i}")
    with st.sidebar:
        st.markdown("## 🌿 About")
        st.markdown("AI disease detection for Tomato & Potato")
        st.markdown("---")
        st.markdown("### 🎯 Classes")
        st.markdown("**Tomato:** Healthy, Early Blight, Late Blight")
        st.markdown("**Potato:** Healthy, Early Blight, Late Blight")
        st.markdown("---")
        st.markdown("### 👨‍💻 Developers")
        st.markdown("**Muhammad Haris** (413826)")
        st.markdown("**Muhammad Arham Siddiqui** (428887)")
        st.markdown("---")
        st.markdown("**CS-471** Machine Learning")
        st.markdown("**Class:** BEE-14 B, NUST")

if __name__ == "__main__":
    main()
