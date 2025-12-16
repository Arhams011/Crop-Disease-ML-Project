
import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

# Page config
st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# Load model artifacts
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, scaler, class_names

# Feature extraction functions
def preprocess_image(image, target_size=(256, 256)):
    img_resized = cv2.resize(image, target_size)
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20, 20, 20])
    upper_green = np.array([100, 255, 255])
    lower_brown = np.array([5, 20, 20])
    upper_brown = np.array([30, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask = cv2.bitwise_or(mask_green, mask_brown)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    gray_masked = img_as_ubyte(gray_masked / 255.0) if gray_masked.max() > 1 else img_as_ubyte(gray_masked)

    try:
        glcm = graycomatrix(gray_masked, distances=[1, 2, 3],
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256, symmetric=True, normed=True)

        contrast = graycoprops(glcm, "contrast").mean()
        correlation = graycoprops(glcm, "correlation").mean()
        energy = graycoprops(glcm, "energy").mean()
        homogeneity = graycoprops(glcm, "homogeneity").mean()
        dissimilarity = graycoprops(glcm, "dissimilarity").mean()

        return np.array([contrast, correlation, energy, homogeneity, dissimilarity])
    except:
        return np.array([0, 0, 0, 0, 0])

def extract_shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.array([0, 0, 0, 0, 0])

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    img_area = mask.shape[0] * mask.shape[1]
    area_normalized = area / img_area
    perimeter_normalized = perimeter / (2 * (mask.shape[0] + mask.shape[1]))

    return np.array([area_normalized, perimeter_normalized, aspect_ratio, circularity, solidity])

def extract_all_features(image):
    orig, hsv, mask, segmented = preprocess_image(image)
    color_features = extract_color_features(hsv, mask)
    texture_features = extract_glcm_features(segmented, mask)
    shape_features = extract_shape_features(mask)
    return np.concatenate([color_features, texture_features, shape_features])

def format_class_name(class_name):
    """Format class name for display."""
    parts = class_name.split("___")
    if len(parts) == 2:
        crop, condition = parts
        crop = crop.replace("_", " ").replace("(", "").replace(")", "")
        condition = condition.replace("_", " ")
        return f"{crop} - {condition}"
    return class_name.replace("_", " ")

# Main app
def main():
    st.title("🌿 Crop Leaf Disease Detection")
    st.markdown("Upload a leaf image to detect diseases in **Tomato**, **Potato**, or **Corn/Maize** plants.")

    # Load model
    try:
        model, scaler, class_names = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure best_model.pkl, scaler.pkl, and class_names.pkl are in the app directory.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Predict button
        if st.button("🔍 Analyze Leaf", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Extract features
                    features = extract_all_features(img_array)
                    features_scaled = scaler.transform(features.reshape(1, -1))

                    # Predict
                    prediction_idx = model.predict(features_scaled)[0]
                    prediction = class_names[prediction_idx]

                    # Get confidence if available
                    confidence = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(features_scaled)[0]
                        confidence = proba[prediction_idx]

                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Diagnosis Results")

                    formatted_prediction = format_class_name(prediction)

                    # Check if healthy or diseased
                    if "healthy" in prediction.lower():
                        st.success(f"✅ **{formatted_prediction}**")
                        st.balloons()
                    else:
                        st.warning(f"⚠️ **{formatted_prediction}**")

                    if confidence:
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.progress(confidence)

                    # Show top predictions if available
                    if hasattr(model, "predict_proba"):
                        st.markdown("---")
                        st.subheader("All Predictions")
                        proba = model.predict_proba(features_scaled)[0]
                        top_indices = np.argsort(proba)[::-1][:5]

                        for idx in top_indices:
                            class_name = format_class_name(class_names[idx])
                            prob = proba[idx]
                            st.text(f"{class_name}: {prob:.1%}")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses traditional machine learning
        (SVM/Random Forest/KNN) with handcrafted features:

        - **Color**: HSV histograms
        - **Texture**: GLCM features
        - **Shape**: Morphological features

        **Supported Crops:**
        - 🍅 Tomato
        - 🥔 Potato
        - 🌽 Corn/Maize
        """)

        st.markdown("---")
        st.markdown("""
        **CS-471 Machine Learning Project**
        Muhammad Haris & Muhammad Arham Siddiqui
        BEE-14 B, NUST
        """)

if __name__ == "__main__":
    main()

