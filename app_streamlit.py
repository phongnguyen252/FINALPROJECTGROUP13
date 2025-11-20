import os
import tempfile
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from detect_tray import perspective_tray, crop_cell
from cnn_classification import CNNFoodClassifier
from infer_bill import BillGenerator

# Cấu hình kích thước hiển thị
ORIGINAL_MAX_WIDTH = 500   # tối đa width cho ảnh gốc khi hiển thị
FIXED_MAX_WIDTH = 500      # tối đa width cho ảnh sau khi chỉnh phối cảnh

# Hàm hỗ trợ resize ảnh để hiển thị (giữ tỉ lệ)
def pil_resize_for_display(pil_img: Image.Image, max_width: int) -> Image.Image:
    """Thay đổi kích thước ảnh PIL để width <= max_width, giữ tỉ lệ."""
    w, h = pil_img.size
    if w <= max_width:
        return pil_img
    ratio = max_width / float(w)
    new_size = (int(w * ratio), int(h * ratio))
    return pil_img.resize(new_size, Image.LANCZOS)

def cv2_to_pil(cv2_bgr_img: np.ndarray) -> Image.Image:
    """Chuyển ảnh OpenCV (BGR) sang PIL (RGB)."""
    rgb = cv2.cvtColor(cv2_bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# Giao diện
st.set_page_config(
    page_title="UEH Smart Canteen",
    page_icon="🍱",
    layout="wide",
)

# Header & Logo
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown("<h1 style='color:#005BAC;'>🍱 UEH Smart Canteen</h1>", unsafe_allow_html=True)
    st.write("Phân loại món ăn & tạo hóa đơn tự động bằng AI")
with col2:
    if os.path.exists("logo_color.jpg"):
        # Fix: Adjust the image width with an integer value, e.g., width=300
        st.image("logo_color.jpg", width=300)

st.markdown("---")

# Tải mô hình
@st.cache_resource
def load_models():
    classifier = CNNFoodClassifier()
    bill_gen = BillGenerator()
    return classifier, bill_gen

classifier, bill_gen = load_models()

# Chọn nguồn ảnh: upload hoặc webcam
st.subheader("Chọn nguồn ảnh")
mode = st.radio("Nguồn ảnh", ("Tải ảnh lên", "Webcam"))

img_path = None

if mode == "Tải ảnh lên":
    st.info("Tải ảnh khay cơm lên với định dạng .jpg/.jpeg/.png, đảm bảo ảnh chụp rõ nét, đủ sáng và ít nhất 3 góc khay nằm trong khung.")
    uploaded_file = st.file_uploader("📁 Tải ảnh khay cơm:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

else:  # Webcam
    st.info("Sử dụng webcam: Chụp ảnh khay cơm rõ nét, đủ sáng và ít nhất 3 góc khay nằm trong khung.")
    # Giới hạn kích thước khung webcam
    st.markdown(
        """
        <style>
        [data-testid="stCameraInput"] video {
            width: 900px !important;    /* Giảm chiều rộng video */
            height: auto !important;    /* Giữ tỉ lệ */
        }
        [data-testid="stCameraInput"] canvas {
            width: 350px !important;    /* Khung chụp ảnh */
            height: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    cam_file = st.camera_input("📷 Chụp ảnh khay bằng webcam")
    if cam_file:
        # cam_file giống file-like; lưu tạm để dùng chung pipeline
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(cam_file.getvalue())
            img_path = tmp.name

# Nếu có ảnh thì xử lý pipeline như trước
if img_path:
    # Hiển thị ảnh gốc (đã resize)
    try:
        pil_orig = Image.open(img_path).convert("RGB")
        pil_small = pil_resize_for_display(pil_orig, ORIGINAL_MAX_WIDTH)
        st.image(pil_small, caption="Ảnh khay gốc", use_container_width=False)
    except Exception:
        st.image(img_path, caption="Ảnh khay gốc", use_container_width=True)

    # Bước 1: Phát hiện & chỉnh phối cảnh
    st.subheader("1️⃣ Nhận diện khay cơm")
    fixed_img = perspective_tray(img_path)
    if fixed_img is None:
        st.error("❌ Không thể nhận diện được khay. Vui lòng thử lại ảnh khác.")
        st.stop()

    # Chuyển OpenCV -> PIL và resize trước khi hiển thị
    pil_fixed = cv2_to_pil(fixed_img)
    pil_fixed_small = pil_resize_for_display(pil_fixed, FIXED_MAX_WIDTH)
    st.image(pil_fixed_small, caption="Khay sau khi chỉnh phối cảnh", use_container_width=False)

    # Bước 2: Cắt 5 ô
    st.subheader("2️⃣ Cắt các ô thức ăn")
    crops = crop_cell(fixed_img)
    if not crops:
        st.error("❌ Không thể cắt được 5 ô. Hãy chụp lại ảnh rõ khay hơn.")
        st.stop()

    cols = st.columns(5)
    for i, (name, crop) in enumerate(crops.items()):
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cols[i % 5].image(rgb_crop, caption=f"Ô {i+1}: {name}", use_container_width=True)

    # Bước 3: Phân loại từng ô
    st.subheader("3️⃣ Kết quả phân loại món ăn bằng mô hình CNN")
    results = []
    result_cols = st.columns(5)

    for i, (name, crop) in enumerate(crops.items()):
        tmp_crop = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmp_crop.name, crop)
        pred = classifier.predict_image(tmp_crop.name)
        results.append(pred)

        with result_cols[i % 5]:
            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"{pred['predicted_class']}", use_container_width=True)
            st.metric("Độ tin cậy", f"{pred['confidence']*100:.1f}%")

    # Bước 4: Tạo hóa đơn PDF
    st.subheader("4️⃣ Tạo hóa đơn thanh toán")

    if st.button("🧾 Tạo & tải hóa đơn PDF"):
        with st.spinner("Đang tạo hóa đơn..."):
            bill, pdf_path = bill_gen.generate_bill_from_predictions(results)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.success("✅ Hóa đơn đã được tạo thành công!")
            st.download_button(
                label="📥 Tải hóa đơn PDF",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )

            st.info("Hóa đơn cũng được lưu trong thư mục `bills/` của dự án.")

st.markdown("---")
st.caption("© 2025 UEH Smart Canteen | Đồ án môn Trí tuệ nhân tạo của nhóm sinh viên 3I")
