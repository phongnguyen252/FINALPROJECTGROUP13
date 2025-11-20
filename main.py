import os
import cv2
import datetime
from tkinter import Tk, filedialog, messagebox
from detect_tray import perspective_tray, crop_cell
from cnn_classification import CNNFoodClassifier
from infer_bill import BillGenerator

# Thư mục xuất
CROP_DIR = "./data_crop"
os.makedirs(CROP_DIR, exist_ok=True)

# Chương trình chính
if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    # Hướng dẫn người dùng
    user_choice = messagebox.askquestion(
        "Chương trình nhận diện và tính tiền phần ăn",
        "HƯỚNG DẪN CHỤP ẢNH KHAY CƠM:\n\n"
        "1. Chụp toàn bộ khay cơm theo góc 90° từ trên xuống.\n"
        "2. Đảm bảo ít nhất 3 góc khay nằm trong khung hình.\n"
        "3. Tránh nguồn sáng chói hoặc thiếu sáng.\n\n"
        "Bạn có muốn tiếp tục?",
        icon='info')

    if user_choice != 'yes':
        messagebox.showinfo("Thoát chương trình", "Cảm ơn bạn đã trải nghiệm!")
        exit()

    # Chọn ảnh và kiểm tra tính hợp lệ
    img_path = filedialog.askopenfilename(
        title="Chọn ảnh khay cơm",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])

    if not img_path or not os.path.exists(img_path):
        exit()

    try:
        # Bước 1: Phát hiện & chỉnh phối cảnh khay
        fixed = perspective_tray(img_path)
        if fixed is None:
            print("❌ Không thể xác định khay! Vui lòng chụp lại ảnh khay cơm!\n")
            exit()

        # Bước 2: Cắt khay ra thành 5 ô thức ăn
        cropped = crop_cell(fixed)
        if not cropped:
            print("❌ Không thể cắt ảnh khay! Vui lòng chụp lại ảnh khay cơm!\n")
            exit()

        # Bước 3: Tạo thư mục con và lưu ảnh từng ô
        subfolder = os.path.join(CROP_DIR, f"crop_{datetime.datetime.now().strftime('%H-%M-%S_%d-%m')}")
        os.makedirs(subfolder, exist_ok=True)

        for i, (_, crop_img) in enumerate(cropped.items(), start=1):
            save_path = os.path.join(subfolder, f"cell_{i}.jpg")
            success = cv2.imwrite(save_path, crop_img)

        # Bước 4: Dự đoán món ăn bằng mô hình CNN đã huấn luyện
        classifier = CNNFoodClassifier()
        results = []

        for filename in os.listdir(subfolder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(subfolder, filename)
                result = classifier.predict_image(img_path)
                results.append(result)
        
        print("\n✅ Kết quả phân loại 5 ô:")
        for r in results:
            print(f" + {os.path.basename(r['path'])}: {r['predicted_class']} ({r['confidence']:.1%})")

        # Bước 5: Xuất hóa đơn PDF
        bill_gen = BillGenerator()
        bill, pdf_path = bill_gen.generate_bill_from_predictions(results)
        print(f"\n📂 Hóa đơn đã được tạo thành công và lưu tại: {pdf_path}")

    except Exception as e:
        print(f"❌ Lỗi xử lý: {e}")
        exit()

    cv2.waitKey(0)
    cv2.destroyAllWindows()