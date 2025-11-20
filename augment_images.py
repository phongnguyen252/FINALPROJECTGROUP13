import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Cấu hình các tham số augment
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Đường dẫn dữ liệu gốc và nơi lưu ảnh mới
base_dir = './data'
output_dir = './data_augmented'
os.makedirs(output_dir, exist_ok=True)

# Lặp qua từng lớp (folder con)
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue  # bỏ qua file lạ nếu có
    
    # Tạo folder đích cho lớp đó
    save_path = os.path.join(output_dir, class_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Duyệt qua từng ảnh trong lớp
    for img_name in os.listdir(class_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_path, img_name)
        img = load_img(img_path, target_size=(128,128))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Sinh ra 5 ảnh mới và lưu
        i = 0
        for batch in datagen.flow(
            x, batch_size=1, 
            save_to_dir=save_path,
            save_prefix='aug', 
            save_format='jpg'
        ):
            i += 1
            if i >= 5:  # tạo 5 ảnh mới cho mỗi ảnh gốc
                break

print("Hoàn thành! Các ảnh mới tạo được lưu tại ./data_augmented")
