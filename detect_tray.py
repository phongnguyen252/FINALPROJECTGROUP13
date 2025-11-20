import cv2
import numpy as np
from utils import line_angle, intersection

def perspective_tray(img_path):
    # Đọc và xử lý ảnh cơ bản
    raw = cv2.imread(img_path)
    if raw is None:
        print("❌ Không thể đọc ảnh đầu vào!")
        return None

    if raw.shape[0] > raw.shape[1]:
        raw = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(raw, (800, 600))

    # Xác định hướng khay (3 ô trên – 2 ô dưới)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h = thresh.shape[0]
    upper = thresh[:h // 2, :]
    lower = thresh[h // 2:, :]

    bright_upper = np.sum(upper == 255)
    bright_lower = np.sum(lower == 255)
    mean_upper = np.mean(blur[:h // 2, :])
    mean_lower = np.mean(blur[h // 2:, :])

    if bright_upper > bright_lower * 1.1 or mean_upper > mean_lower * 1.05:
        img = cv2.rotate(img, cv2.ROTATE_180)

    # Phát hiện biên và các đường thẳng biên
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, 
                            minLineLength=200, maxLineGap=20)
    if lines is None:
        print("❌ Không thể xác định khay! Đảm bảo ít nhất 3 góc khay nằm trong khung hình!\n")
        return None

    horizontals, verticals = [], []
    for l in lines:
        ang = line_angle(l)
        if abs(ang) < 25 or abs(ang) > 155:
            horizontals.append(l)
        elif 65 < abs(ang) < 115:
            verticals.append(l)

    if not horizontals or not verticals:
        print("❌ Không thể xác định khay! Đảm bảo ít nhất 3 góc khay nằm trong khung hình!\n")
        return None

    # Tìm 4 cạnh và 4 đỉnh của tứ giác biên
    top = min(horizontals, key=lambda l: min(l[0][1], l[0][3]))
    bottom = max(horizontals, key=lambda l: max(l[0][1], l[0][3]))
    left = min(verticals, key=lambda l: min(l[0][0], l[0][2]))
    right = max(verticals, key=lambda l: max(l[0][0], l[0][2]))

    corners = [
        intersection(top, left),
        intersection(top, right),
        intersection(bottom, right),
        intersection(bottom, left)]

    # Kiểm tra tính hợp lệ
    valid = [c for c in corners if c is not None]
    if len(valid) != 4 or len(np.unique(valid, axis=0)) < 4:
        print("❌ Không thể xác định khay! Đảm bảo ít nhất 3 góc khay nằm trong khung hình!\n")
        return None

    distance = np.linalg.norm(np.array(corners[1]) - np.array(corners[0]))
    if distance < 400:
        print("❌ Khoảng cách đỉnh quá nhỏ, khay có thể bị che hoặc lệch góc!\n")
        return None

    # Chỉnh phối cảnh khay
    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [800, 0], [800, 600], [0, 600]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (800, 600))

    #cv2.imshow("Output Image", img_output)
    return img_output

def crop_cell(img):
    '''Cắt khay sau điều chỉnh phối cảnh thành 5 ô chứa thức ăn'''
    # Tọa độ cắt (x, y, w, h)
    regions = {
        "top_left": (0, 0, 266, 250),
        "top_mid": (266, 0, 266, 250),
        "top_right": (532, 0, 268, 250),
        "bottom_left": (0, 250, 400, 350),
        "bottom_right": (400, 250, 400, 350)
    }

    # Cắt và lưu trong bộ nhớ (chưa ghi tệp)
    crops = {name: img[y:y+h, x:x+w] for name, (x, y, w, h) in regions.items()}
    return crops