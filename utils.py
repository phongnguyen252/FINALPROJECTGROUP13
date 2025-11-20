import cv2
import numpy as np
import json
import os

def line_angle(line):
    '''Tính độ nghiêng của đường thẳng'''
    x1, y1, x2, y2 = line[0]
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def intersection(line1, line2):
    '''Tìm giao điểm của hai đường thẳng'''
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return int(px), int(py)

def print_bill(bill):
    """Nicely print bill to console."""
    print("\n----- HÓA ĐƠN -----")
    for it in bill.get("items", []):
        print(f'{it["file"]:<12} {it["name"]:<25} {it["price"]:>8}  (p={it["prob"]})')
    print("--------------------")
    print(f'TỔNG: {bill.get("total", 0)} VND')
    print("--------------------")

def save_bill_json(folder, predictions, bill, filename="predictions.json"):
    """Save predictions+bill into json file under folder."""
    os.makedirs(folder, exist_ok=True)
    out = {"predictions": predictions, "bill": bill}
    with open(os.path.join(folder, filename), "w", encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return os.path.join(folder, filename)