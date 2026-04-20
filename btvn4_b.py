import cv2
import numpy as np

# =========================
# ĐỌC ẢNH (GRAYSCALE)
# =========================
img = cv2.imread('anh1.jpg', 0)

if img is None:
    print("Không đọc được ảnh!")
    exit()

# =========================
# 1. NEGATIVE (âm bản)
# =========================
negative = cv2.bitwise_not(img)

# =========================
# 2. THRESHOLD
# =========================
_, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# =========================
# 3. CONTRAST STRETCHING
# =========================
# dùng normalize để giãn dải cường độ
contrast = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# =========================
# 4. LOG TRANSFORM
# (OpenCV không có sẵn → dùng numpy)
# =========================
c = 255 / np.log(1 + np.max(img))
log_img = c * np.log(1 + img)
log_img = np.uint8(log_img)

# =========================
# 5. GAMMA CORRECTION
# =========================
gamma = 0.5

# tạo bảng tra (LUT)
table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")

gamma_img = cv2.LUT(img, table)

# =========================
# HIỂN THỊ
# =========================
cv2.imshow("Original", img)
cv2.imshow("Negative", negative)
cv2.imshow("Threshold", threshold)
cv2.imshow("Contrast Stretching", contrast)
cv2.imshow("Log Transform", log_img)
cv2.imshow("Gamma Correction", gamma_img)

cv2.waitKey(0)
cv2.destroyAllWindows()