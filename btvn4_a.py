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
# 1. CONTRAST STRETCHING
# =========================
def contrast_stretching(img, r1=70, s1=0, r2=140, s2=255):
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = img[i, j]

            if r < r1:
                s = (s1 / r1) * r
            elif r1 <= r <= r2:
                s = ((s2 - s1)/(r2 - r1)) * (r - r1) + s1
            else:
                s = ((255 - s2)/(255 - r2)) * (r - r2) + s2

            result[i, j] = np.clip(s, 0, 255)

    return result


# =========================
# 2. THRESHOLDING
# =========================
def thresholding(img, T=127):
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > T:
                result[i, j] = 255
            else:
                result[i, j] = 0

    return result


# =========================
# 3. LOG TRANSFORM
# =========================
def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    result = c * np.log(1 + img)
    return np.array(result, dtype=np.uint8)


# =========================
# 4. GAMMA CORRECTION
# =========================
def gamma_transform(img, gamma=0.5):
    img_norm = img / 255.0
    result = np.power(img_norm, gamma)
    return np.uint8(result * 255)


# =========================
# THỰC THI
# =========================
contrast_img = contrast_stretching(img)
threshold_img = thresholding(img)
log_img = log_transform(img)
gamma_img = gamma_transform(img, gamma=0.5)

# =========================
# HIỂN THỊ
# =========================
cv2.imshow("Original", img)
cv2.imshow("Contrast Stretching", contrast_img)
cv2.imshow("Threshold", threshold_img)
cv2.imshow("Log Transform", log_img)
cv2.imshow("Gamma Correction", gamma_img)

cv2.waitKey(0)
cv2.destroyAllWindows()