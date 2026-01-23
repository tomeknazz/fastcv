import time

import cv2
import torch
import fastcv


img = cv2.imread("../artifacts/test.jpg")
img1 = cv2.imread("../artifacts/binary.jpg")
img2 = cv2.imread("../artifacts/grayscale.jpg")

img_tensor1 = torch.from_numpy(img).pin_memory()
t1 = time.time()
median_tensor1 = fastcv.median_blur(img_tensor1, 5)
t2 = time.time()
median_np1 = median_tensor1.squeeze(-1).cpu().numpy()
cv2.imwrite("output_median.jpg", median_np1)
print(f"median blur time CUDA: {t2 - t1} seconds")

t7 = time.time()
median_cpu = cv2.medianBlur(img, 5)
t8 = time.time()
cv2.imwrite("output_median_cpu.jpg", median_cpu)
print(f"median blur time OpenCV CPU: {t8 - t7} seconds")
