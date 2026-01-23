import time

import cv2
import torch
import fastcv


img = cv2.imread("../artifacts/test.jpg")
img1 = cv2.imread("../artifacts/binary.jpg")
img2 = cv2.imread("../artifacts/grayscale.jpg")

img_tensor1 = torch.from_numpy(img).cpu()
t1 = time.time()
median_tensor1 = fastcv.median_blur(img_tensor1, 5)
t2 = time.time()
median_np1 = median_tensor1.squeeze(-1).numpy()
cv2.imwrite("output_median.jpg", median_np1)
print(f"median blur time CUDA: {t2 - t1} seconds")

t1 = time.time()
median_tensor1 = fastcv.median_blur(img_tensor1, 5)
t2 = time.time()
median_np1 = median_tensor1.squeeze(-1).numpy()
cv2.imwrite("output_median.jpg", median_np1)
print(f"median blur time CUDA: {t2 - t1} seconds")

t1 = time.time()
median_tensor1 = fastcv.median_blur(img_tensor1, 5)
t2 = time.time()
median_np1 = median_tensor1.squeeze(-1).numpy()
cv2.imwrite("output_median.jpg", median_np1)
print(f"median blur time CUDA: {t2 - t1} seconds")


img_tensor2 = torch.from_numpy(img1).cpu()
t3 = time.time()
median_tensor2 = fastcv.median_blur(img_tensor2, 5)
t4 = time.time()
median_np2 = median_tensor2.squeeze(-1).numpy()
cv2.imwrite("output_median2.jpg", median_np2)
print(f"median blur time CUDA: {t4 - t3} seconds")


img_tensor3 = torch.from_numpy(img2).cpu()
t5 = time.time()
median_tensor3 = fastcv.median_blur(img_tensor3, 5)
t6 = time.time()
median_np3 = median_tensor3.squeeze(-1).numpy()
cv2.imwrite("output_median3.jpg", median_np3)
print(f"median blur time CUDA: {t6 - t5} seconds")


t7 = time.time()
median_cpu = cv2.medianBlur(img, 5)
t8 = time.time()
cv2.imwrite("output_median_cpu.jpg", median_cpu)
print(f"median blur time OpenCV CPU: {t8 - t7} seconds")
