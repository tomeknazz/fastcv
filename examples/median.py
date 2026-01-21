import cv2
import torch
import fastcv
import time

img = cv2.imread("../artifacts/test.jpg")
img_tensor = torch.from_numpy(img).cuda()
t1=time.time()
median_tensor = fastcv.median_blur(img_tensor, 5)
median_np = median_tensor.squeeze(-1).cpu().numpy()
cv2.imwrite("output_median.jpg", median_np)
t2=time.time()
print(f"median blur time CUDA: {t2-t1} seconds")

t3=time.time()
median_cpu=cv2.medianBlur(img,5)
cv2.imwrite("output_median_cpu.jpg", median_cpu)
t4=time.time()
print(f"median blur time OpenCV CPU: {t4-t3} seconds")
