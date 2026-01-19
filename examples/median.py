import cv2
import torch
import fastcv

img = cv2.imread("../artifacts/test.jpg")
img_tensor = torch.from_numpy(img).cuda()
median_tensor = fastcv.median_blur(img_tensor, 5)
median_np = median_tensor.squeeze(-1).cpu().numpy()
cv2.imwrite("output_median.jpg", median_np)

print("saved median blur image.")