import time

import cv2
import torch
import fastcv
import numpy as np


def benchmark_blur(sizes=[1024, 2048, 4096], runs=100):
    results = []

    for size in sizes:
        print(f"\n=== Benchmarking {size}x{size} image ===")

        img_np = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        """
        start = time.perf_counter()
        for _ in range(runs):
            _ = cv2.medianBlur(img_np, 5)

        end = time.perf_counter()
        cv_time = (end - start) / runs * 1000  # ms per run
        
        results.append((size, cv_time))
        print(f"openCV (CPU): {cv_time:.4f} ms")
        
        """
        #for simple
        #img_torch = torch.from_numpy(img_np).cuda()
        #_ = fastcv.median_blur_simple(img_torch, 5)
        
        #for normal
        img_torch = torch.from_numpy(img_np).pin_memory()
        img_torch = torch.from_numpy(img_np).cuda()
        _ = fastcv.median_blur_simple_split(img_torch, 5)

        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.median_blur_simple_split(img_torch, 5)

        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_time = (end - start) / runs * 1000  # ms per run

        results.append((size, fc_time))
        print(f"fastcv (CUDA): {fc_time:.4f} ms")


    return results


if __name__ == "__main__":
    results = benchmark_blur()

    print("\n=== Final Results ===")
    print("Size\t\t\tfastcv (CUDA)")
    for size, fc_time in results:
        print(f"{size}x{size}\tms\t{fc_time:.4f} ms")