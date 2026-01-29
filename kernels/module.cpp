#include <torch/extension.h>

// forward declarations
torch::Tensor rgb_to_gray(torch::Tensor img);
torch::Tensor box_blur(torch::Tensor img, int blurSize);
torch::Tensor sobel(torch::Tensor img);
torch::Tensor dilation(torch::Tensor img, int filterSize);
torch::Tensor erosion(torch::Tensor img, int filterSize);
torch::Tensor median_blur(torch::Tensor img, int blur_size);
torch::Tensor median_blur_split(torch::Tensor img, int blur_size);
torch::Tensor median_blur_simple(torch::Tensor img, int blur_size);
torch::Tensor median_blur_simple_split(torch::Tensor img, int blur_size);
torch::Tensor median_blur_simple_noshared(torch::Tensor img, int blur_size);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("rgb2gray", &rgb_to_gray, "rgb to grayscale kernel");
    m.def("blur", &box_blur, "box blur kernel");
    m.def("sobel", &sobel, "sobel filter kernel");
    m.def("dilate", &dilation, "dilation kernel");
    m.def("erode", &erosion, "erosion kernel");
    m.def("median_blur", &median_blur, "median blur kernel");
    m.def("median_blur_split", &median_blur_split, "median blur split kernel");
    m.def("median_blur_simple", &median_blur_simple, "median blur simple kernel");
    m.def("median_blur_simple_split", &median_blur_simple_split, "median blur simple split kernel");
    m.def("median_blur_simple_noshared", &median_blur_simple_noshared, "median blur simple no shared kernel");
}