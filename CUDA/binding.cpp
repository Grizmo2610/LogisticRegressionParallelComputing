#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C" void train(
    const float* X, const float* y,
    float lr, float epsilon, int max_iter,
    int N, int d, float* weight, float* bias
);

namespace py = pybind11;

py::tuple train_py(py::array_t<float> X_np, py::array_t<float> y_np,
                   float lr, float epsilon, int max_iter)
{
    auto bufX = X_np.request();
    auto bufY = y_np.request();
    int N = bufX.shape[0];
    int d = bufX.shape[1];

    py::array_t<float> weight_np(d);
    py::array_t<float> bias_np(1);

    auto w_ptr = static_cast<float*>(weight_np.request().ptr);
    auto b_ptr = static_cast<float*>(bias_np.request().ptr);

    train(static_cast<float*>(bufX.ptr), static_cast<float*>(bufY.ptr),
          lr, epsilon, max_iter, N, d, w_ptr, b_ptr);

    return py::make_tuple(weight_np, bias_np);
}

PYBIND11_MODULE(logreg, m) {
    m.def("train", &train_py, "Train logistic regression using CUDA");
}
