#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "model.h"

namespace py = pybind11;
PYBIND11_MODULE(Logistic, m) {
    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<int>())
        .def("fit", &LogisticRegression::fit)
        .def("predict", &LogisticRegression::predict);
}