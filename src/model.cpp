#include "model.h"
#include "utils.h"
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

LogisticRegression::LogisticRegression() = default;

LogisticRegression::LogisticRegression(const int core) {
    if (core < 1) {
        this->core = 1;
        this->parallel = false;
    }else {
        this->core = core;
        this->parallel = true;
    }
}

vector<double> LogisticRegression::fit(const vector<vector<double>>& X,
                                       const vector<int>& y,
                                       const double lr,
                                       const double epsilon,
                                       const int max_iter){
    const int N = static_cast<int>(X.size());
    const int d = static_cast<int>(X[0].size());

    vector<double> w(d);
    for (auto& wi : w) {
        wi = ((double)rand() / RAND_MAX - 0.5) * 0.01;
    }
    this->bias = 0.0;

    const vector<double> X_flat = flatten(X);

    // Thread-local vectors for safe parallel reduction
    vector<vector<double>> local_grad_w_threads(core, vector<double>(d, 0.0));
    vector<double> local_grad_b_threads(core, 0.0);

    double grad_b = 0.0;
    vector<double> grad_w(d, 0.0);

    for (int epoch = 0; epoch < max_iter; epoch++){

        fill(grad_w.begin(), grad_w.end(), 0.0);
        grad_b = 0.0;
        for (int t = 0; t < core; t++) {
            fill(local_grad_w_threads[t].begin(), local_grad_w_threads[t].end(), 0.0);
            local_grad_b_threads[t] = 0.0;
        }

        #pragma omp parallel num_threads(core)
        {
            const int tid = omp_get_thread_num();

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < N; i++){
                double z = 0.0;

                // vectorization
                #pragma omp simd reduction(+:z) if(parallel)
                for (int j = 0; j < d; j++) {
                    z += X_flat[i * d + j] * w[j];
                }

                z += this->bias;

                const double pred = sigmoid(z);
                const double error = pred - y[i];

                // vectorization
                #pragma omp simd if(parallel)
                for (int j = 0; j < d; j++) {
                    local_grad_w_threads[tid][j] += error * X_flat[i * d + j];
                }
                local_grad_b_threads[tid] += error;
            }
        } // end parallel

        // Combine thread-local results into global grad_w and grad_b
        for (int t = 0; t < core; t++) {
            #pragma omp simd if(parallel)
            for (int j = 0; j < d; j++) {
                grad_w[j] += local_grad_w_threads[t][j];
            }
            grad_b += local_grad_b_threads[t];
        }

        // Update weights vectorization
        #pragma omp simd if(parallel)
        for (int j = 0; j < d; ++j) {
            w[j] -= lr * grad_w[j] / N;
        }

        this->bias -= lr * grad_b / N;

        // Check convergence
        if (!this->weights.empty()) {
            if (norm(this->weights, w, core) <= epsilon){
                break;
            }
        }

        this->weights = w;
    }

    return this->weights;
}

vector<int> LogisticRegression::predict(const vector<vector<double> > &X, const double thresh) const {
    const int N = static_cast<int>(X.size());
    vector y(N, 0);
    const vector<double> X_flat = flatten(X);

    #pragma omp parallel for num_threads(core)
    for (int i = 0; i < N; ++i) {
        double z = 0.0;

        #pragma omp simd if(parallel)
        for (int j = 0; j < this->weights.size(); ++j) {
            z += X_flat[i * X[0].size() + j]* this->weights[j];
        }
        z += this->bias;

        const double pred = sigmoid(z);
        y[i] = (pred > thresh) ? 1 : 0;
    }
    return y;
}

PYBIND11_MODULE(Logistic, m) {
    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<int>())
        .def("fit", &LogisticRegression::fit)
        .def("predict", &LogisticRegression::predict);
}