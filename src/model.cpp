#include "model.h"
#include "utils.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <algorithm>


using namespace std;

LogisticRegression::LogisticRegression() = default;

LogisticRegression::LogisticRegression(const int core) {
    if (core < 1) {
        this->core = 1;
        this->parallel = false;
    } else {
        this->core = core;
        this->parallel = true;
    }
}

// =======================
// Fit logistic regression model
// Uses OpenMP for parallel gradient computation
// =======================
vector<double> LogisticRegression::fit(
    const vector<vector<double>>& X,
    const vector<int>& y,
    const double lr,
    const double epsilon,
    const int max_iter,
    const double lambda_l2
) {
    const int N = static_cast<int>(X.size());
    const int d = static_cast<int>(X[0].size());

    // -------------------------------
    // 1) Standardize input features
    // -------------------------------
    vector<double> mean(d, 0.0);
    vector<double> var(d, 0.0);

    for (int j = 0; j < d; ++j) {
        for (int i = 0; i < N; ++i) {
            mean[j] += X[i][j];
        }
        mean[j] /= N;

        for (int i = 0; i < N; ++i) {
            var[j] += (X[i][j] - mean[j]) * (X[i][j] - mean[j]);
        }
        var[j] = sqrt(var[j] / N + 1e-12);
    }

    vector<vector<double>> X_scaled(N, vector<double>(d));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            X_scaled[i][j] = (X[i][j] - mean[j]) / var[j];
        }
    }

    vector<double> X_flat = flatten(X_scaled);

    // -------------------------------
    // 2) Initialize weights and bias
    // -------------------------------
    vector<double> w(d);
    for (auto& wi : w) {
        wi = ((double)rand() / RAND_MAX - 0.5) * 0.01;
    }
    this->bias = 0.0;

    // Thread-local gradient buffers
    vector<vector<double>> local_grad_w_threads(core, vector<double>(d, 0.0));
    vector<double> local_grad_b_threads(core, 0.0);

    vector<double> grad_w(d, 0.0);
    double grad_b = 0.0;

    double prev_loss = 1e18;

    // -------------------------------
    // 3) Training loop
    // -------------------------------
    for (int epoch = 0; epoch < max_iter; ++epoch) {

        fill(grad_w.begin(), grad_w.end(), 0.0);
        grad_b = 0.0;

        for (int t = 0; t < core; ++t) {
            fill(local_grad_w_threads[t].begin(), local_grad_w_threads[t].end(), 0.0);
            local_grad_b_threads[t] = 0.0;
        }

        #pragma omp parallel num_threads(core)
        {
            const int tid = omp_get_thread_num();

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < N; ++i) {
                double z = this->bias;

                #pragma omp simd reduction(+:z)
                for (int j = 0; j < d; ++j) {
                    z += X_flat[i * d + j] * w[j];
                }

                double p = 1.0 / (1.0 + exp(-z));
                double error = p - y[i];

                #pragma omp simd
                for (int j = 0; j < d; ++j) {
                    local_grad_w_threads[tid][j] += error * X_flat[i * d + j];
                }

                local_grad_b_threads[tid] += error;
            }
        }

        // Combine gradients from threads
        for (int t = 0; t < core; ++t) {
            #pragma omp simd
            for (int j = 0; j < d; ++j) {
                grad_w[j] += local_grad_w_threads[t][j];
            }
            grad_b += local_grad_b_threads[t];
        }

        // Add L2 regularization to gradient
        #pragma omp simd
        for (int j = 0; j < d; ++j) {
            grad_w[j] = grad_w[j] / N + lambda_l2 * 2.0 * w[j];
        }

        grad_b /= N;

        // Update weights and bias
        #pragma omp simd
        for (int j = 0; j < d; ++j) {
            w[j] -= lr * grad_w[j];
        }

        this->bias -= lr * grad_b;

        // Check for convergence using loss
        double loss = logistic_loss(w, this->bias, X_flat, y, N, d, lambda_l2);
        if (fabs(prev_loss - loss) < epsilon) {
            break;
        }

        prev_loss = loss;
        this->weights = w;
    }

    return this->weights;
}

// =======================
// Predict method
// Uses flattened standardized input
// =======================
vector<int> LogisticRegression::predict(
    const vector<vector<double>>& X,
    const double thresh
) const {
    const int N = static_cast<int>(X.size());
    vector<int> y(N, 0);
    const int d = static_cast<int>(this->weights.size());

    vector<double> X_flat = flatten(X);

    #pragma omp parallel for num_threads(core)
    for (int i = 0; i < N; ++i) {
        double z = this->bias;

        #pragma omp simd
        for (int j = 0; j < d; ++j) {
            z += X_flat[i * d + j] * this->weights[j];
        }

        double p = 1.0 / (1.0 + exp(-z));
        y[i] = (p > thresh) ? 1 : 0;
    }

    return y;
}
