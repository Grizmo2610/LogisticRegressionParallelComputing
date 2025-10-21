#include "model.h"
#include "utils.h"
#include <omp.h>

using namespace std;

LogisticRegression::LogisticRegression() = default;

LogisticRegression::LogisticRegression(const int core) : core(core) {
}

vector<double> LogisticRegression::fit(const vector<vector<double>>& X,
                                       const vector<int>& y,
                                       const double lr,
                                       const double epsilon,
                                       const int max_iter){
    const int N = static_cast<int>(X.size());
    const int d = static_cast<int>(X[0].size());

    vector<double> w(d, 1.0);
    this->bias = 0.0;

    for (int epoch = 0; epoch < max_iter; ++epoch){
        vector<double> grad_w(d, 0.0);
        double grad_b = 0.0;

        // Thread-local vectors for safe parallel reduction
#pragma omp parallel num_threads(core)
        {
            vector<double> local_grad_w(d, 0.0);
            double local_grad_b = 0.0;

#pragma omp for
            for (int i = 0; i < N; ++i)
            {
                double z = 0.0;
                for (int j = 0; j < d; ++j)
                    z += X[i][j] * w[j];
                z += this->bias;

                double pred = sigmoid(z);
                double error = pred - y[i];

                for (int j = 0; j < d; ++j) {
                    local_grad_w[j] += error * X[i][j];
                }
                local_grad_b += error;
            }

            // Combine thread-local results into global grad_w and grad_b
#pragma omp critical
            {
                for (int j = 0; j < d; ++j)
                    grad_w[j] += local_grad_w[j];
                grad_b += local_grad_b;
            }
        } // end parallel

        // Update weights
        for (int j = 0; j < d; ++j) {
            w[j] -= lr * grad_w[j] / N;
        }
        this->bias -= lr * grad_b / N;

        // Check convergence
        if (!this->weights.empty())
        {
            if (norm(this->weights, w, core) <= epsilon)
            {
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

    #pragma omp parallel for num_threads(core)
    for (int i = 0; i < N; ++i) {
        double z = 0.0;
        for (int j = 0; j < this->weights.size(); ++j) {
            z += X[i][j] * this->weights[j];
        }
        z += this->bias;

        const double pred = sigmoid(z);
        y[i] = (pred > thresh) ? 1 : 0;
    }
    return y;
}
