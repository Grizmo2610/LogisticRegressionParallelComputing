#include <cmath>
#include <vector>
#include <iostream>
#include <omp.h>
#include <random>
#include <algorithm>
#include <stdexcept>

using namespace std;

//-----------------------------------
// Matrix multiplication
//-----------------------------------
vector<vector<double>> dot(const vector<vector<double>>& A,
                           const vector<vector<double>>& B,
                           int n_threads = omp_get_max_threads()) {
    const int n = static_cast<int>(A.size());
    const int m = static_cast<int>(A[0].size());
    const int p = static_cast<int>(B[0].size());
    vector C(n, vector<double>(p, 0.0));

    #pragma omp parallel for collapse(2) num_threads(n_threads)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
    return C;
}

//-----------------------------------
// Transpose
//-----------------------------------
vector<vector<double>> transpose(const vector<vector<double>>& matrix,
                                 int n_threads = omp_get_max_threads()) {
    const int m = static_cast<int>(matrix.size());
    const int n = static_cast<int>(matrix[0].size());
    vector result(n, vector<double>(m, 0.0));

    #pragma omp parallel for collapse(2) num_threads(n_threads)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            result[j][i] = matrix[i][j];
    return result;
}

//-----------------------------------
// Norm
//-----------------------------------
double norm(const vector<double>& a, const vector<double>& b,
            int n_threads = omp_get_max_threads()) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(n_threads)
    for (int i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

//-----------------------------------
// Metrics
//-----------------------------------
double accuracy_score(const vector<int>& y_true, const vector<int>& y_pred,
                      int n_threads = omp_get_max_threads()) {
    double count = 0.0;
    #pragma omp parallel for reduction(+:count) num_threads(n_threads)
    for (int i = 0; i < y_true.size(); i++)
        if (y_true[i] == y_pred[i]) count++;
    return count / static_cast<double>(y_true.size());
}

double recall_score(const vector<int>& y_true, const vector<int>& y_pred,
                    int n_threads = omp_get_max_threads()) {
    int tp = 0, fn = 0;
    #pragma omp parallel for reduction(+:tp,fn) num_threads(n_threads)
    for (int i = 0; i < y_true.size(); i++) {
        if (y_true[i] == 1 && y_pred[i] == 1) tp++;
        if (y_true[i] == 1 && y_pred[i] == 0) fn++;
    }
    return (tp + fn == 0) ? 0.0 : static_cast<double>(tp) / (tp + fn);
}

double precision_score(const vector<int>& y_true, const vector<int>& y_pred,
                       int n_threads = omp_get_max_threads()) {
    int tp = 0, fp = 0;
    #pragma omp parallel for reduction(+:tp,fp) num_threads(n_threads)
    for (int i = 0; i < y_true.size(); i++) {
        if (y_true[i] == 1 && y_pred[i] == 1) tp++;
        if (y_true[i] == 0 && y_pred[i] == 1) fp++;
    }
    return (tp + fp == 0) ? 0.0 : static_cast<double>(tp) / (tp + fp);
}

double f1_score(const vector<int>& y_true, const vector<int>& y_pred,
                const int n_threads = omp_get_max_threads()) {
    const double p = precision_score(y_true, y_pred, n_threads);
    const double r = recall_score(y_true, y_pred, n_threads);
    return (p + r == 0.0) ? 0.0 : 2 * p * r / (p + r);
}

double sigmoid(const double z) {
    return 1.0 / (1.0 + exp(-z));
}

void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (const double val : row) cout << val << " ";
        cout << endl;
    }
}

void generate_random_data(const int n_samples,
                          const int n_features,
                          vector<vector<double>>& X,
                          vector<int>& y,
                          const double noise, const int random_state) {
    random_device rd;
    mt19937 gen(random_state);  // fixed seed
    normal_distribution<double> dist_x(0.0, 1.0);   // Input data
    uniform_real_distribution<double> dist_w(-1.0, 1.0); // Hidden weights
    normal_distribution<double> dist_n(0.0, noise); // Gaussian noise
    uniform_real_distribution<double> dist_b(-0.5, 0.5); // bias

    // Generate random weight
    vector<double> w(n_features);
    for (double& wi : w) wi = dist_w(gen);
    const double bias = dist_b(gen);

    // Generate data
    X.resize(n_samples, vector<double>(n_features));
    y.resize(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        double z = bias;
        for (int j = 0; j < n_features; ++j) {
            X[i][j] = dist_x(gen);
            z += X[i][j] * w[j];
        }
        z += dist_n(gen);
        y[i] = (z > 0.0) ? 1 : 0;
    }
}

/*
X, y: input data
X_train, X_test, y_train, y_test: outputs
test_size: proportion of test data (0.0 - 1.0)
shuffle: true to shuffle before splitting
random_state: fixed seed for reproducibility
*/
void train_test_split(const vector<vector<double>>& X,
                      const vector<int>& y,
                      vector<vector<double>>& X_train,
                      vector<vector<double>>& X_test,
                      vector<int>& y_train,
                      vector<int>& y_test,
                      double test_size,
                      bool shuffle,
                      unsigned int random_state) {
    if (X.size() != y.size()) {
        throw invalid_argument("X and y must have the same size");
    }
    const int N = static_cast<int>(X.size());

    vector<int> indices(N);
    for (int i = 0; i < N; ++i) indices[i] = i;

    if (shuffle) {
        mt19937 gen(random_state);  // fixed seed
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    const int n_test = static_cast<int>(N * test_size);
    const int n_train = N - n_test;

    X_train.resize(n_train);
    y_train.resize(n_train);
    X_test.resize(n_test);
    y_test.resize(n_test);

    for (int i = 0; i < n_train; ++i) {
        X_train[i] = X[indices[i]];
        y_train[i] = y[indices[i]];
    }

    for (int i = 0; i < n_test; ++i) {
        X_test[i] = X[indices[n_train + i]];
        y_test[i] = y[indices[n_train + i]];
    }
}