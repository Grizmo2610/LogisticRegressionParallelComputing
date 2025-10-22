#include <iostream>
#include <vector>
#include <chrono>

#include "utils.h"
#include "model.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    const int core = atoi(argv[1]);
    const int n_samples = atoi(argv[2]);
    const int n_features = atoi(argv[3]);
    const double noise = atof(argv[4]);
    const double test_ratio = atof(argv[5]);  // assuming test_size as fraction

    vector<vector<double>> X;
    vector<int> y;
    const auto start = high_resolution_clock::now();
    generate_random_data(n_samples, n_features, X, y, noise);
    const auto stop = high_resolution_clock::now();
    const auto duration_gen = duration_cast<milliseconds>(stop - start).count();
    cout << "Generating time: " << duration_gen << " ms" << endl;

    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;

    train_test_split(X, y, X_train, X_test, y_train, y_test, test_ratio, true, 42);

    LogisticRegression model(core);

    // measure training time
    auto t1 = high_resolution_clock::now();
    const vector<double> weights = model.fit(X_train, y_train, 1e-3, 1e-6, 30);
    auto t2 = high_resolution_clock::now();
    const auto duration_train = duration_cast<milliseconds>(t2 - t1).count();
    cout << "Training time: " << duration_train << " ms" << endl;

    cout << "=====METRICS=====" << endl;
    // measure prediction time on train set
    t1 = high_resolution_clock::now();
    const vector<int> y_train_predict = model.predict(X_train);
    t2 = high_resolution_clock::now();
    const auto duration_predict_train = duration_cast<milliseconds>(t2 - t1).count();

    // measure prediction time on test set
    t1 = high_resolution_clock::now();
    const vector<int> y_test_predict = model.predict(X_test);
    t2 = high_resolution_clock::now();
    const auto duration_predict_test = duration_cast<milliseconds>(t2 - t1).count();

    cout << "Prediction time (train set): " << duration_predict_train << " ms" << endl;
    cout << "Prediction time (test set): " << duration_predict_test << " ms" << endl;

    cout << "Train set Accuracy: " << accuracy_score(y_train, y_train_predict) << endl;
    cout << "Train set F1 score: " << f1_score(y_train, y_train_predict) << endl;

    cout << "Test set Accuracy: " << accuracy_score(y_test, y_test_predict) << endl;
    cout << "Test set F1 score: " << f1_score(y_test, y_test_predict) << endl;
    return 0;
}
