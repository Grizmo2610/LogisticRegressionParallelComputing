// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <fstream>
// #include <cstdlib>
// #include <sstream>

// #include "utils.h"
// #include "model.h"

// using namespace std;
// using namespace std::chrono;

// struct Result {
//     size_t n_samples;
//     size_t n_features;
//     long training_time;
//     long predict_time;
//     int core;
//     double acc;
//     double f1;
// };

// string test_data(const string data_path, int core, double test_ratio = 0.2) {
//     ifstream file(data_path);
//     string line;

//     vector<vector<double>> X;
//     vector<int> y;

//     cout << "Reading data from " << data_path << endl;
//     auto t1 = high_resolution_clock::now();
//     if (!file.is_open()) {
//         cerr << "Can't open file\n";
//         return "";
//     }

//     bool header_skipped = false;
//     while (getline(file, line)) {
//         stringstream ss(line);
//         string value;
//         vector<double> row;

//         // Skip header
//         if (!header_skipped) {
//             header_skipped = true;
//             if (!isdigit(line[0])) {
//                 continue;
//             }
//         }

//         while (getline(ss, value, ',')) {
//             row.push_back(stod(value));
//         }

//         int label = static_cast<int>(row.back());
//         X.push_back(row);
//         y.push_back(label);
//     }

//     file.close();
//     auto t2 = high_resolution_clock::now();
//     const auto duration_reading_data = duration_cast<milliseconds>(t2 - t1).count() / 1000;
//     cout << "Reading data time: " << duration_reading_data << " s\n";

//     cout << "Loaded X: " << X.size() << " samples, " << X[0].size() << " features" << endl;

//     vector<vector<double>> X_train, X_test;
//     vector<int> y_train, y_test;

//     t1 = high_resolution_clock::now();
//     train_test_split(X, y, X_train, X_test, y_train, y_test, test_ratio, true, 42);
//     t2 = high_resolution_clock::now();
//     const auto split_time = duration_cast<milliseconds>(t2 - t1).count();
//     cout << "Split data time: "<< split_time << " ms" << endl;

//     LogisticRegression model(core);

//     t1 = high_resolution_clock::now();
//     const vector<double> weights = model.fit(X_train, y_train, 1e-3, 1e-6, 30);
//     t2 = high_resolution_clock::now();
//     const auto duration_train = duration_cast<milliseconds>(t2 - t1).count();

//     t1 = high_resolution_clock::now();
//     const vector<int> y_test_predict = model.predict(X_test);
//     t2 = high_resolution_clock::now();
//     const auto duration_predict = duration_cast<milliseconds>(t2 - t1).count();

//     double acc = accuracy_score(y_test, y_test_predict);
//     double f1 = f1_score(y_test, y_test_predict);

//     cout << "=====METRICS=====" << endl;
//     cout << "Accuracy: " << acc << endl;
//     cout << "F1 score: " << f1 << endl;
//     cout << "Train time: " << duration_train << " ms\n";
//     cout << "Predict time: " << duration_predict << " ms\n";

//     // --- Lưu kết quả ---
//     Result result = {
//         X.size(),
//         X[0].size(),
//         duration_train,
//         duration_predict,
//         core,
//         acc,
//         f1
//     };

//     bool new_file = !ifstream("results.csv").good();
//     ofstream out("results.csv", ios::app);
//     if (new_file) {
//         out << "n_samples,n_features,training_time,predict_time,core,accuracy,f1\n";
//     }

//     out << result.n_samples << ","
//         << result.n_features << ","
//         << result.training_time << ","
//         << result.predict_time << ","
//         << result.core << ","
//         << result.acc << ","
//         << result.f1 << "\n";
//     out.close();

//     return "Done";
// }

// void random_test(int argc, char* argv[]) {
//     const int core = atoi(argv[1]);
//     const int n_samples = atoi(argv[2]);
//     const int n_features = atoi(argv[3]);
//     const double noise = atof(argv[4]);
//     const double test_ratio = atof(argv[5]);  // assuming test_size as fraction

//     vector<vector<double>> X;
//     vector<int> y;
//     const auto start = high_resolution_clock::now();
//     generate_random_data(n_samples, n_features, X, y, noise);
//     const auto stop = high_resolution_clock::now();
//     const auto duration_gen = duration_cast<milliseconds>(stop - start).count();
//     cout << "Generating time: " << duration_gen << " ms" << endl;

//     vector<vector<double>> X_train, X_test;
//     vector<int> y_train, y_test;

//     auto t1 = high_resolution_clock::now();
//     train_test_split(X, y, X_train, X_test, y_train, y_test, test_ratio, true, 42);
//     auto t2 = high_resolution_clock::now();
//     const auto split_time = duration_cast<milliseconds>(t2 - t1).count();
//     cout << "Split data time: "<< split_time << endl;
//     LogisticRegression model(core);

//     // measure training time
//     t1 = high_resolution_clock::now();
//     const vector<double> weights = model.fit(X_train, y_train, 1e-3, 1e-6, 30);
//     t2 = high_resolution_clock::now();
//     const auto duration_train = duration_cast<milliseconds>(t2 - t1).count();

//     cout << "Training time: " << duration_train << " ms" << endl;

//     cout << "=====METRICS=====" << endl;

//     // measure prediction time on test set
//     t1 = high_resolution_clock::now();
//     const vector<int> y_test_predict = model.predict(X_test);
//     t2 = high_resolution_clock::now();
//     const auto duration_predict_test = duration_cast<milliseconds>(t2 - t1).count();

//     cout << "Prediction time (test): " << duration_predict_test << " ms" << endl;

//     cout << "Accuracy: " << accuracy_score(y_test, y_test_predict) << endl;
//     cout << "F1 score: " << f1_score(y_test, y_test_predict) << endl;
// }

// void manual_test(int argc, char* argv[]) {
//     const int core = atoi(argv[1]);
//     string path = argv[2];
//     const double test_ratio = atof(argv[3]);  // assuming test_size as fraction
//     test_data(path, core, test_ratio);
// }

// int main(const int argc, char* argv[]) {
//     manual_test(argc, argv);

//     // random_test(argc, argv);
//     return 0;
// }
