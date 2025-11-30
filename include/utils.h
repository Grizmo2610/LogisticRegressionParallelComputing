#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <random>

using namespace std;

//-----------------------------------
// Math functions
//-----------------------------------
double sigmoid(double x);

vector<vector<double> > dot(const vector<vector<double> > &A,
                            const vector<vector<double> > &B,
                            int n_threads = 0);

vector<vector<double> > transpose(const vector<vector<double> > &matrix,
                                  int n_threads = 0);

double norm(const vector<double> &a,
            const vector<double> &b,
            int n_threads = 0);

//-----------------------------------
// Utility
//-----------------------------------
void printMatrix(const vector<vector<double> > &matrix);

//-----------------------------------
// Metrics
//-----------------------------------
double accuracy_score(const vector<int> &y_true,
                      const vector<int> &y_predict,
                      int n_threads = 0);

double recall_score(const vector<int> &y_true,
                    const vector<int> &y_predict,
                    int n_threads = 0);

double precision_score(const vector<int> &y_true,
                       const vector<int> &y_predict,
                       int n_threads = 0);

double f1_score(const vector<int> &y_true,
                const vector<int> &y_predict,
                int n_threads = 0);

//-----------------------------------
// CSV loader
//-----------------------------------
vector<vector<double> > read_csv(const string &filename);


void generate_random_data(int n_samples, int n_features,
                          vector<vector<double> > &X,
                          vector<int> &y,
                          double noise = 0.2, int random_state = 42);


void train_test_split(const vector<vector<double>>& X,
                      const vector<int>& y,
                      vector<vector<double>>& X_train,
                      vector<vector<double>>& X_test,
                      vector<int>& y_train,
                      vector<int>& y_test,
                      double test_size = 0.2,
                      bool shuffle = true,
                      unsigned int random_state = 42);

vector<double> flatten(const vector<vector<double>>& X);

double logistic_loss(
    const vector<double>& w,
    double bias,
    const vector<double>& X_flat,
    const vector<int>& y,
    int N,
    int d,
    double lambda_l2
);
#endif
