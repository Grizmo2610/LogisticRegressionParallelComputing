#ifndef MODEL_H
#define MODEL_H

#include <vector>
using namespace std;

class LogisticRegression {
private:
    vector<double> weights;
    double bias{};
    int core{};
    bool parallel{};

public:
    LogisticRegression();
    explicit LogisticRegression(int core);

    vector<double> fit(const vector<vector<double>>& X,
                       const vector<int>& y,
                       double lr,
                       double epsilon,
                       int max_iter,
                       double lambda_l2);

    [[nodiscard]] vector<int> predict(const vector<vector<double>>& X,
                        double thresh = 0.5) const;
};

#endif
