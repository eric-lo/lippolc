#include <limits>
#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace lippolc {

// Linear regression model
template <class T>
class LinearModel
{
public:
    double a = 0; // slope
    long double b = 0; // intercept

    LinearModel() = default;
    LinearModel(double a, long double b) : a(a), b(b) {}
    explicit LinearModel(const LinearModel &other) : a(other.a), b(other.b) {}

    inline int predict(T key) const
    {
        return std::floor(a * static_cast<long double>(key) + b);
    }

    inline double predict_double(T key) const
    {
        return a * static_cast<long double>(key) + b;
    }
};

}
