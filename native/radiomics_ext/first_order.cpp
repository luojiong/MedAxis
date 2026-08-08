#include "first_order.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace medaxis
{
namespace radiomics_ext
{

std::map<std::string, double> first_order_features(const std::vector<double>& values)
{
    if (values.empty())
    {
        throw std::invalid_argument("first_order_features: values must not be empty");
    }

    const double count = static_cast<double>(values.size());
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / count;

    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    double second_moment = 0.0;
    double third_moment = 0.0;
    double fourth_moment = 0.0;
    for (const double value : values)
    {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        const double centered = value - mean;
        const double squared = centered * centered;
        second_moment += squared;
        third_moment += squared * centered;
        fourth_moment += squared * squared;
    }

    const double variance = second_moment / count;
    const double standard_deviation = std::sqrt(variance);
    const double skewness = standard_deviation > 0.0
        ? (third_moment / count) / std::pow(standard_deviation, 3.0)
        : 0.0;
    const double kurtosis = standard_deviation > 0.0
        ? (fourth_moment / count) / std::pow(standard_deviation, 4.0) - 3.0
        : 0.0;

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t middle = sorted.size() / 2;
    const double median = sorted.size() % 2 == 0
        ? (sorted[middle - 1] + sorted[middle]) / 2.0
        : sorted[middle];

    return {
        {"count", count},
        {"mean", mean},
        {"variance", variance},
        {"standard_deviation", standard_deviation},
        {"minimum", minimum},
        {"maximum", maximum},
        {"median", median},
        {"skewness", skewness},
        {"kurtosis", kurtosis},
    };
}

} // namespace radiomics_ext
} // namespace medaxis
