#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace medaxis
{
namespace radiomics_ext
{

std::map<std::string, double> first_order_features(const std::vector<double>& values);

} // namespace radiomics_ext
} // namespace medaxis
