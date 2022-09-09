#include "mgx_util.h"

namespace torch_migraphx {
namespace util {

namespace {
const std::unordered_map<torch::ScalarType, std::string>& torch_mgx_type_map() {
  static const std::unordered_map<torch::ScalarType, std::string> at_mgx_type_map = {
      {torch::kFloat32, "float_type"},
      {torch::kFloat16,  "half_type"},
      {torch::kFloat64, "double_type"},
      {torch::kInt32, "int32_type"},
      {torch::kInt16, "int16_type"},
      {torch::kInt64, "int64_type"},
      {torch::kInt8, "int8_type"},
      {torch::kUInt8, "uint8_type"},
      {torch::kBool, "bool_type"}};
  return at_mgx_type_map;
}

const std::unordered_map<std::string, torch::ScalarType>& mgx_torch_type_map() {
  static const std::unordered_map<std::string, torch::ScalarType> mgx_at_type_map = {
      {"float_type", torch::kFloat32},
      {"half_type", torch::kFloat16},
      {"double_type", torch::kFloat64},
      {"int32_type", torch::kInt32},
      {"int16_type", torch::kInt16},
      {"int64_type", torch::kInt64},
      {"int8_type", torch::kInt8},
      {"uint8_type", torch::kUInt8},
      {"bool_type", torch::kBool}};
  return mgx_at_type_map;
}
} // namespace

const std::unordered_map<torch::ScalarType, std::string>& get_torch_mgx_type_map() {
  return torch_mgx_type_map();
}

const std::unordered_map<std::string, torch::ScalarType>& get_mgx_torch_type_map() {
  return mgx_torch_type_map();
}

torch::ScalarType MGXDataTypeToScalarType(std::string t) {
  auto mgx_torch_type_map = get_mgx_torch_type_map();
  return mgx_torch_type_map.at(t);
}

std::string ScalarTypeToMGXDataType(torch::ScalarType t) {
  auto torch_mgx_type_map = get_torch_mgx_type_map();
  return torch_mgx_type_map.at(t);
}

} // namespace util
} // namespace torch_migraphx