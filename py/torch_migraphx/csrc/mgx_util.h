#include <torch/extension.h>

namespace torch_migraphx {
namespace util {

const std::unordered_map<torch::ScalarType, std::string>& get_torch_mgx_type_map();
const std::unordered_map<std::string, torch::ScalarType>& get_mgx_torch_type_map();
torch::ScalarType MGXDataTypeToScalarType(std::string t);
std::string ScalarTypeToMGXDataType(torch::ScalarType t);

}
}