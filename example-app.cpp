// #include <torch/script.h> // One-stop header.
// #include <iostream>
// #include <memory>

// int main() {
//   // Deserialize the ScriptModule from a file using torch::jit::load().
//   std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("Z:\\PhospheneAI\\Inference\\code\\imdn_48_8_x2.pt");

//   assert(module != nullptr);
//   std::cout << "ok\n";
//   // Create a vector of inputs.
//   std::vector<torch::jit::IValue> inputs;
//   inputs.push_back(torch::ones({ 1, 3, 224, 224 }));

//   // Execute the model and turn its output into a tensor.
//   at::Tensor output = module->forward(inputs).toTensor();

//   std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
//   while (1);
// }

#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({5, 3});
  std::cout << tensor << std::endl;
}