  
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <memory> //Better memory management
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
#include <conio.h>
//#include <dir>
#include <process.h>
#include <stdio.h>


using namespace cv;
using namespace std;


cv::Mat torchTensortoCVMat(torch::Tensor& tensor)
{	
	tensor = tensor.squeeze().detach();
	tensor = tensor.permute({ 1, 2, 0 }).contiguous();
	tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
	tensor = tensor.to(torch::kCPU);
	int64_t height = tensor.size(0);
	int64_t width = tensor.size(1);
	cout << "Into torchTensortoCVMat Function da..";
	cv::Mat mat = cv::Mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr<uchar>());
	return mat.clone();
}

std::string getPathName(const string& s) {

	char sep = '/';

	#ifdef _WIN32
		sep = '\\';
	#endif

	std::size_t i = s.rfind(sep, s.length());
	if (i != string::npos) {
		return(s.substr(0, i));
	}

	return("");
}

int main(int argc, const char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage:<path-to-exported-script-module> "
			"<path-to-image-file>" << std::endl;
		return -1;
	}

	torch::jit::script::Module module = torch::jit::load(argv[1]);
	assert(module != nullptr);

	torch::DeviceType device_type;

	std::cout << "Training on CPU" << std::endl;
	device_type = torch::kCPU;
	
	torch::Device device(device_type);
	module.to(device);

	cv::Mat origin_image;
	origin_image = cv::imread(argv[2]);
	

	cv::Mat img_float;
	origin_image.convertTo(img_float, CV_32FC3, 1.0 / 255);

	auto img_tensor = torch::from_blob(img_float.data, { 1, img_float.rows, img_float.cols, 3 }).to(device);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });

	// Create a vector of inputs.
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);
	//auto img_output;
	clock_t startTime, endTime;
	startTime = clock();
	torch::Tensor output = module.forward(inputs).toTensor();
	cout << output.sizes() << endl;

	cout << output.size(0) << " " << output.size(1) << " " << output.size(2) << " " << output.size(3) << endl;
	cv::Mat result = torchTensortoCVMat(output);
	cout << "Saving image";

	cv::imwrite("./output/op_one.jpg", result);
	endTime = clock();
	std::cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
}
