#include <torch/torch.h>
#include <torch/script.h>
using namespace std;
using namespace torch;
using namespace torch::nn;

int main() {

	std::shared_ptr<torch::jit::script::Module> net; //!< the neural network for face landmarks
	net = make_shared<torch::jit::script::Module>(torch::jit::load("/var/fcs_res/MobileNetv2_Adam_01_Eyes_Norm_Without_Flip_v2_70epochs.pt", Device("cuda")));
	cout << "warming up..." << endl;
	for(int i = 0; i < 5; i++){
		net->eval();
		NoGradGuard guard;
		torch::Tensor tensor = torch::randn({1,3,256, 256});
		tensor = tensor.to("cuda");
		vector<torch::jit::IValue> inputs;
		inputs.push_back(tensor);
		Tensor output = net->forward(inputs).toTensor();
		output = output.to("cpu");
		cout << output << endl;
	}
	cout << "warm up ended" << endl << endl;
	return 0;
}

