#include <torch/torch.h>
#include <torch/script.h>
using namespace std;
using namespace torch;
using namespace torch::nn;

int main() {

	std::shared_ptr<torch::jit::script::Module> net; //!< the neural network for face landmarks
	net = torch::jit::load("Retrain_pretrained_model_8epoch.pt", Device("cuda"));
	int szModel = 160;  //!< the size of the images in the trained model
	bool swapped; //!< true if we have to swap the coordinates output
	Tensor mean; //!< mean for normalization
	Tensor std; //!< standard deviation for normalization
	auto dev;
	cout << "warming up..." << endl;
	for(int i = 0; i < 5; i++){
		dev = warmUp();
	}
	cout << "warm up ended" << endl << endl;
	cout << dev << endl;
	return 0;
}

auto warmUp(){
			net->eval();
			NoGradGuard guard;
			torch::Tensor tensor = torch::randn({1,3,szModel, szModel});
			tensor = tensor.to("cuda");
			vector<torch::jit::IValue> inputs;
			inputs.push_back(tensor);
			Tensor output = net->forward(inputs).toTensor();
			output = output.to("cpu");
			return output;
		}
