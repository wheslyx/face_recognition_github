#include <torch/torch.h>
#include <torch/script.h>
using namespace std;
using namespace torch;
using namespace torch::nn;

int main() {

	std::shared_ptr<torch::jit::script::Module> net; //!< the neural network for face landmarks
	net = make_shared<torch::jit::script::Module>(torch::jit::load("Retrain_pretrained_model_8epoch.pt", Device("cuda")));
	cout << "warming up..." << endl;
	for(int i = 0; i < 5; i++){
		warmUp();
	}
	cout << "warm up ended" << endl << endl;
	return 0;
}

void warmUp(){
			net->eval();
			NoGradGuard guard;
			torch::Tensor tensor = torch::randn({1,3,160, 160});
			tensor = tensor.to("cuda");
			vector<torch::jit::IValue> inputs;
			inputs.push_back(tensor);
			Tensor output = net->forward(inputs).toTensor();
			output = output.to("cpu");
			cout << output << endl;
		}
