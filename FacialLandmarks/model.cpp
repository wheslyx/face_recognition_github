/*!
@file model.h    

@brief The C++ implementation for a pytorch model prediction.

@details 
Created on : March 19, 2020
Author : Diego Hurtado de Mendoza 
Author : Cesar Segura Del Rio  
*/

#include "model.hpp"
		
/*!
@brief warms up the model with a random tensor.

@details ***********************************************************************************************

The first prediction take too much time to be processed
that's why is better to warm up the model with a random tensor.
		
@param none
@return none
*/
void Model::warmUp() {
	net->eval();
	torch::NoGradGuard guard;
	torch::Tensor tensor = torch::randn({1,3,szModel, szModel});
	tensor = tensor.to("cuda");
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(tensor);
	torch::Tensor output = net->forward(inputs).toTensor();
	output = output.to("cpu");
}

/*!
@brief It recieves a landmark in scale [-1 , 1] and	transforms it to its original scale.
		
@details ***********************************************************************************************

The output of the model will be in the range [-1 , 1] (real number)
To return to its original scale (let's say its [0 , x] ), we do the following :

1) Add 1 to the result so its range will be [0 , 2]

2) Divide the result by two so its range will be [0 , 1]

3) Multiply the result by x so its range will be [0 , x]
		
@param[out] P the point to be rescale
@param[in] rows the rows of the original Mat image
@param[in] cols the columns of the original Mat image
@return none
*/
void Model::rescale(cv::Point2f &P, int rows, int cols){
	//assuming that points are in range [-1 , 1]
	//we will transform them to its original scale
	P.x = (1 + P.x) * cols / 2;
	P.y = (1 + P.y) * rows / 2;
}

/*!
@brief A function that converts an opencv image to a tensor 

@details ***********************************************************************************************

The image should have 3 channels (if it's grayscale, you should copy the same tensor for each channel)

It will execute the following actions :

1) Conver from RGB to gray scale (if it's already grayscale it doesn't matter)

2) Divide all the value by 255 so the range of the tensor is between 0 and 1

3) Transform it to a torch tensor

4) Permute the dimensions of the tensor (for Pytorch format)

5) Normalize the tensor with the mean and standard deviation provided in the constructor
		
@param[in] image input image
@return the tensor of the image
*/
torch::Tensor Model::imageToTensor(cv::Mat image) {
	if(!image.data ){
		std::cout <<  "No image" << std::endl;
	}
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
	cv::resize(image, image, cv::Size(szModel, szModel) );
	int channels = 3;
	torch::Tensor tensor = torch::from_blob(image.data, {1, image.rows, image.cols , channels});
	tensor = tensor.permute({0, 3, 1, 2});
	tensor = tensor.to(torch::kFloat);
	tensor = (tensor - mean) / std; //normalization
	return tensor;
}


Model::Model(){} //!< default constructor

/*!
@brief The constructor

@details ***********************************************************************************************

Initialize the attributes, incluiding constructing the mean and std tensor and checking if cuda is available.

It also load the model and the cascade for face detection. Last but not least, it warms up the model.
		
@param[in] pathModel the path of the model
@param[in] szModel the size of the images in the trained model
@param[in] swapped true if we have to swap the coordinates output (for fastai models)
@param[in] mean the mean of the 3 channels for normalization
@param[in] std the standard deviation of the 3 channels for normalization
*/
Model::Model(std::string pathModel, int szModel, bool swapped, std::vector<double> mean, std::vector<double> std){
	//pathModel should be a .pt
	this->szModel = szModel;
	this->swapped = swapped;
	this->mean = torch::randn({1, 3, szModel , szModel});
	this->std = torch::randn({1, 3, szModel , szModel});
	for(int channels = 0; channels < 3; channels++){
		for(int i = 0; i < szModel ; i++){
			for(int j = 0; j < szModel; j++){
				this->mean[0][channels][i][j] = mean[channels];
				this->std[0][channels][i][j] = std[channels];
			}
		}
	}
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		std::cout << "Loading pytorch jit model..." << std::endl;
		net = std::make_shared<torch::jit::script::Module>(torch::jit::load(pathModel, torch::Device("cuda")));
		std::cout << "Model loaded successfully" << std::endl;
	}
	catch (const c10::Error& e) {
		std::cout << "no path found for the model" << std::endl;
		return ;
	}
	
	std::cout << "warming up..." << std::endl;
	for(int i = 0; i < 5; i++){
		warmUp();
	}
	std::cout << "warm up ended" << std::endl;
}
	/*!
	@brief The destructor
	
	@details ***********************************************************************************************
	
	Destroys Model.
	*/
	Model::~Model(){};
	
	/*!
	@brief Prediction of the facial landmarks
	
	@details ***********************************************************************************************
	
	First, the image is to a tensor. Then it passes it to the pytorch model and it gets
	the tensor representing the landmarks.
	
	@param[in] face the image representing the image face
	
	@return a vector of opencv Points 
	*/
	std::vector<cv::Point2f> Model::getLandmarks(cv::Mat face){
		torch::Tensor tensor = imageToTensor(face);
		net->eval();
		torch::NoGradGuard guard;
		
		tensor = tensor.to("cuda");
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(tensor);
		torch::Tensor output = net->forward(inputs).toTensor();
		output = output.to("cpu");
		auto aux = output.sizes();
		int szOutput = aux[1];
		
		std::vector<cv::Point2f> landmarks;
		for(int i = 0; i < szOutput ; i += 2){
			cv::Point2f P(output[0][i].item<double>() , output[0][i + 1].item<double>());
			if(swapped){
				std::swap(P.x, P.y);
			}
			rescale(P , face.rows, face.cols);
			landmarks.push_back(P);
		}
		return landmarks;
	}

