/*!
@file model.h

@brief The C++ implementation for a pytorch model prediction.

@details 
Created on : March 19, 2020
Author : Diego Hurtado de Mendoza
Author : Cesar Segura Del Rio  
*/

#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


class Model{
	
	private:
		std::shared_ptr<torch::jit::script::Module> net; //!< the neural network for face landmarks
		int szModel;  //!< the size of the images in the trained model
		bool swapped; //!< true if we have to swap the coordinates output
		torch::Tensor mean; //!< mean for normalization
		torch::Tensor std; //!< standard deviation for normalization
		
		/*!
		@brief warms up the model with a random tensor.
		
		@details ***********************************************************************************************
		
		The first prediction take too much time to be processed
		that's why is better to warm up the model with a random tensor.
				
		@param none

		@return none
		*/
		void warmUp();
		
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
		void rescale(cv::Point2f &P, int rows, int cols);
		
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
		torch::Tensor imageToTensor(cv::Mat image);
		
		
		
	
	public:
		Model(); //!< default constructor
		
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
		Model(std::string pathModel, int szModel, bool swapped, std::vector<double> mean, std::vector<double> stdv);
		
		/*!
		@brief The destructor
		
		@details ***********************************************************************************************
		
		Destroys Model.

		*/
		~Model();

		/*!
		@brief Prediction of the facial landmarks
		
		@details ***********************************************************************************************
		
		First, the image is to a tensor. Then it passes it to the pytorch model and it gets
		the tensor representing the landmarks.
		
		@param[in] face the image representing the image face
		
		@return a vector of opencv Points 
		*/
		std::vector<cv::Point2f> getLandmarks(cv::Mat face);
};
#endif
