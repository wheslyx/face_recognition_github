/*!
@file facial_landmarks.h

@brief a c++ program to predict the facial landmarks.

@details 
Created on : March 19, 2020
Author : Diego Hurtado de Mendoza 
Authon: Cesar Segura Del Rio
*/


#include <chrono>
#include <random>
#include <math.h>
#include "model.hpp"
#include "../FaceDetector/face_detector.hpp"
#include "../util.h"

class FacialLandmarks{
	
	private:
		std::shared_ptr<Model> modelFace; //!< the model for face landmarks from pytorch
		std::shared_ptr<Model> modelEyes; //!< the model only for eyes landmarks from pytorch
		std::pair<int,int> rangeEyes; //!< range of position of the eyes landmarks
		
		FaceDetector faceDetector; //!< face detector 
		std::mt19937_64 rng; //!< a random number generator
		
		/*!
		@brief give the current time in milliseconds.
			
		@param none

		@return time in milliseconds
		*/
		long double getTime();
		
	public:
	
		
		/*!
		@brief The constructor
		
		@details ***********************************************************************************************
		
		Initialize the attributes and check if cuda is available.
				
		@param[in] model an object of the class Model
		@param[in] faceDetector the face detector
		*/
		FacialLandmarks(Model model, FaceDetector faceDetector);

		/*!
		@brief The destructor
		
		@details ***********************************************************************************************
		
		Destroys FaceLandmarks object
				
		*/
		~FacialLandmarks();
		
		/*!
		@brief Add a model trained only with eyes
		
		@param[in] model an object of class Model
		@param[in] rangeEyes a pair representing the range (inclusive) of the landmarks that belongs to the eyes
		
		@return none 
		*/
		void addEyesModel(Model model, std::pair<int,int> rangeEyes);
		
		
		/*!
		@brief Prediction of the facial landmarks
		
		@details ***********************************************************************************************
		
		First it detect the cropped face and transform it to a tensor. Then it passes it to the model
		and gets the landmarks.
		
		If there is an eyes' model (only trained with eyes), it will replace all the eyes 
		landmarks with the output of this model.
		
		@param[in] img the img input
		 
		@return a vector of opencv Points 
		*/
		std::vector<cv::Point2f> getLandmarks(cv::Mat img);
		
		/*!
		@brief Test the performance of the model with a directory full of images
		
		@details ***********************************************************************************************
		
		First, all the images are opened and randomly shuffled. Then, for each image, its facial landmarks are predicted.
		
		Finally it will calculate the average time consumed for image in seconds
		
		@param[in] pathDir the directory of the images
		@param[in] showImages true if we want to see the landmarks prediction of the image
		
		@return none
		*/
		void testPerformance(std::string pathDir , bool showImages);
		
		/*!
		@brief Predict the facial landmarks for a video
		
		@details ***********************************************************************************************
		If pathInputVideo is empty, the camera will be used; otherwise the video passed in the path will be used
		
		If pathOutputVideo is not empty, it will save the video in the path.
		
		In all the cases, it will also show the video with the predicted landmarks
		
		@param[in] pathInputVideo path of the input video
		@param[in] pathOutputVideo path ot the output video
		
		@return none
		*/
		
		void runVideo(std::string pathInputVideo, std::string pathOutputVideo);
		
};
