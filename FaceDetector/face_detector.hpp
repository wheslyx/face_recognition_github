/*!
@file facial_detector.h

@brief a c++ program to get the rectangle of the cropped face using opencv.

@details 
Created on : April 24, 2020
Author : Diego Hurtado de Mendoza
Author : Cesar Segura Del Rio 
*/

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "../constants.h"

class FaceDetector{
	private:
		cv::dnn::Net model; //!< face detector from opencv dnn
		
		/*!
		@brief warms up the model with a random image.
		
		@details ***********************************************************************************************
		
		The first prediction take too much time to be processed
		that's why is better to warm up the model with a random image.
				
		@param none

		@return none
		*/
		void warmUp();
		
	public:
	
		/*!
		@brief The default  constructor
		
		@details ***********************************************************************************************
		
		Initialize and warms up the deep neural network from caffe
				
		@param[in] prototxt the path for the prototxt of the model (.txt)
		@param[in] prototxt the path for the model (.caffemodel)
		*/
		FaceDetector();
		
		/*!
		@brief The default  destructor
		
		@details ***********************************************************************************************
		
		Destroy Face Detector and release memory
		
		*/
		~FaceDetector();


		/*!
		@brief Get the rectangles of the faces in the image
		
		@details ***********************************************************************************************
		
		The image is resized to 300 x 300 for the dnn model , the it calculates the coordinates of the
		corner of the rectangle(upper left and lower right corners) in the range [0 , 1 ] and then they are
		rescale to the originale scale of the image
		
		@param[in] imgInput the input image
	
		@return vector of opencv rect, the rectangles of the faces
		*/
		std::vector<cv::Rect> getFaces(cv::Mat imgInput);
};

#endif
