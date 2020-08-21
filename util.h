/*!
@file util.h

@brief Useful functions

@details 
Created on : April 30, 2020
Author : Diego Hurtado de Mendoza 
*/

#ifndef UTIL_H
#define UTIL_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


namespace Util{
	/*!
	@brief show the input image on a pop up window

	@param[in] image input image

	@return none
	*/
	void showImage(cv::Mat image){
		std::string name = "display";
		cv::namedWindow( name , cv::WINDOW_NORMAL   );// Create a window for display.
		cv::resizeWindow(name , 500, 500);
		cv::imshow( name , image );               // Show our image inside it.
		cv::waitKey(0);
	}

	/*!
	@brief Put the landmarks on the image.

	@details ***********************************************************************************************

	It change the input image to an image with the landmarks on it.
	It uses the function circle of opencv
		
	@param[out] image an opencv Mat that represents the image to put the landmarks on
	@param[in] landmarks a vector of opencv points that represents the landmarks
	@param[in] r the radius of the points
	@param[in] color an opencv scalar that represent the color of the landmarks RGB

	@return none
	*/
	void putLandmarks(cv::Mat &image, std::vector<cv::Point2f> landmarks, int r = 2 , cv::Scalar color = cv::Scalar(0,255,0)){
		for(cv::Point2f P : landmarks){
			cv::circle(image , P, r , color,cv::FILLED);
		}
	}

	/*!
	@brief Put the landmarks of different faces on the image. It's a generalization of the method above

	@details ***********************************************************************************************

	It change the input image to an image with the landmarks on it.
	It uses the function circle of opencv
		
	@param[out] image an opencv Mat that represents the image to put the landmarks on
	@param[in] allLandmarks a vector of vector of opencv points that represents the landmarks of different people
	@param[in] r the radius of the points
	@param[in] color an opencv scalar that represent the color of the landmarks RGB

	@return none
	*/
	void putLandmarks(cv::Mat &image, std::vector<std::vector<cv::Point2f>> allLandmarks, int r = 2 , cv::Scalar color = cv::Scalar(0,255,0)){
		for(int i = 0; i < (int) allLandmarks.size(); i++){
			putLandmarks(image , allLandmarks[i]);
		}
	}
};

#endif
