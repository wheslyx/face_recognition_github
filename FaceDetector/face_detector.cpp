/*!  
@file facial_detector.h 

@brief a c++ program to get the rectangle of the cropped face using opencv.

@details 
Created on : April 24, 2020
Author : Diego Hurtado de Mendoza 
Author : Cesar Segura Del Rio
*/
#include "face_detector.hpp"
	
/*!
@brief warms up the model with a random image.

@details ***********************************************************************************************

The first prediction take too much time to be processed
that's why is better to warm up the model with a random image.
		
@param none
@return none
*/
void FaceDetector::warmUp() {
	cv::Mat img(cv::Size(300,300), CV_8UC3);
	cv::randu(img, cv::Scalar(0,0,0), cv::Scalar(255,255,255));
	cv::Mat blob = cv::dnn::blobFromImage(img, 1.0f, cv::Size(300,300), 0.0f);
	model.setInput(blob);
	model.forward();
}

/*!
@brief The default constructor

@details ***********************************************************************************************

Initialize and warms up the deep neural network from caffe
		
@param[in] prototxt the path for the prototxt of the model (.txt)
@param[in] prototxt the path for the model (.caffemodel)
*/
FaceDetector::FaceDetector() {
	std::cout << "Loading face detector's model..." << std::endl;
	model = cv::dnn::readNetFromCaffe(FACE_NETWORK, FACE_WEIGHTS);
	model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	
	std::cout << "Model loaded" << std::endl;
	std::cout << "warming up face detector..." << std::endl;
	warmUp();
	std::cout << "warm up of face detector ended" << std::endl;
}


/*!
@brief The default  destructor

@details ***********************************************************************************************

Destroy Face Detector and release memory

*/
FaceDetector::~FaceDetector() {};

/*!
@brief Get the rectangles of the faces in the image

@details ***********************************************************************************************

The image is resized to 300 x 300 for the dnn model , the it calculates the coordinates of the
corner of the rectangle(upper left and lower right corners) in the range [0 , 1 ] and then they are
rescale to the originale scale of the image

@param[in] imgInput the input image
@return vector of opencv rect, the rectangles of the faces
*/
std::vector<cv::Rect> FaceDetector::getFaces(cv::Mat imgInput) {

	cv::Mat img;
	
	cv::resize(imgInput, img, cv::Size(300, 300) );
	
	cv::Mat blob = cv::dnn::blobFromImage(img , 1.0 , cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), true);
	model.setInput(blob);
	cv::Mat faces = model.forward();
	
	if(faces.size[2] == 0){
		return {};
	}
	
	std::vector<cv::Rect> rectangles;
	
	for(int i = 0; i < faces.size[2]; i++){
		cv::Vec<int , 4> index(0,0, i , 2);
		double confidence = faces.at<float>(index);
		if(confidence < 0.5){
			break;
		}
		index = cv::Vec<int,4>(0,0,i,3);
		float xMin = faces.at<float>(index);
		index = cv::Vec<int,4>(0,0,i,4);
		float yMin = faces.at<float>(index);
		index = cv::Vec<int,4>(0,0,i,5);
		float xMax = faces.at<float>(index);
		index = cv::Vec<int,4>(0,0,i,6);
		float yMax = faces.at<float>(index);
		
		xMin *= imgInput.cols;
		yMin *= imgInput.rows;
		xMax *= imgInput.cols;
		yMax *= imgInput.rows;
		
		int x = std::max((int) xMin , 0);
		int y = std::max((int) yMin , 0);
		x = std::min(x , imgInput.cols);
		y = std::min(y , imgInput.rows);
		
		int w = std::min((int)xMax - x , imgInput.cols - x );
		int h = std::min((int)yMax - y , imgInput.rows - y );
		w = std::max(w , 0);
		h = std::max(h , 0);
		if(w < 20 || h < 20) continue; //too small rectangle
	
		rectangles.push_back(cv::Rect(x, y , w , h));
	}
	return rectangles;
}
