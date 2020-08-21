/*!
@file facial_landmarks.h

@brief a c++ program to predict the facial landmarks.

@details 
Created on : March 19, 2020
Author : Diego Hurtado de Mendoza 
Authon: Cesar Segura Del Rio
*/

#include "facial_landmarks.hpp"

/*!
@brief give the current time in milliseconds.
	
@param none
@return time in milliseconds
*/
long double FacialLandmarks::getTime() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}


/*!
@brief The constructor

@details ***********************************************************************************************

Initialize the attributes and check if cuda is available.
		
@param[in] model an object of the class Model
@param[in] faceDetector the face detector
*/
FacialLandmarks::FacialLandmarks(Model model, FaceDetector faceDetector){

	this->modelFace = std::make_shared<Model>(model);
	this->faceDetector = faceDetector;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Working on GPU." << std::endl;
	} else {
		std::cout << "Error! No Cuda available" << std::endl;
		return; 
	}
	this->rng = std::mt19937_64(std::chrono::steady_clock::now().time_since_epoch().count());

}

/*!
@brief The destructor

@details ***********************************************************************************************

Destroys FaceLandmarks object
		
*/
FacialLandmarks::~FacialLandmarks(){};

/*!
@brief Add a model trained only with eyes

@param[in] model an object of class Model
@param[in] rangeEyes a pair representing the range (inclusive) of the landmarks that belongs to the eyes

@return none 
*/
void FacialLandmarks::addEyesModel(Model model, std::pair<int,int> rangeEyes){

	this->rangeEyes = rangeEyes;
	this->modelEyes = std::make_shared<Model>(model);
}


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
std::vector<cv::Point2f> FacialLandmarks::getLandmarks(cv::Mat img) {

	std::vector<cv::Rect> faces = faceDetector.getFaces(img);
	if(faces.empty()){
		return {};
	}
	
	cv::Rect rectFace = faces[0];
	cv::rectangle(img,rectFace, cv::Scalar( 255, 0, 0 ));
	cv::Mat face = img(rectFace);
	std::vector<cv::Point2f> landmarks = modelFace->getLandmarks(face);
	
	if(modelEyes != nullptr){
		std::vector<cv::Point2f> eyesLandmarks = modelEyes->getLandmarks(face);
		for(int i = rangeEyes.first ; i <= rangeEyes.second; i++){
			landmarks[i] = eyesLandmarks[i - rangeEyes.first];
		}
	}
	for(int i = 0; i < (int) landmarks.size(); i++){
		landmarks[i].x += rectFace.x;
		landmarks[i].y += rectFace.y;
	}
	return landmarks;
}

/*!
@brief Test the performance of the model with a directory full of images

@details ***********************************************************************************************

First, all the images are opened and randomly shuffled. Then, for each image, its facial landmarks are predicted.

Finally it will calculate the average time consumed for image in seconds

@param[in] pathDir the directory of the images
@param[in] showImages true if we want to see the landmarks prediction of the image

@return none 
*/
void FacialLandmarks::testPerformance(std::string pathDir , bool showImages){
	std::vector<std::string> filenames; 
	double totalTime = 0;
	cv::glob(pathDir, filenames); 
	
	std::cout << "Processing images from " << pathDir << "..." << std::endl;
	std::cout << "# of files = " << filenames.size() << std::endl;
	
	std::shuffle(filenames.begin() , filenames.end() , rng);
	int cnt = 0;
	for(int i = 0; i < (int) filenames.size(); i++) {
		cv::Mat img = cv::imread(filenames[i]);
		
		if(!img.data){
			std::cout << "Problem loading image " << filenames[i] << std::endl;	
		} else {
			double ini = getTime();
			std::vector<cv::Point2f> points = getLandmarks(img);
			double fin = getTime();
			if(!points.empty()){
				if(showImages){
					Util::putLandmarks(img , points);
					Util::showImage(img);
				}
				totalTime += fin - ini;
				cnt++;
			} else{
				std::cout << "No face found in " << filenames[i] << std::endl;
			}
		}
		int step = round(filenames.size() / 10.0);
		if(i % step == step - 1){
			int percent = round(100.0 * (i + 1) / filenames.size());
			std::cout << percent  << "%  completed" << std::endl;
		}
	}
	std::cout << "Testing finished" << std::endl;
	if(cnt != 0){
		std::cout << std::fixed << std::setprecision(8) << "avg time = " << totalTime / (1000.0 * cnt) << " seg / imagen" << std::endl;
	}
}

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

void FacialLandmarks::runVideo(std::string pathInputVideo, std::string pathOutputVideo) {
	cv::VideoCapture cap;
	if(pathInputVideo.empty()){
		cap = cv::VideoCapture(0);
	} else{
		cap = cv::VideoCapture(pathInputVideo);
	}
	
	if(!cap.isOpened() ){
		if(pathInputVideo.empty()){
			std::cout << "Error in camera" << std::endl;
		} else{
			std::cout << "No video found" << std::endl;
		}
		return;
	}

	int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH); 
	int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	cv::VideoWriter video;
	if(!pathOutputVideo.empty()){
		video = cv::VideoWriter(pathOutputVideo ,cv::VideoWriter::fourcc('M','J','P','G'),10, cv::Size(frame_width,frame_height));
	} 
	
	while(true){
		cv::Mat frame;
		cap >> frame;
		if(frame.empty()){
			break;
		}
		std::vector<cv::Point2f> points = getLandmarks(frame);
		Util::putLandmarks(frame, points );
		Util::showImage(frame);
		if(!pathOutputVideo.empty()){
			video.write(frame);
		} 
		char c = (char) cv::waitKey(25);
		if(c == 27 || c == 32){
			break;
		}
	}
	cap.release();
	if(!pathOutputVideo.empty()){
		video.release();
	} 
	cv::destroyAllWindows();
}
