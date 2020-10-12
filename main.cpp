#include "FaceRecognition/face_recognition.hpp"
#include "FaceDetector/face_detector.hpp"
#include "constants.h"
int main() {
	
	FaceDetector faceDetector;
	Model modelEyes(EYESLANDMARKS, 256 , false , {0.485, 0.456, 0.406} ,{0.229, 0.224, 0.225});
	/*
	FaceRecognition recognizer(faceDetector, FACE_RECOGNITION);
	
	recognizer.addEyesModel(modelEyes, {20, 39}, {0, 19});
	
	std::string trainPath = "/media/disk/dataset_test";
	
	recognizer.train(trainPath, RECOGNITION_WEIGHTS);
	
	std::string svmModel = "/media/disk/svmModel/recognizer.xml";  
	*/
	std::string labelsTxt = "/media/disk/svmModel/labels.txt";
	
	FaceRecognition recognizer(FACE_RECOGNITION , RECOGNITION_WEIGHTS, labelsTxt);
	recognizer.addEyesModel(modelEyes, {20, 39}, {0, 19});
	
	//real time
	std::string pathInputVideo = "";
	std::string pathOutputVideo= "";

	// This is the run video attribute
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
		return 0;
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
		std::vector<cv::Rect> rectFaces = faceDetector.getFaces(frame);
		if (rectFaces.empty()) {
			continue;
		}
		std::cout << "rectangles : " << rectFaces << std::endl;
		cv::Rect rectFace = rectFaces[0];
		std::cout << "rectangle : " << rectFace << std::endl;
		long double startrun = recognizer.getTime();
		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		std::pair<std::string,float> prediction = recognizer.recognize(frame, rectFace);
		long double endrun = recognizer.getTime();
		std::string name = prediction.first;
		float recognition_confidence = prediction.second;
		if(!(name == "no face detected" || recognition_confidence < RECOGNITION_THRESHOLD)){
			cv::rectangle(frame ,rectFace, cv::Scalar( 255, 0, 0 ));
			int yDraw = rectFace.y - 10;
			if(yDraw <= 10){
				yDraw = rectFace.y + 15;
			}
			int xDraw = rectFace.x + 60;
			if (xDraw >=frame_width){
				xDraw =rectFace.x - 60;
			}
					
			cv::putText(frame , name, cv::Point(rectFace.x, yDraw),cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 2);
			cv::putText(frame, std::to_string( (int) (recognition_confidence*100)), cv::Point(xDraw, yDraw), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,255,0),2);
		}
		cv::imshow("on air",frame);
		if(!pathOutputVideo.empty()){
			video.write(frame);
		} 

		char c = (char) cv::waitKey(25);
		if(c == 27 || c == 32){
			break;
		}
		std::cout << "FPS = " << std::fixed << std::setprecision(1) << 1000/(endrun - startrun) << std::endl;
		std::cout << "Name is " << name << " "<< "Probability is " << recognition_confidence <<std::endl;
	}
	cap.release();
	if(!pathOutputVideo.empty()){
		video.release();
	} 
	cv::destroyAllWindows();	

	return 0;
}

