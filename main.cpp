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
	
	FaceRecognition recognizer(faceDetector, FACE_RECOGNITION , RECOGNITION_WEIGHTS, labelsTxt);
	recognizer.addEyesModel(modelEyes, {20, 39}, {0, 19});
	
	//real time
	std::string pathInputVideo = "";
	std::string pathOutputVideo= "/media/disk/video_test/face_recognition.avi";

	recognizer.runVideo(pathInputVideo, pathOutputVideo);

	return 0;
}

