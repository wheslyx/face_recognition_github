#ifndef FACE_RECOGNITION_H
#define FACE_RECOGNITION_H 

/*!
@file face_recognition.h

@brief C++ functions for face recognition using opencv

@details 
Created on : May 12, 2020
Author : Diego Hurtado de Mendoza 
Author : Hans Martin Acha Carranza
Author : Cesar Segura Del Rio
Sources:
https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
https://github.com/MasteringOpenCV/code/blob/master/Chapter8_FaceRecognition/preprocessFace.cpp
*/

#include <map>
#include <vector>
#include "../FacialLandmarks/model.hpp"

class FaceRecognition {

	private:
		cv::dnn::Net faceEmbedder; //!< face embedder model from opencv
		cv::Ptr<cv::ml::SVM> svm; //!< Support Vector Machine model from opencv
		std::vector<std::vector<cv::Ptr<cv::ml::SVM>>> svmTrained; //!< list of list of Support Vector Machine model for each pair of classes
		std::vector<std::string> labels; //!< list of all the labels (names)
		std::shared_ptr<Model> modelEyes; //!< the model only for eyes landmarks from pytorch
		std::pair<int,int> rangeLeftEye; //!< range of position of the landmarks of the left eye
		std::pair<int,int> rangeRightEye; //!< range of position of the landmarks of the right eye
		std::vector<cv::Rect> rectangles;
		int minFrequency;
		
		
		/*!
		@brief give the directory of the path.
		
		@details ***********************************************************************************************
		
		For the training,it is expected that all the images are in a proper directory with the 
		name of the person. This method will search for the name of the directory.
		For example /trainingDataset/Diego/001.jpg should output Diego
			
		@param path the absolute path

		@return the name of the directory
		*/
		
		const std::string getName(const std::string &path);
		
		
		/*!
		@brief resize the image with the corresponding with, maintaining the ratio with the height.
		
		@details ***********************************************************************************************
		
		It will resize the image with the width provided and it will calculate
		the new height so the ratio is preserved. 
		
		Initially, the metodh was created to improve the accuracy in face recognition 
		but it does not seem to have a significant impact so it is not used in the preprocess of the image.
			
		@param[out] img the input image to be resized
		@param[in] width width of the image

		@return none
		*/
		void resizeRatio(cv::Mat &img, int width);
		
		
		/*!
		@brief Align the face so the eyes are horizontal and the distance between eyes is always the same.
		
		@details ***********************************************************************************************
		
		It will use the facial landmarks of the eye to rotate the image so the eyes are horizontal.
		It will also scale the image so the distance between eyes will be the same for all images
		Finally it will crop the image so only the face is visible.
		
		Those linear transformations are achieve with the function warpAffine.
		
		The most important parameters are DESIRED_LEFT_EYE_X and DESIRED_LEFT_EYE_Y. 
		Results have improved significantly with face alignment with 0.28 as the parameter.
		
		@param[in] img the input image to be aligned (should be bigger than the face itself)
		@param[in] faceRect the rectangle of img containing the face
		@param[out] face the output face aligned (should be empty in the input)

		@return none
		*/
		void faceAlignment(cv::Mat img, cv::Rect faceRect, cv::Mat &face);
				
		
		/*!
		@brief equalize the brightness of the face.
		
		@details ***********************************************************************************************
		
		It will use the function equalizeHist of opencv to get a more homogenous brightness in the face.
		To get better results, the author of the book "Mastering OpenCV" recommend to do an equalization
		to the left and right half of face and a mix equalization for the center part of the face.
		
		This functions has not improve the accuracy of the face recognition in our tests yet.
		Maybe some parameters may be tuned to get better results
		
		@param[out] face the face to be equalized

		@return none
		*/
		
		void equalizeLeftAndRight(cv::Mat &face);
		
		
		/*!
		@brief equalize the brightness of the face.
		
		@details ***********************************************************************************************
		
		It has the same intention of the function above but with another implementation, using CLAHE 
		from Opencv.
		
		This functions has not improve the accuracy of the face recognition in our tests yet.
		Maybe some parameters may be tuned to get better results
		
		@param[out] face the face to be equalized

		@return none
		*/
		void lightnessEqualization(cv::Mat &face);
		
		/*!
		@brief preprocess the img and return a face preprocessed.
		
		@details ***********************************************************************************************
		
		Currently it will only use face alignment 
		
		@param[in] img the input image
		@param[out] face the face detected and preprocessed
		@param[out] the rectangle of the face found

		@return true if a found were found and preprocessed; otherwise, false
		*/
		bool preprocessedFace(cv::Mat img, cv::Mat &face,cv::Rect &rectFace);
		
	public:
		
		/*!
		@brief Constructor for training
		
		@details ***********************************************************************************************
		
		Initialize the face detector , face embedder and support vector machine.
		
		@param faceDetector the face detector
		@param embedderModel a path for the embedder model (.t7)

		*/
		FaceRecognition(std::string embedderModel);
		
		/*!
		@brief Constructor for predicting
		
		@details ***********************************************************************************************
		
		Initialize the face detector , face embedder.
		It will also load the names of the people labeled and all SVM model for each pair of classes
		
		@param faceDetector the face detector
		@param embedderModel a path for the embedder model (.t7)
		@param svmModel a path for the SVM model (folder)
		@param nameTxt a path for the names of the people labeled (.txt)

		*/
		FaceRecognition(std::string embedderModel, std::string svmModel, std::string nameTxt);
		
		/*!
		@brief Face Recignition destructor
		
		@details ***********************************************************************************************
		
		Destroys the face detector , face embedder and support vector machine.
		
		
		*/
		~FaceRecognition();

		/*!
		@brief Add a model trained only with eyes
		
		@param model an object of class Model
		@param rangeLeftEye a pair representing the range (inclusive) of the landmarks that belongs to the left eye
		@param rangeRightEye a pair representing the range (inclusive) of the landmarks that belongs to the right eye
		
		@return none 
		*/
		void addEyesModel(Model model, std::pair<int,int> rangeLeftEye, std::pair<int,int> rangeRightEye);
		
		/*!
		@brief predict the name of the person in the image and return also the rectangle of the face
		
		@details ***********************************************************************************************
		
		First , it found a face and preprocessed it. Then calculates the embeddings for the face.
		Finally it passes the embeddings through the Support Vector Machine with One vs One approach and gets the label.
		If no face is found, it will return "no face found"
		
		@param[in] img the input image to predict
		@param[out] label of recognition and probability
		@param[in] minimum frequency of vote to consider in confidence percentage

		@return a string and confidence of the name predicted 
		*/
		std::pair<std::string,float> recognize(cv::Mat &img, cv::Rect &rectFace);
		
		/*!
		@brief Test the performance of the model with a path containing a group of directory of images for each person
		
		@details ***********************************************************************************************
		
		It will open the directory of each person and it will predict its name based on the image
		
		Finally it will calculate accuracy and the average time consumed for image in seconds
		
		Adittionally, if a path for saving is provided, it will create a csv with the accuracy
		for each person.
	
		@param pathDir the directory of the images
		@param savePath an optional path for saving a csv with the accuracy
		
		@return none
		*/
		void testPerformance(std::string pathDir, std::string savePath);

		/*!
		@brief One vs One approach splits a multi-class classification into one binary classification problem per each pair of classes.
		
		@details ***********************************************************************************************
		
		Each group of 128 embeddings for each pair of classes is calssify by SVM algorithm

		Finally the SVM model is store in the folder savePath in .xml file for each pair
		
		@param cntPeople amount of classes
		@param embeddings list of list of embeddings of all classes
		@param labels list of list of labels of all classes
		@param savePath path where .xml file is save
		
		@return none
		*/
		void OVOPolicy(int &cntPeople, std::vector<std::vector<cv::Mat>> &embeddings, std::vector<std::vector<int>> &label, std::string &savePath);
		
		/*!
		@brief Train the model with a directory containing images for each person to be recognized
		
		@details ***********************************************************************************************
		
		It will open the directory of each person and it will calculate the embeddings of all of them.
		Then it will train the Support Vector Machine with all the embeddings.
		
		Finally it will save the model and the names founded. Also it will show the time elapsed
		
		@param trainPath the directory of the images
		@param savePath the path for saving the model and the names
		
		@return none
		*/
		void train(std::string trainPath, std::string savePath);
		
		
		/*!
		@brief Face recognition for a video
		
		@details ***********************************************************************************************
		If pathInputVideo is empty, the camera will be used; otherwise the video passed in the path will be used
		
		If pathOutputVideo is not empty, it will save the video in the path.
		
		In all the cases, it will also show the video with the predicted label of the person predicted
		
		@param[in] pathInputVideo optional path of the input video
		@param[in] pathOutputVideo optional path ot the output video
		
		@return none
		*/
		
		void runVideo(std::string pathInputVideo, std::string pathOutputVideo);

		/*!
		@brief give the current time in milliseconds.
		
		@param none

		@return time in milliseconds
		*/
		long double getTime();
	
};

#endif
