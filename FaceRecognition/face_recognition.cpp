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

#include "face_recognition.hpp"
		
		/*!
		@brief give the current time in milliseconds.
		
		@param none

		@return time in milliseconds
		*/
		long double FaceRecognition::getTime() {

			return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		}
		
		/*!
		@brief give the directory of the path.
		
		@details ***********************************************************************************************
		
		For the training,it is expected that all the images are in a proper directory with the 
		name of the person. This method will search for the name of the directory.
		For example /trainingDataset/Diego/001.jpg should output Diego
			
		@param path the absolute path

		@return the name of the directory
		*/
		
		const std::string FaceRecognition::getName(const std::string &path) {

			int i = (int)path.size() - 1;
			while(i >= 0 && (path[i] != '/' && path[i] != '\\' )){
				i--;
			}
			std::string name = "";
			i--;
			while(i >= 0 && (path[i] != '/' && path[i] != '\\' )){
				name += path[i];
				i--;
			}
			std::reverse(name.begin(), name.end());
			return name;
		}
		
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
		void FaceRecognition::resizeRatio(cv::Mat &img, int width){
			int h = img.rows;
			int w = img.cols;
			double r = 1.0 * width / w;
			
			resize(img , img, cv::Size(width , (int) h * r));
		}
		
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
		void FaceRecognition::faceAlignment(cv::Mat img, cv::Rect faceRect, cv::Mat &face){
			face = img(faceRect);
			std::vector<cv::Point2f> landmarks = modelEyes->getLandmarks(face);
			for(int i = 0; i < (int) landmarks.size(); i++){
				landmarks[i].x += faceRect.x;
				landmarks[i].y += faceRect.y;
			}
			cv::Point2f leftEyeCenter(0,0);
			for(int i = rangeLeftEye.first; i <= rangeLeftEye.second; i++){
				leftEyeCenter.x += landmarks[i].x;
				leftEyeCenter.y += landmarks[i].y;
			}
			leftEyeCenter.x /= rangeLeftEye.second - rangeLeftEye.first + 1;
			leftEyeCenter.y /= rangeLeftEye.second - rangeLeftEye.first + 1;
			
			cv::Point2f rightEyeCenter(0,0);
			for(int i = rangeRightEye.first; i <= rangeRightEye.second; i++){
				rightEyeCenter.x += landmarks[i].x;
				rightEyeCenter.y += landmarks[i].y;
			}
			rightEyeCenter.x /= rangeRightEye.second - rangeRightEye.first + 1;
			rightEyeCenter.y /= rangeRightEye.second - rangeRightEye.first + 1;
			
			cv::Point2f eyesCenter = (leftEyeCenter + rightEyeCenter) / 2;
			
			cv::Point2f d = rightEyeCenter - leftEyeCenter;
			double len = sqrt(d.x * d.x + d.y * d.y);
			double angle = atan2(d.y , d.x) * 180.0 / CV_PI;
			
			const double DESIRED_LEFT_EYE_X = 0.28; 
			const double DESIRED_LEFT_EYE_Y = 0.28;
			
			const double DESIRED_RIGHT_EYE_X = (1.0 - DESIRED_LEFT_EYE_X);
			
			const int DESIRED_FACE_WIDTH = 256;
			const int DESIRED_FACE_HEIGHT = 256;
			double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * DESIRED_FACE_WIDTH;
			double scale = desiredLen / len;
			cv::Mat rot_mat = cv::getRotationMatrix2D(eyesCenter, angle, scale);
			
			rot_mat.at<double>(0, 2) += DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
            rot_mat.at<double>(1, 2) += DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y;
			
			cv::Mat warped = cv::Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, CV_8U, cv::Scalar(128)); // Clear the output image to a default grey.
			
			warpAffine(img, face, rot_mat, warped.size());
			//Util::showImage(face);
		}	
		
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
		void FaceRecognition::equalizeLeftAndRight(cv::Mat &face){
			cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);
			int w = face.cols;
			int h = face.rows;
			
			// 1) First, equalize the whole face.
			cv::Mat wholeFace;
			cv::equalizeHist(face, wholeFace);
			
			// 2) Equalize the left half and the right half of the face separately.
			int midX = w / 2;
			cv::Mat leftSide = face(cv::Rect(0,0, midX,h));
			cv::Mat rightSide = face(cv::Rect(midX,0, w - midX,h));
			cv::equalizeHist(leftSide, leftSide);
			cv::equalizeHist(rightSide, rightSide);
			
			// 3) Combine the left half and right half and whole face together, so that it has a smooth transition.
			
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					int val;
					if (x < w / 4) {          // Left 25%: just use the left face.
						val = leftSide.at<uchar>(y , x);
					}
					else if (x < w * 2 / 4) {   // Mid-left 25%: blend the left face & whole face.
						int lv = leftSide.at<uchar>(y , x);
						int wv = wholeFace.at<uchar>(y , x);
						// Blend more of the whole face as it moves further right along the face.
						float f = (x - w / 4) / (w * 0.25f);
						val = round((1.0f - f) * lv + f * wv);
					}
					else if (x < w * 3 / 4) {   // Mid-right 25%: blend the right face & whole face.
						int rv = rightSide.at<uchar>(y , x - midX);
						int wv = wholeFace.at<uchar>(y , x);
						// Blend more of the right-side face as it moves further right along the face.
						float f = (x - w * 2 / 4) / (w * 0.25f);
						val = round((1.0f - f) * wv + f * rv);
					}
					else {                  // Right 25%: just use the right face.
						val = rightSide.at<uchar>(y , x - midX);
					}
					face.at<uchar>(y , x) = val;
				}
			}
			
			// Use the "Bilateral Filter" to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face.
			cv::Mat filtered = cv::Mat(face.size(), CV_8U);
            cv::bilateralFilter(face, filtered, 0, 20.0, 2.0);
            
            cv::cvtColor(filtered, face, cv::COLOR_GRAY2BGR);
		}
		
		
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
		void FaceRecognition::lightnessEqualization(cv::Mat &face){
			cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);
			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2);
			clahe->apply(face , face);
			cv::Mat filtered = cv::Mat(face.size(), CV_8U);
			cv::bilateralFilter(face, filtered, 0, 20.0, 2.0);
			cv::cvtColor(filtered, face, cv::COLOR_GRAY2BGR);
		}
		
		/*!
		@brief preprocess the img and return a face preprocessed.
		
		@details ***********************************************************************************************
		
		Currently it will only use face alignment 
		
		@param[in] img the input image
		@param[out] face the face detected and preprocessed
		@param[out] the rectangle of the face found

		@return true if a found were found and preprocessed; otherwise, false
		*/
		bool FaceRecognition::preprocessedFace(cv::Mat img, cv::Mat &face,cv::Rect &rectFace){
			
			if(rectFace.empty()) {

				return false;

			}

			if(modelEyes != nullptr){
			
				faceAlignment(img , rectFace, face);
			
			} else {

				face = img(rectFace);
			
			}
			//equalizeLeftAndRight(face);
			//lightnessEqualization(face);
			//Util::showImage(face);
			return true;
		}
				
		/*!
		@brief Constructor for training
		
		@details ***********************************************************************************************
		
		Initialize the face detector , face embedder and support vector machine.
		
		@param faceDetector the face detector
		@param embedderModel a path for the embedder model (.t7) 

		*/
		FaceRecognition::FaceRecognition(std::string embedderModel){

			faceEmbedder = cv::dnn::readNetFromTorch(embedderModel);
			faceEmbedder.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			faceEmbedder.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
			svm = cv::ml::SVM::create();
			svm->setType(cv::ml::SVM::C_SVC);
			svm->setKernel(cv::ml::SVM::LINEAR );
			svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
			svm->setC(1);
			}
		
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
		FaceRecognition::FaceRecognition(std::string embedderModel, std::string svmModel, std::string nameTxt){
			
			faceEmbedder = cv::dnn::readNetFromTorch(embedderModel);
			faceEmbedder.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			faceEmbedder.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); 
			std::ifstream input(nameTxt);
			std::vector<std::pair<int,std::string>> dataPerson;
			int idPerson;
			std::string label;
			while(input >> idPerson >> label){
				dataPerson.push_back(make_pair(idPerson,label));
			}
			input.close();

			std::sort(dataPerson.begin(),dataPerson.end());
			labels.resize((int) dataPerson.size());
			for(int i = 0; i < (int) dataPerson.size(); i++){
				labels[i] = dataPerson[i].second;
			}

			std::cout << "loading svm model from " << svmModel <<  std::endl; 
			int cnt = (int)labels.size();
			svmTrained.resize(cnt);
			for(int i = 0; i < cnt; i++){
				svmTrained[i].resize(cnt);
				for(int j = i + 1; j < cnt; j++){
					svmTrained[i][j] = cv::ml::SVM::load(svmModel + "/" + std::to_string(i) + "_" + std::to_string(j) + ".xml");
				}
			}
			std::cout << "svm model loaded" << std::endl;
		}

		/*!
		@brief Face Recignition destructor
		
		@details ***********************************************************************************************
		
		Destroys the face detector , face embedder and support vector machine.
		
		
		*/
		FaceRecognition::~FaceRecognition(){};
		
		/*!
		@brief Add a model trained only with eyes
		
		@param model an object of class Model
		@param rangeLeftEye a pair representing the range (inclusive) of the landmarks that belongs to the left eye
		@param rangeRightEye a pair representing the range (inclusive) of the landmarks that belongs to the right eye
		
		@return none 
		*/
		void FaceRecognition::addEyesModel(Model model, std::pair<int,int> rangeLeftEye, std::pair<int,int> rangeRightEye){
			this->rangeLeftEye = rangeLeftEye;
			this->rangeRightEye = rangeRightEye;
			this->modelEyes = std::make_shared<Model>(model);
		}
		
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
		std::pair<std::string,float> FaceRecognition::recognize(cv::Mat &img, cv::Rect &rectFace){
			cv::Mat face;
			int minFrequency = (int) nameTxt.size() - 2;
			std::cout << "min frequency = " <<  minFrequency << std::endl;
			if(!preprocessedFace(img , face , rectFace)){
				std::cout << "0" << std::endl;
				std::string fail = "no face detected";
				float failConfidence = 0.0;
				return make_pair(fail,failConfidence);
			}
			std::cout << "1" << std::endl;
			cv::Mat blob =  cv::dnn::blobFromImage(face, 1.0/255.0, cv::Size(96, 96) , cv::Scalar(0, 0, 0));
			faceEmbedder.setInput(blob);
			cv::Mat embeddings = faceEmbedder.forward();
			std::cout << "2" << std::endl;
			int cnt = (int)labels.size();
			std::vector<std::tuple<int,double,int>> score(cnt);
			for(int i = 0; i < cnt; i++){
				score[i] = std::make_tuple(0,0.0,0);
			}
			for(int i = 0; i < cnt; i++){
				for(int j = i + 1; j < cnt; j++){
					cv::Mat output;
					svmTrained[i][j]->predict(embeddings,output,cv::ml::StatModel::RAW_OUTPUT);
					float dist = output.at<float>(0, 0);
					if(dist < 0.0){
						std::get<0>(score[j]) += 1;
						std::get<1>(score[j]) += fabs(dist);
						std::get<2>(score[j]) = j;
					}else{
						std::get<0>(score[i]) += 1;
						std::get<1>(score[i]) += fabs(dist);
						std::get<2>(score[i]) = i;
					}
				}
			}
			std::sort(score.rbegin(),score.rend());

			int frecPredict = std::get<0>(score[0]);
			double distPredict = std::get<1>(score[0]);
			int idPredcit = std::get<2>(score[0]);
			double sumDist = 0.0;
			if(frecPredict < minFrequency){
				std::string fail = "no face detected";
				float failConfidence = 0.0;
				return std::make_pair(fail,failConfidence);
			}

			for(int i = 0; i < cnt; i++){
				if(std::get<0>(score[i]) < minFrequency){
					break;
				}
				sumDist += (double)std::get<1>(score[i]);
			}

			float proba = (float) (distPredict/sumDist);
			return std::make_pair(labels[idPredcit], proba);
		}
		
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
		void FaceRecognition::testPerformance(std::string pathDir, std::string savePath){
			std::vector<cv::String> filenames; 
			
			cv::glob(pathDir, filenames, true); 
			
			std::cout << "Processing images from " << pathDir << "..." << std::endl;
			
			int good = 0;
			int total = 0;
			std::map<std::string , std::pair<int,int>> acPerson;
			
			double totalTime = 0;
			int cnt = 0;
			for(int i = 0; i < (int)filenames.size(); i++) {
				cv::Mat img = cv::imread(filenames[i]);
				
				if(!img.data){
					std::cout << "Problem loading image " << filenames[i] << std::endl;		
				} else {
					double startTime = getTime();
					cv::Rect rectFace = cv::Rect(20,20,80,80);
					std::pair<std::string,float> prediction = recognize(img, rectFace);
					std::string predicted = prediction.first;
					float proba = prediction.second;
					if(predicted == "no face detected" || proba < 0.5){
						continue;
					}
					totalTime += getTime() - startTime;
					cnt++;
					const std::string name = getName(filenames[i]);
					if(predicted == name){
						good++;
						acPerson[name].first++;
					}
					acPerson[name].second++;
					total++;
				}
				
				
				int step = (int) filenames.size() / 10;
				if(i % step == step - 1){
					int percent = (int) round(100.0 * (i + 1) / filenames.size());
					std::cout << percent  << "%  completed" << std::endl;
				}
			}
			std::cout << "Testing ended " << std::endl;
			std::cout << "Avg time per image = " << totalTime / (1000.0 * cnt) << " seg / img" << std::endl; 
			if(!savePath.empty()){
				std::fstream csv; 
				csv.open(savePath + "accuracy.csv", std::ios::out | std::ios::app); 
				csv << "name, correct, total, accuracy\n";
				for(std::pair<std::string , std::pair<int,int>> data : acPerson){
					std::pair<int,int> stats = data.second;
					csv << data.first << ", " << stats.first << ", " << stats.second << ", " << 100.0 * stats.first / stats.second << "%\n";
				}
				csv << "\n";
				csv << "total, " << good << ", " << total << ", " <<  100.0 * good / total << "\n";
				csv.close();
				std::cout << "accuracy saved in " << savePath + "accuracy.csv" << std::endl;
			}
			
			if(total != 0) {
				std::cout << "accuracy total = " << std::fixed << std::setprecision(3) << 100.0 * good / total << " %" << std::endl;
			}
		}

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
		void FaceRecognition::OVOPolicy(int &cntPeople, std::vector<std::vector<cv::Mat>> &embeddings, std::vector<std::vector<int>> &label, std::string &savePath){
			for(int l = 0; l < cntPeople; l++){
				for(int r = l + 1; r < cntPeople; r++){
					int sz1 = (int)label[l].size();
					int sz2 = (int)label[r].size();
					cv::Mat trainMat(sz1 + sz2, 128, CV_32F);
					cv::Mat labelMat(sz1 + sz2, 1, CV_32S );
					for(int i = 0; i < sz1; i++){
						for(int j = 0; j < 128; j++){
							trainMat.at<float>(i , j) = embeddings[l][i].at<float>(0, j);
						}
						labelMat.at<int>(i , 0) = (int)label[l][i];
					}
					for(int i = 0; i < sz2; i++){
						for(int j = 0; j < 128; j++){
							trainMat.at<float>(i + sz1 , j) = embeddings[r][i].at<float>(0, j);
						}
						labelMat.at<int>(i + sz1 , 0) = (int)label[r][i];
					}
					svm->train(trainMat , cv::ml::SampleTypes::ROW_SAMPLE , labelMat);
					svm->save(savePath + "/" + std::to_string(l) + "_" + std::to_string(r) + ".xml");
				}
			}
		}
		
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
		void FaceRecognition::train(std::string trainPath, std::string savePath){
			std::vector<std::vector<cv::Mat>> embeddings;
			std::vector<std::vector<int>> label;
			
			std::map<std::string , int> idLabel;
			std::map<int, std::string> nameLabel; 
			
			std::vector<cv::String> filenames;
			cv::glob(trainPath, filenames, true);
			std::cout << "Starting processing " << filenames.size() << " files" << std::endl;

			int cnt = 0;
			for(int i = 0; i < (int) filenames.size(); i++) {
				const std::string name = getName(filenames[i]);
				if(name.empty()){
					std::cout << "Format error on file " << filenames[i] << std::endl;
					continue;
				}
				if(idLabel.find(name) == idLabel.end()){
					nameLabel[cnt] = name;
					idLabel[name] = cnt;
					cnt++;
					embeddings.resize(cnt);
					label.resize(cnt);
				}
				cv::Mat img = cv::imread(filenames[i]);
				if(!img.data){
					std::cout << "Problem loading image " << filenames[i] << std::endl;	
					continue;
				}
				//resizeRatio(img , 600);
				cv::Mat face;
				cv::Rect rectFace;
				if(!preprocessedFace(img , face, rectFace)){
					continue;
				}
				
				cv::Mat blob =  cv::dnn::blobFromImage(face, 1.0/255.0, cv::Size(96, 96) , cv::Scalar(0, 0, 0));
				faceEmbedder.setInput(blob);

				cv::Mat vec = faceEmbedder.forward();
				embeddings[idLabel[name]].push_back(vec.clone());
				label[idLabel[name]].push_back(idLabel[name]);

				int step = (int) round(filenames.size() / 10.0);
				if(i % step == step - 1){
					int percent = (int) round(100.0 * (i + 1) / filenames.size());
					std::cout << percent  << "%  completed" << std::endl;
				}
			} 
			
			
			std::cout << "training SVM..." << std::endl;
			OVOPolicy(cnt, embeddings, label, savePath);
			std::cout << "training ended and saved in " + savePath << std::endl;
			
			std::string outputTxt = savePath + "/labels.txt";
			std::ofstream output(outputTxt);
			for(std::pair<int , std::string> match : nameLabel){
				output << match.first << " " << match.second<< std::endl;
			}
			output.close();
			std::cout << "names of labels save in " << outputTxt << std::endl ;
		}
		
		
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
		
		void FaceRecognition::runVideo(std::string pathInputVideo, std::string pathOutputVideo){
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
				cv::Rect rectFace = cv::Rect(20,20,80,80);
				long double startrun = getTime();
				std::pair<std::string,float> prediction = recognize(frame, rectFace);
				long double endrun = getTime();
				std::string name = prediction.first;
				float proba = prediction.second;
				if(!(name == "no face detected" || proba < 0.5)){
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
					cv::putText(frame, std::to_string( (int) (proba*100)), cv::Point(xDraw, yDraw), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,255,0),2);
				}
				
				if(!pathOutputVideo.empty()){
					video.write(frame);
				} 

				char c = (char) cv::waitKey(25);
				if(c == 27 || c == 32){
					break;
				}
				std::cout << "FPS = " << std::fixed << std::setprecision(1) << 1000/(endrun - startrun) << std::endl;
			}
			cap.release();
			if(!pathOutputVideo.empty()){
				video.release();
			} 
			cv::destroyAllWindows();
		}

