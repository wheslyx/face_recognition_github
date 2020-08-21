# Face Recognition in C++

## Prerequisites

* CUDA
* libtorch 1.4.0; 1.5.0
* opencv 4.3.0 with DNN module on CUDA


## Running the code

### Construction of the face recognizer FOR TRAINING

Pass as arguments :

* faceDetector : An object of class Face Detector
* embedderModel : A path to the embedder model

Then add an "only eyes pytorch model" to use it for face alignment

Sample code :

```C++
string prototxt = ".../deploy.prototxt.txt";
string caffeModel = ".../res10_300x300_ssd_iter_140000.caffemodel";
FaceDetector faceDetector(prototxt, caffeModel);

string pathModelEyes = ".../MobileNetv2_Adam_01_Eyes_Norm_Without_Flip_v2_70epochs.pt";
Model modelEyes(pathModelEyes, 256 , false , {0.485, 0.456, 0.406} ,{0.229, 0.224, 0.225});
	
string embedderModel = ".../nn4.small2.v1.t7";

FaceRecognition recognizer(faceDetector, embedderModel);
recognizer.addEyesModel(modelEyes, {20, 39}, {0, 19});
```

### Train the face recognizer

Pass as arguments :
* trainPath : A path that should contain directories with the name of each person and each of them should contain at least 30 images
* savePath : A path to save the model Support Vector Machine model(.xml) and the names of the people (.txt)

Sample code :
```C++
string trainPath = "...";
string savePath = ".../svmModel"; //it can be another name
recognizer.train(trainPath, savePath);
```

### Construction of the face recognizer FOR PREDICTION

Pass as arguments :
* faceDetector : An object of class Face Detector
* embedderModel : A path to the embedder model 
* svmModel : A path containing the model of the Support Vector Machine (.xml)
* labelsTxt : A path containing the names of the people with whom the model trained (.txt)

An then add the model of the eyes.

Sample Code:
```C++
string svmModel = ".../svmModel/recognizer.xml";
string labelsTxt = ".../svmModel/labels.txt";

FaceRecognition recognizer = FaceRecognition(faceDetector, embedderModel , svmModel, labelsTxt);
recognizer.addEyesModel(modelEyes, {20, 39}, {0, 19});
```

### Prediction

Use the "recognize" method and 

Pass as arguments:
* img : An opencv Mat that represents the image
* rectFace : An opencv Rect (it doesn't need to contain any information)

The function will  return a string representing the name of the person recognized or
"no face found" if no face was found. It will also replace the value of the parameter "rectFace" with the
rectangle of the face found.

Sample Code:
```C++
cv::Mat img = cv::imread(path);
cv::Rect rectFace;
string predicted = recognize(img, rectFace);
```

### New

OVO Policy: all versus all policy to calculate face prediction confidence.
The metric is the confidence between 0 and 100%

### Test performance

Pass as arguments:
* pathDir : A path that should contain directories with the name of each person and each of them should contain images of the person
* savePath (optional) : A path to create a csv file and save the accuracy of each person

Sample Code:
```C++
string testFolder = "...";
string savePath = "...";
recognizer.testPerformance(testFolder,  savePath);
```

### Test video

Similar to runVideo() of facial landmarks.
Use the runVideo method. If you provide no arguments, it will use the camera to capture the video. 
Or you can provide an input path video to read. Also you can provide an output path video to put the results.

Sample code :
```C++
recognizer.runVideo();
```

### Details of model

For training, we use One vs One approach for each pair of classes with svm class for two classes (Opencv library).
For recognize, we use a custom approach allow us to get a percentage confidence and it is calculated with next pseudo-code:
1. For each pair of classes accumualte the frequency and distance of the win class.
2. If the max frequency is smaller than amount of classes - 1, don't predict
3. Otherwise return the class whit greater frequency and greater distance in tie case
4. Confidence return = (distance of win class) / (sum distances with frequency> = classes - 1)

## Authors

* **[Diego Hurtado](https://github.com/DiegoHDMGZ "Diego Hurtado")** *diego.hurtado@services.mss.pe* *Initial work*
* **[Hans Acha](https://github.com/DiegoHDMGZ "Hans Acha")** *hans.acha@services.mss.pe* *Initial work*
* **[Cesar Segura](https://github.com/wheslyx "Cesar Segura")** *cesar.segura@services.mss.pe*
