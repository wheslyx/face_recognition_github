
#ifndef CUDA_VERSION
#define CUDA_VERSION cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice())    
#endif
 
#ifndef VIDEO_STORAGE
#define VIDEO_STORAGE "/media/disk/m3u8/*"
#endif

#ifndef VIDEO_BACKUP
#define VIDEO_BACKUP "/media/disk/backup/"
#endif

#ifndef FACE_NETWORK
#define FACE_NETWORK "/var/fcs_res/deploy.prototxt.txt"
#endif

#ifndef FACE_WEIGHTS
#define FACE_WEIGHTS "/var/fcs_res/res10_300x300_ssd_iter_140000.caffemodel"
#endif

#ifndef FACELANDMARKS
#define FACELANDMARKS "/var/fcs_res/MobileNetv2_Adam_01_Face_Norm_Without_Flip_v2_75epochs.pt"
#endif 

#ifndef EYESLANDMARKS
#define EYESLANDMARKS "/var/fcs_res/MobileNetv2_Adam_01_Eyes_Norm_Without_Flip_v2_70epochs.pt"
#endif

#ifndef FACELANDMARKSFIXED
#define FACELANDMARKSFIXED "/var/fcs_res/MobileNetv2_Adam_106_256x256_Fix_60epochs.pt"
#endif

#ifndef PHONE_DETECTOR
#define PHONE_DETECTOR "/var/fcs_res/resnet18_512_phone_detector.pt"
#endif

#ifndef FACE_RECOGNITION
#define FACE_RECOGNITION "/var/fcs_res/openface_nn4.small2.v1.t7"
#endif

#ifndef RECOGNITION_WEIGHTS
#define RECOGNITION_WEIGHTS "/media/disk/svmModel"
#endif

#ifndef USB_GPS_PORT
#define USB_GPS_PORT "/dev/ttyTHS2"
#endif

#ifndef PATH_VIBRATOR
#define PATH_VIBRATOR "/media/disk/m3u8/FCS_EVENTS.txt"
#endif

#ifndef FACE_THRESHOLD
#define FACE_THRESHOLD 0.5
#endif

#ifndef GPS_SPEED_THRESHOLD
#define GPS_SPEED_THRESHOLD 10 //10 Speed is in kilometer/hour
#endif

#ifndef TRANSLATION_TOLERANCE_X
#define TRANSLATION_TOLERANCE_X 50 //300 
#endif

#ifndef TRANSLATION_TOLERANCE_Y
#define TRANSLATION_TOLERANCE_Y 50 //300
#endif

#ifndef TRANSLATION_TOLERANCE_Z
#define TRANSLATION_TOLERANCE_Z 100 //300
#endif

#ifndef ROTATION_TOLERANCE
#define ROTATION_TOLERANCE 10//175 //15 // 10
#endif

#ifndef TRANSLATION_CALIB_X
#define TRANSLATION_CALIB_X 600 //41
#endif

#ifndef TRANSLATION_CALIB_Y
#define TRANSLATION_CALIB_Y 420 // 6
#endif

#ifndef TRANSLATION_CALIB_Z
#define TRANSLATION_CALIB_Z -1500 //-818
#endif

#ifndef ROTATION_CALIB_X
#define ROTATION_CALIB_X -83
#endif

#ifndef ROTATION_CALIB_Y
#define ROTATION_CALIB_Y -15//475//-22//-20 // 9
#endif

#ifndef ROTATION_CALIB_Z
#define ROTATION_CALIB_Z -2
#endif

#ifndef MAR_THRESHOLD
#define MAR_THRESHOLD 0.5 // Upper value which is considered open mouth
#endif
 
#ifndef MAXIMUM_OPEN_MOUTH
#define MAXIMUM_OPEN_MOUTH 1.5 // Min time in seconds to considen Yawn 
#endif

#ifndef EAR_THRESHOLD
#define EAR_THRESHOLD 0.27 // cantidad minima que se pueden cerrar los ojos para detectar parpadeo
#endif

#ifndef MAXIMUM_CLOSED_EYE
#define MAXIMUM_CLOSED_EYE 3 // En segundos
#endif

#ifndef MAXIMUM_DISTRACTION
#define MAXIMUM_DISTRACTION 1.5 // Min time in seconds to cosider left or right distraction
#endif

#ifndef thres_durationObj
#define thres_durationObj 1.5 // Min time in seconds to cosider the presence of phone
#endif

#ifndef THRESHOLD_NOFACE  // Maximum intermission time to verify it is not face, must be less than threshold for alerts
#define THRESHOLD_NOFACE 0.5
#endif

#ifndef thres_recognition
#define thres_recognition 0.5 // wait this time to do face recognition
#endif 

#ifndef DEVIATION_BOX_X
#define DEVIATION_BOX_X 0.5
#endif

#ifndef DEVIATION_BOX_Y
#define DEVIATION_BOX_Y 0.38
#endif

#ifndef EVENT_MAXLINE 
#define EVENT_MAXLINE 18
#endif

#ifndef PORT_CS // PORT TO COMMUNICATE WITH CONTROL SCREEN 
#define PORT_CS 8031
#endif

#ifndef SIZEFACE
#define SIZEFACE 300 // This is the size of the image processed by the face detector algorithm.
#endif

#ifndef SIZELANDMARKS
#define SIZELANDMARKS 256 // This is the size of the image processed by the facial landmarks algorithm.
#endif

#ifndef SIZEPHONE
#define SIZEPHONE 512 // This is the size of the image processed by the phone detector algorithm.
#endif

#ifndef LANDMARK_NUMBER
#define LANDMARK_NUMBER	106 // This is the number of face landmarks calculated by the facial landmarks algorithm.
#endif

#ifndef FRAME_WIDTH
#define FRAME_WIDTH	640.0 // This is the camera frame width or number of columns. constant type Double
#endif

#ifndef FRAME_HEIGHT
#define FRAME_HEIGHT 480.0 // Camera height or number of rows. constant type Double
#endif

#ifndef HERTZ
#define HERTZ 30 // Camera frame refresh rate in HZ.
#endif

#ifndef WIDTH_OFFSET
#define WIDTH_OFFSET 0.0 // This is offset to the left of right of the camera location. "Yaw offset". It is zero when camera is in front of driver. Constant type double.
#endif 

#ifndef HEIGHT_OFFSET
#define HEIGHT_OFFSET 0.0 // This is offset to the up or down of camera location. "Pitch Camera". It is zero when camera is in front of driver. Constant type double.
#endif
