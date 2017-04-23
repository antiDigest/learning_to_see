#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "dirent.h"

using namespace std;
using namespace cv;


/* 
   The cascade classifiers that come with opencv are kept in the
   following folder: bulid/etc/haarscascades
   Set OPENCV_ROOT to the location of opencv in your system
*/
// string OPENCV_ROOT = "C:/opencv/";
string cascades = "cascades/";
string FACES_CASCADE_NAME = cascades + "haarcascade_frontalface_alt.xml";
string EYES_CASCADE_NAME = cascades + "haarcascade_eye.xml";
string LEFT_EYE = cascades + "haarcascade_mcs_lefteye.xml";
string RIGHT_EYE = cascades + "haarcascade_mcs_righteye.xml";
string text = "BLINK!";


void drawEllipse(Mat frame, const Rect rect, int r, int g, int b) {
  int width2 = rect.width/2;
     int height2 = rect.height/2;
     Point center(rect.x + width2, rect.y + height2);
     ellipse(frame, center, Size(width2, height2), 0, 0, 360, 
         Scalar(r, g, b), 2, 8, 0 );
}

void drawRect(Mat frame, const Rect rect, int r, int g, int b) {
  int width2 = rect.width/2;
     int height2 = rect.height/2;
     Point center(rect.x + width2, rect.y + height2);
     rectangle(frame, Point (rect.x, rect.y), Point (rect.x+rect.width, rect.y+rect.height), 
         Scalar(r, g, b), 2, 8, 0 );
}


int detectRightEye(Mat frame, Point location, Mat ROI, CascadeClassifier cascade) {
  // frame,ctr are only used for drawing the detected eyes
    vector<Rect> eyes;
    cascade.detectMultiScale(ROI, eyes, 1.1, 8, CV_HAAR_DO_CANNY_PRUNING, Size(20, 20));

    int neyes = (int)eyes.size();
    for( int i = 0; i < neyes ; i++ ) {
      Rect eyes_i = eyes[i];
      drawRect(frame, eyes_i + location, 255, 255, 0);
    }
    return neyes;
}

int detectLeftEye(Mat frame, Point location, Mat ROI, CascadeClassifier cascade) {
  // frame,ctr are only used for drawing the detected eyes
    vector<Rect> eyes;
    cascade.detectMultiScale(ROI, eyes, 1.1, 8, CV_HAAR_DO_CANNY_PRUNING, Size(20, 20));

    int neyes = (int)eyes.size();
    for( int i = 0; i < neyes ; i++ ) {
      Rect eyes_i = eyes[i];
      drawRect(frame, eyes_i + location, 255, 255, 0);
    }
    return neyes;
}

bool detectWink(Mat frame, Point location, Mat ROI, CascadeClassifier cascade) {
  // frame,ctr are only used for drawing the detected eyes
    vector<Rect> eyes;
    cascade.detectMultiScale(ROI, eyes, 1.09, 8, CV_HAAR_DO_CANNY_PRUNING, Size(20, 20));

    int neyes = (int)eyes.size();
    for( int i = 0; i < neyes ; i++ ) {
      Rect eyes_i = eyes[i];

      drawRect(frame, eyes_i + location, 255, 255, 0);
    }
    return(neyes == 1);
}

// you need to rewrite this function
int detect(Mat frame, 
       CascadeClassifier cascade_face, CascadeClassifier cascade_eyes,
       CascadeClassifier cascade_left_eye, CascadeClassifier cascade_right_eye) {
  Mat frame_gray;
  vector<Rect> faces;

  cvtColor(frame, frame_gray, CV_BGR2GRAY);

//  equalizeHist(frame_gray, frame_gray); // input, outuput
//  medianBlur(frame_gray, frame_gray, 5); // input, output, neighborhood_size
//  blur(frame_gray, frame_gray, Size(5,5), Point(-1,-1));
/*  input,output,neighborood_size,center_location (neg means - true center) */


  cascade_face.detectMultiScale(frame_gray, faces, 
               1.09, 3, CV_HAAR_DO_CANNY_PRUNING, Size(30, 30));

  /* frame_gray - the input image
     faces - the output detections.
     1.1 - scale factor for increasing/decreasing image or pattern resolution
     3 - minNeighbors. 
         larger (4) would be more selective in determining detection
     smaller (2,1) less selective in determining detection
     0 - return all detections.
     0|CV_HAAR_SCALE_IMAGE - flags. This flag means scale image to match pattern
     Size(30, 30)) - size in pixels of smallest allowed detection
  */
  
  int detected = 0;

  int nfaces = (int)faces.size();
  for( int i = 0; i < nfaces ; i++ ) {
    Rect face = faces[i];
    Rect rect = Rect(face.x, face.y+face.height/5, face.width, 2*(face.height/5));
    Rect rect1 = Rect(face.x+(face.width/2), face.y+face.height/5, face.width/2, 2*(face.height/5));
    Rect rect2 = Rect(face.x, face.y+face.height/5, face.width/2, 2*(face.height/5));
    // cout << face << endl;
    // cout << rect << endl;

    drawRect(frame, face, 255, 0, 255);
    Mat faceROI = frame_gray(rect);
    Mat faceROI1 = frame_gray(rect1);
    Mat faceROI2 = frame_gray(rect2);
    // int righteyes = detectRightEye(frame, Point(face.x+(face.width/2), face.y+face.height/5), faceROI1, cascade_right_eye);
    // int lefteyes = detectLeftEye(frame, Point(face.x, face.y+face.height/5), faceROI2, cascade_left_eye);
    bool wink = detectWink(frame, Point(face.x, face.y+face.height/5), faceROI, cascade_eyes);
    if(wink) {
      drawRect(frame, rect, 0, 255, 0);
      // cout << "Blink Detection1: " << wink << endl;
      // cout << "Blink Detection2: " << (righteyes+lefteyes==1) << endl;
      putText(frame, text, Point(rect.x, rect.y), CV_FONT_HERSHEY_PLAIN, 1.0, cvScalar(0,255,0), 1, CV_AA);
      detected++;
    }
  }
  return(detected);
}

int runonFolder(const CascadeClassifier cascade1, 
        const CascadeClassifier cascade2,
        const CascadeClassifier cascade3,
        const CascadeClassifier cascade4,
        string folder) {
  if(folder.at(folder.length()-1) != '/') folder += '/';
  DIR *dir = opendir(folder.c_str());
  if(dir == NULL) {
      cerr << "Can't open folder " << folder << endl;
      exit(1);
    }
  bool finish = false;
  string windowName;
  struct dirent *entry;
  int detections = 0;
  while (!finish && (entry = readdir(dir)) != NULL) {
    char *name = entry->d_name;
    string dname = folder + name;
    Mat img = imread(dname.c_str(), CV_LOAD_IMAGE_UNCHANGED);
    if(!img.empty()) {
      int d = detect(img, cascade1, cascade2, cascade3, cascade4);
      cerr << d << " detections" << endl;
      detections += d;
      if(!windowName.empty()) destroyWindow(windowName);
      windowName = name;
      namedWindow(windowName.c_str(),CV_WINDOW_AUTOSIZE);
      imshow(windowName.c_str(), img);
      int key = cvWaitKey(0); // Wait for a keystroke
      switch(key) {
      case 27 : // <Esc>
    finish = true; break;
      default :
    break;
      }
    } // if image is available
  }
  closedir(dir);
  return(detections);
}

void runonVideo(const CascadeClassifier cascade1,
        const CascadeClassifier cascade2,
        const CascadeClassifier cascade3,
        const CascadeClassifier cascade4) {
  VideoCapture videocapture(0);
  if(!videocapture.isOpened()) {
    cerr <<  "Can't open default video camera" << endl ;
    exit(1);
  }
  string windowName = "Live Video";
  namedWindow("video", CV_WINDOW_AUTOSIZE);
  Mat frame;
  bool finish = false;
  while(!finish) {
    if(!videocapture.read(frame)) {
      cout <<  "Can't capture frame" << endl ;
      break;
    }
    detect(frame, cascade1, cascade2, cascade3, cascade4);
    imshow("video", frame);
    if(cvWaitKey(30) >= 0) finish = true;
  }
}

int main(int argc, char** argv) {
  if(argc != 1 && argc != 2) {
    cerr << argv[0] << ": "
     << "got " << argc-1 
     << " arguments. Expecting 0 or 1 : [image-folder]" 
     << endl;
    return(-1);
  }

  string foldername = (argc == 1) ? "" : argv[1];
  CascadeClassifier faces_cascade, eyes_cascade, left_eye_cascade, right_eye_cascade;
  
  if( 
     !faces_cascade.load(FACES_CASCADE_NAME) 
     || !eyes_cascade.load(EYES_CASCADE_NAME)
     || !left_eye_cascade.load(LEFT_EYE)
     || !right_eye_cascade.load(RIGHT_EYE)) {
    cerr << FACES_CASCADE_NAME << " or " << EYES_CASCADE_NAME << " or " << LEFT_EYE << " or " << RIGHT_EYE
     << " are not in a proper cascade format" << endl;
    return(-1);
  }

  int detections = 0;
  if(argc == 2) {
    detections = runonFolder(faces_cascade, eyes_cascade, left_eye_cascade, right_eye_cascade, foldername);
    cout << "Total of " << detections << " detections" << endl;
  }
  else runonVideo(faces_cascade, eyes_cascade, left_eye_cascade, right_eye_cascade);

  return(0);
}