


#include <android/log.h>
#include <jni.h>


#include <string>
#include <vector>
#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;

inline void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}

class CascadeDetectorAdapter: public DetectionBasedTracker::IDetector
{
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector):
            IDetector(),
            Detector(detector)
    {
        LOGD("CascadeDetectorAdapter::Detect::Detect");
        CV_Assert(detector);
    }

    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
    {
        LOGD("CascadeDetectorAdapter::Detect: begin");
        LOGD("CascadeDetectorAdapter::Detect: scaleFactor=%.2f, minNeighbours=%d, minObjSize=(%dx%d), maxObjSize=(%dx%d)", scaleFactor, minNeighbours, minObjSize.width, minObjSize.height, maxObjSize.width, maxObjSize.height);
        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
        LOGD("CascadeDetectorAdapter::Detect: end");
    }

    virtual ~CascadeDetectorAdapter()
    {
        LOGD("CascadeDetectorAdapter::Detect::~Detect");
    }

private:
    CascadeDetectorAdapter();
    cv::Ptr<cv::CascadeClassifier> Detector;
};

struct DetectorAgregator
{
    cv::Ptr<CascadeDetectorAdapter> mainDetector;
    cv::Ptr<CascadeDetectorAdapter> trackingDetector;

    cv::Ptr<DetectionBasedTracker> tracker;
    DetectorAgregator(cv::Ptr<CascadeDetectorAdapter>& _mainDetector, cv::Ptr<CascadeDetectorAdapter>& _trackingDetector):
            mainDetector(_mainDetector),
            trackingDetector(_trackingDetector)
    {
        CV_Assert(_mainDetector);
        CV_Assert(_trackingDetector);

        DetectionBasedTracker::Parameters DetectorParams;
        tracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
    }
};
extern "C" {
JNIEXPORT jlong JNICALL Java_com_jupiter_facedetection_DetectionTracker_nativeCreateObject
        (JNIEnv * jenv, jclass, jstring jFileName, jint faceSize)
{
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeCreateObject enter");
    const char* jnamestr = jenv->GetStringUTFChars(jFileName, NULL);
    string stdFileName(jnamestr);
    jlong result = 0;

    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeCreateObject");

    try
    {
        cv::Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(
                makePtr<CascadeClassifier>(stdFileName));
        cv::Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(
                makePtr<CascadeClassifier>(stdFileName));
        result = (jlong)new DetectorAgregator(mainDetector, trackingDetector);
        if (faceSize > 0)
        {
            mainDetector->setMinObjectSize(Size(faceSize, faceSize));
            //trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeCreateObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeCreateObject()");
        return 0;
    }

    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeCreateObject exit");
    return result;
}

JNIEXPORT void JNICALL Java_com_jupiter_facedetection_DetectionTracker_nativeDestroyObject
        (JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeDestroyObject");

    try
    {
        if(thiz != 0)
        {
            ((DetectorAgregator*)thiz)->tracker->stop();
            delete (DetectorAgregator*)thiz;
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeestroyObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeDestroyObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeDestroyObject()");
    }
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeDestroyObject exit");
}

JNIEXPORT void JNICALL Java_com_jupiter_facedetection_DetectionTracker_nativeStart
        (JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeStart");

    try
    {
        ((DetectorAgregator*)thiz)->tracker->run();
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStart caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeStart caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStart()");
    }
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeStart exit");
}

JNIEXPORT void JNICALL Java_com_jupiter_facedetection_DetectionTracker_nativeStop
        (JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeStop");

    try
    {
        ((DetectorAgregator*)thiz)->tracker->stop();
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeStop caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStop()");
    }
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeStop exit");
}

JNIEXPORT void JNICALL Java_com_jupiter_facedetection_DetectionTracker_nativeSetFaceSize
        (JNIEnv * jenv, jclass, jlong thiz, jint faceSize)
{
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeSetFaceSize -- BEGIN");

    try
    {
        if (faceSize > 0)
        {
            ((DetectorAgregator*)thiz)->mainDetector->setMinObjectSize(Size(faceSize, faceSize));
            //((DetectorAgregator*)thiz)->trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeSetFaceSize caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeSetFaceSize()");
    }
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeSetFaceSize -- END");
}


JNIEXPORT void JNICALL Java_com_jupiter_facedetection_DetectionTracker_nativeDetect
        (JNIEnv * jenv, jclass, jlong thiz, jlong imageGray, jlong faces)
{
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeDetect");

    try
    {
        vector<Rect> RectFaces;
        ((DetectorAgregator*)thiz)->tracker->process(*((Mat*)imageGray));
        ((DetectorAgregator*)thiz)->tracker->getObjects(RectFaces);
        *((Mat*)faces) = Mat(RectFaces, true);
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeDetect caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code DetectionBasedTracker.nativeDetect()");
    }
    LOGD("Java_com_jupiter_opencv_DetectionBasedTracker_nativeDetect END");
}

};

const int DEFUALT_MARKER = 0;
const int DEFAULT_MARKER_EDGE = 128;
const int DEFAULT_MARKER_WALL = 255;

const int DEV = 0;
const int DEV2 = 0;

VideoCapture g_VideoCapture;
bool g_bVideoCaptureEnable = false;
unsigned int g_nVideoWidth = 0;
unsigned int g_nVideoHeight= 0;

CascadeClassifier cascade, nestedCascade;
//string cascadeName = "e:/Temp/haarcascade_frontalface_alt.xml";
//string nestedCascadeName = "e:/Temp/haarcascade_eye_tree_eyeglasses.xml";
string cascadeName = "/sdcard/tmp/haarcascade_frontalface_alt.xml";
string nestedCascadeName = "/sdcard/tmp/haarcascade_eye_tree_eyeglasses.xml";



/**
 * Function to initialize markers in the image.
 * It takes
 * 		src          input image ( 3-channel RGB or grayscale )
 * Returns the markers ( image of the same size with labels as pixel values ).
 */
Mat initMarkers(Mat src) {
    Mat gray, sobelX, sobelY;
    if(src.type() == CV_8UC3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    Sobel(gray, sobelX, -1, 1, 0);
    Sobel(gray, sobelY, -1, 0, 1);
    Mat sobelNet = abs(sobelX) + abs(sobelY);
    Scalar mean, stdDev;
    meanStdDev(sobelNet, mean, stdDev);
    int s = 5;
    double t = -3*stdDev[0];

    Mat thresh;
    adaptiveThreshold(sobelNet, thresh, DEFAULT_MARKER_EDGE, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, s, t); //EDIT from GAUSSIAN THRESH
    //if(DEV2) imshow("thresh", thresh);
    // waitKey(0);
    Mat dst;
    thresh.convertTo(dst, CV_32S);
    // cout << dst.type() << endl << CV_32SC1 << endl;
    return dst;
}

/**
 * Utility function to extract wall's mask.
 */
void extractMask(Mat src, int pts[][2], int n_pts, Mat markersOrig, Mat & mask) {
    int x, y;
    Mat markers;
    markersOrig.copyTo(markers);
    for(int i=0; i<n_pts; ++i) {
        x = pts[i][0], y = pts[i][1];
        markers.at<int>(y, x) = DEFAULT_MARKER_WALL;
    }

    // cout << 61 << " " << src.type() << " " << CV_8UC3 << endl;
    watershed(src, markers);

    Mat _markers;
    markers.convertTo(_markers, CV_8U);

    //if(DEV) imshow("markers", _markers);

    threshold(_markers, mask, DEFAULT_MARKER_WALL - 1, 255, THRESH_BINARY);
}

/**
 * Utility function to match illuminations in src and target image.
 * Uses L of LAB space as a measure of illumination and makes the L channel same
 * for both the images.
 */
void matchIllumination(Mat src, Mat & target, Mat mask) {

    Mat srcLab, targetLab, targetLabMatched;
    cvtColor(src, srcLab, COLOR_BGR2YUV);
    cvtColor(target, targetLab, COLOR_BGR2YUV);

    vector<Mat> srcLabSplits;
    split(srcLab, srcLabSplits);

    vector<Mat> targetLabSplits;
    split(targetLab, targetLabSplits);

    vector<Mat> targetLabSplitsMatched;
    targetLabSplitsMatched.push_back(srcLabSplits[0]);
    targetLabSplitsMatched.push_back(targetLabSplits[1]);
    targetLabSplitsMatched.push_back(targetLabSplits[2]);

    merge(targetLabSplitsMatched, targetLabMatched);
    cvtColor(targetLabMatched, target, COLOR_YUV2BGR);

}

/**
 * Utility function to fill the wall, given the source image, mask and the wall.
 * Generic base function for both color/pattern based painting.
 */
void wallFill(Mat src, Mat wall, Mat mask, Mat & dst) {
    Mat maskInv = Mat(mask.size(), CV_8UC1, Scalar(255, 255, 255)) - mask;
    Mat m1, m2, m3;

    //if(DEV) imshow("mask", mask);

    matchIllumination(src, wall, mask);

    bitwise_and(src, src, m1, maskInv);
    bitwise_and(wall, wall, m2, mask);

    m3 = m1 + m2;
//bilateralFilter(m3, dst, 3, 0.5, 0.5);

    GaussianBlur(m3,dst,Size(3,3),0.5);
}

/**
 * Function to mark a particular region as wall ( truth ), and extends the
 * wall-group throughout the image. It takes
 * 		src          input image ( 3-channel RGB )
 * 		pts          1D array of points ( mouse-clicks or touches )
 * 		n_pts        length of pts array
 * 		markersOrig  markers ( to be obtained from initMarkers function )
 * 		wall         image depicting a wall
 * Returns the wall-painted image.
 */
Mat wallFillPattern(Mat src, int pts[][2], int n_pts, Mat markersOrig, Mat wall) {
    Mat mask, maskInv, dst;
    extractMask(src, pts, n_pts, markersOrig, mask);

    wallFill(src, wall, mask, dst);
    return dst;
}

/**
 * Function to mark a particular region as wall ( truth ), and extends the
 * wall-group throughout the image. It takes
 * 		src          input image ( 3-channel RGB )
 * 		pts          1D array of points ( mouse-clicks or touches )
 * 		n_pts        length of pts array
 * 		markersOrig  markers ( to be obtained from initMarkers function )
 * 		color        scalar denoting the color
 * Returns the wall-painted image.
 */
Mat wallFillColor(Mat src, int pts[][2], int n_pts, Mat markersOrig, Scalar color) {
    Mat mask, maskInv, dst;
    extractMask(src, pts, n_pts, markersOrig, mask);

    Mat wall = Mat(mask.size(), CV_8UC3, color);
    wallFill(src, wall, mask, dst);
    return dst;
}

/** UTIL FUNCTIONS FOR SAMPLE RUN ON A COMPUTER == START **/
vector<Point> pts;
Point prevPt(-1, -1), INVALID_POINT(-1, -1);
char * out_name;
Mat _img, _markers, _wall;
Scalar _color;
/*
Rescales the image to 1080 width, and centers the image, padding to 1920x1080 in total
satisfying image.cols<width (1080 default) , image.rows<height (1920 default), such that

 */

Mat rescale(Mat src ,int fill_border=0, int width=1080,int height=1920 )
{
    float aspect_ratio = src.cols/((float)(src.rows));
    Scalar fill_color;
    if (fill_border==0)
        fill_color = Scalar(0,0,0); //Black fill_color
    else
        fill_color = Scalar(255,255,255);

    Size dst_Size = Size(1920,1080);
    Mat dst;
    int top,bottom,left,right;
    if (aspect_ratio>16.0/9.0)
    {
        left = 0;
        right = 0;
        top  = (int) (9*src.cols-16*src.rows)/(32.0);
        bottom = top;
    }
    else
    {
        top = 0 ;
        bottom = top;
        left = (int) (16*src.rows-9*src.cols)/(18.0);
        right = left;
    }
    copyMakeBorder(src,src,top,bottom,left,right,BORDER_CONSTANT,fill_color);
    resize(src,dst,dst_Size,0,0,INTER_AREA);
    return dst;
}

void DetectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
            {
                    Scalar(255,0,0),
                    Scalar(255,128,0),
                    Scalar(255,255,0),
                    Scalar(0,255,0),
                    Scalar(0,128,255),
                    Scalar(0,255,255),
                    Scalar(0,0,255),
                    Scalar(255,0,255)
            };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1 / scale;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT );
    //equalizeHist( smallImg, smallImg );
    //return;
    t = (double)getTickCount();
    printf("\nAfter resize()");
    //if (true == cascade.empty()) return;
    cascade.detectMultiScale( smallImg, faces,
                              1.1, 2, 0
                                      //|CASCADE_FIND_BIGGEST_OBJECT
                                      //|CASCADE_DO_ROUGH_SEARCH
                                      |CASCADE_SCALE_IMAGE,
                              Size(30, 30) );
    printf("\nAfter cascade.detectMultiScale");
    return;
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                  1.1, 2, 0
                                          //|CASCADE_FIND_BIGGEST_OBJECT
                                          //|CASCADE_DO_ROUGH_SEARCH
                                          |CASCADE_SCALE_IMAGE,
                                  Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
#if 0
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
#endif
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
                                        1.1, 2, 0
                                                //|CASCADE_FIND_BIGGEST_OBJECT
                                                //|CASCADE_DO_ROUGH_SEARCH
                                                //|CASCADE_DO_CANNY_PRUNING
                                                |CASCADE_SCALE_IMAGE,
                                        Size(30, 30) );
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
    //imshow( "result", img );
}

int DetectObject(Mat& matImage)
{
    double dScale = 1.0;
    bool bTryFlip = false;

    if (!matImage.empty())
    {
        DetectAndDraw(matImage, cascade, nestedCascade, dScale, bTryFlip);
        //waitKey(0);
//        rectangle(matImage, cvPoint(10, 10),
//                  cvPoint(300, 200),
//                  Scalar(255,0,0), 3, 8, 0);
    }

    return true;
}

int DetectObjectInit()
{
    if ( !nestedCascade.load( nestedCascadeName ) )
    {
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
        //AfxMessageBox("WARNING: Could not load classifier cascade for nested objects");
    }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        //AfxMessageBox("ERROR: Could not load classifier cascade");
        //help();
        return -1;
    }
}

extern "C"
{
void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_salt(JNIEnv *env, jobject instance,
                                                                           jlong matAddrGray,jlong optaddr,
                                                                           jint nbrElem) {
    Mat &mGr = *(Mat *) matAddrGray;
    Mat &opt = *(Mat *) optaddr;
#if 0
    cvtColor(mGr,mGr,COLOR_RGBA2RGB);
    Mat mgr;
    resize(mGr,mgr,Size(0,0),0.25,0.25);
    Scalar color(255,0,0);
    int pts_arr1[1][2];
    Mat markersOrig = initMarkers(mgr);
    pts_arr1[0][0] = (int) mgr.rows/2.0;
    pts_arr1[0][1] = (int) mgr.cols/2.0;
    //Mat &opt = *(Mat *) optaddr;
    opt = wallFillColor(mgr,pts_arr1,1,markersOrig,color);
    resize(opt,opt,mGr.size());
    //opt = mGr;
#else
    DetectObjectInit();
    DetectObject(mGr);
    mGr.copyTo(opt);
#endif

}
}