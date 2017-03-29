#include "opencv2/highgui.hpp"
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;

Mat M = (Mat_<double>(3,3) << 3.240479, -1.53715, -0.498535,
        -0.969256, 1.875991, 0.041556,
        0.055648, -0.204043, 1.057311);
Mat M_inv = (Mat_<double>(3,3) << 0.412453, 0.35758, 0.180423,
        0.212671, 0.71516, 0.072169,
        0.019334, 0.119193, 0.950227);

Mat rgbtoluv(double r, double g, double b){

    Mat RGB = (Mat_<double>(3,1) << r, g, b);

    // Converting to Linear RGB
    for(int k=0;k<3;k++){
        if(RGB.at<double>(k) < 0.03928){
            RGB.at<double>(k) = (double) RGB.at<double>(k)/12.92;
        }
        else{
            RGB.at<double>(k) = (double) pow((RGB.at<double>(k)+0.055)/1.055,2.4);
        }
    }
    // Linear RGB to XYZ
    Mat XYZ = M_inv * RGB;

    // Separating X, Y, Z
    double X = XYZ.at<double>(0);
    double Y = XYZ.at<double>(1);
    double Z = XYZ.at<double>(2);

    // Calculating subsidiaries for Luv
    double u_w = (double) (4.0*0.95)/(0.95+15.0+3.0*1.09);
    double v_w = (double) (9.0)/(0.95+15.0+3.0*1.09);

    double d = X + 15.0 * Y + 3.0 * Z;

    double u_prime = (4.0*X)/(d);
    double v_prime = (9.0*Y)/(d);

    double t = Y;

    double L_val;

    // Calculating Final L, u, v
    if(t>0.008856){
        L_val = (double) 116*pow(t, double(1.0/3.0)) - 16.0;
    }
    else{
        L_val = (double) (t*903.3);
    }
    double u_val = 13.0*L_val*(u_prime - u_w);
    double v_val = 13.0*L_val*(v_prime - v_w);

    // Clipping
    // if(u_val<0.0)
    //     u_val=0.0;
    // else if(u_val>255.0)
    //     u_val=255.0;

    // if(v_val<0.0)
    //     v_val=0.0;
    // else if(v_val>255.0)
    //     v_val=255.0;

    Mat Luv = (Mat_<double>(3,1) << L_val,u_val,v_val);

    return Luv;
}

Mat luvtorgb(double L, double u, double v){

    double X, Y, Z;

    // Luv to XYZ
    double u_w = (double) (4.0*0.95)/(0.95+15.0+3.0*1.09);
    double v_w = (double) (9.0)/(0.95+15.0+3.0*1.09);

    double u_prime = (double(u) + 13.0*u_w*L)/(13.0*L);
    double v_prime = (double(v) + 13.0*v_w*L)/(13.0*L);

    if(L>7.9996){
        Y = (double) pow((L + 16.0)/116.0, 3.0);
    }
    else{
        Y = (double) (L/903.3);
    }

    X = (double) Y*2.25*(u_prime/v_prime);
    Z = (double) Y*(3.0-(0.75*u_prime)-(5.0*v_prime))/v_prime;
    if(v_prime==0.0){
        X=0.0; Z=0.0;
    }

    // XYZ to Linear sRGB
    Mat XYZ = (Mat_<double>(3,1) << X,Y,Z);
    Mat srgb = M*XYZ;
    
    // Linear to Non Linear sRGB
    for(int k=0;k<3;k++){
        if(srgb.at<double>(k) < 0.00304){
            srgb.at<double>(k) = 12.92*srgb.at<double>(k);
        }
        else{
            srgb.at<double>(k) = 1.055*pow(srgb.at<double>(k),double(1.0/2.4)) - 0.055;
        }
    }

    //Clipping
    for(int k=0;k<3;k++){
        if(srgb.at<double>(k) < 0.0){
            srgb.at<double>(k) = 0.0;
        }
        else if(srgb.at<double>(k) > 1.0){
            srgb.at<double>(k) = 1.0;
        }
    }

    return srgb;
}

void runOnWindow(int W1,int H1, int W2,int H2, Mat inputImage, char *outName) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    vector<Mat> i_planes;
    split(inputImage, i_planes);
    Mat iB = i_planes[0];
    Mat iG = i_planes[1];
    Mat iR = i_planes[2];

    // dynamically allocate RGB arrays of size rows x cols
    int** R = new int*[rows];
    int** G = new int*[rows];
    int** B = new int*[rows];
    for(int i = 0 ; i < rows ; i++) {
        R[i] = new int[cols];
        G[i] = new int[cols];
        B[i] = new int[cols];
    }

    for(int i = 0 ; i < rows ; i++)
        for(int j = 0 ; j < cols ; j++) {
            R[i][j] = iR.at<uchar>(i,j);
            G[i][j] = iG.at<uchar>(i,j);
            B[i][j] = iB.at<uchar>(i,j);
        }


    //     The transformation should be based on the
    //     historgram of the pixels in the W1,W2,H1,H2 range.
    //     The following code goes over these pixels

    Mat M = (Mat_<double>(3,3) << 3.240479, -1.53715, -0.498535,
            -0.969256, 1.875991, 0.041556,
            0.055648, -0.204043, 1.057311);
    Mat M_inv = (Mat_<double>(3,3) << 0.412453, 0.35758, 0.180423,
            0.212671, 0.71516, 0.072169,
            0.019334, 0.119193, 0.950227);

    double** L = new double*[rows];
    double** u = new double*[rows];
    double** v = new double*[rows];
    for(int i = 0 ; i < rows ; i++) {
        L[i] = new double[cols];
        u[i] = new double[cols];
        v[i] = new double[cols];
    }

    // Histogram values init
    int* histogram = new int[101];
    int* pixels_in_range = new int[101];
    int* exchange = new int[101];
    double* stretched = new double[101];
    
    pixels_in_range[-1] = 0;
    for(int k=0;k<101;k++){
        histogram[k] = 0;
        pixels_in_range[k] = 0;
        exchange[k] = k;
        stretched[k] = k;
    }

    // sRGB to Luv
    for(int i = 0 ; i < rows ; i++) 
        for(int j = 0 ; j < cols ; j++) {
            double r = R[i][j]/double(255.0);
            double g = G[i][j]/double(255.0);
            double b = B[i][j]/double(255.0);

            Mat Luv = rgbtoluv(r, g, b);

            L[i][j] = Luv.at<double>(0);
            u[i][j] = Luv.at<double>(1);
            v[i][j] = Luv.at<double>(2);
        }

    for(int i = H1 ; i <= H2 ; i++) 
        for(int j = W1 ; j <= W2 ; j++)
            histogram[int(round(L[i][j]))]++;
    
    // Histogram Equalization
    for(int k=0;k<101;k++){
        pixels_in_range[k] = pixels_in_range[k-1] + histogram[k];
    }
    int max_pixels = pixels_in_range[100];
    for(int k=0;k<101;k++){
        exchange[k] = round((pixels_in_range[k-1]+pixels_in_range[k])*101/max_pixels);
    }

    // Finding Max of L
    double max_L = 0.0;
    double min_L = 1000000.0;
    for(int i = H1 ; i <= H2 ; i++) 
        for(int j = W1 ; j <= W2 ; j++) {
            double L_val = exchange[int(round(L[i][j]))];
            if(max_L<L_val){
                max_L = L_val;
            }
            if(min_L>L_val){
                min_L = L_val;
            }
        }

    // Histogram values stretching
    for(int k=0;k<101;k++){
        stretched[k] = (double) (((exchange[k]-min_L) * (100.0 - 0.0)) / (max_L - min_L)) + 0.0 ;

     //   cout << "i = " << k << " Histogram = " << histogram[k] << " pixels_in_range = " << pixels_in_range[k]
     //   << " exchange = " << exchange[k] << " Stretched value = " << stretched[k] << endl;

    }

    // Luv to sRGB
    for(int i = 0 ; i < rows ; i++) 
        for(int j = 0 ; j < cols ; j++) {
            
            double u_val = u[i][j];
            double v_val = v[i][j];
            double L_val = L[i][j];

            // Stretched values
            L_val = stretched[int(round(L[i][j]))] ;

            Mat srgb = luvtorgb(L_val, u_val, v_val);

            // if(R[i][j]!=(int) srgb.at<double>(0) * 255){
            //     cout << "RGB Old " << R[i][j] << ", " << G[i][j] << ", " << B[i][j] << endl;
            //     cout << "RGB New " << (int) srgb.at<double>(0) * 255
            //     << ", " << (int) srgb.at<double>(1) * 255
            //     << ", " << (int) srgb.at<double>(2) * 255 << endl;
            // }

            // int values of non-linear RGB stretched to 0-255
            R[i][j] = (int) (srgb.at<double>(0)*255);
            G[i][j] = (int) (srgb.at<double>(1)*255);
            B[i][j] = (int) (srgb.at<double>(2)*255);
        }

    Mat oR(rows, cols, CV_8UC1);
    Mat oG(rows, cols, CV_8UC1);
    Mat oB(rows, cols, CV_8UC1);
    for(int i = 0 ; i < rows ; i++)
        for(int j = 0 ; j < cols ; j++) {
            oR.at<uchar>(i,j) = R[i][j];;
            oG.at<uchar>(i,j) = G[i][j];;
            oB.at<uchar>(i,j) = B[i][j];;
        }

    Mat o_planes[] = {oB, oG, oR};
    Mat outImage;
    merge(o_planes, 3, outImage);

    namedWindow("output", CV_WINDOW_AUTOSIZE);
    // imshow("output", outImage);
    imwrite(outName, outImage);
}

int main(int argc, char** argv) {
    if(argc != 7) {
        cerr << argv[0] << ": "
        << "got " << argc-1 
        << " arguments. Expecting six: w1 h1 w2 h2 ImageIn ImageOut." 
        << endl ;
        cerr << "Example: proj1b 0.2 0.1 0.8 0.5 fruits.jpg out.bmp" << endl;
        return(-1);
    }
    double w1 = atof(argv[1]);
    double h1 = atof(argv[2]);
    double w2 = atof(argv[3]);
    double h2 = atof(argv[4]);
    char *inputName = argv[5];
    char *outputName = argv[6];

    if(w1<0 || h1<0 || w2<=w1 || h2<=h1 || w2>1 || h2>1) {
        cerr << " arguments must satisfy 0 <= w1 < w2 <= 1"
        << " ,  0 <= h1 < h2 <= 1" << endl;
        return(-1);
    }

    Mat inputImage = imread(inputName, CV_LOAD_IMAGE_UNCHANGED);
    if(inputImage.empty()) {
        cout <<  "Could not open or find the image " << inputName << endl;
        return(-1);
    }

    string windowInput("input: ");
    windowInput += inputName;

    namedWindow(windowInput, CV_WINDOW_AUTOSIZE);
    // imshow(windowInput, inputImage);

    if(inputImage.type() != CV_8UC3) {
        cout <<  inputName << " is not a standard color image  " << endl;
        return(-1);
    }

    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int W1 = (int) (w1*(cols-1));
    int H1 = (int) (h1*(rows-1));
    int W2 = (int) (w2*(cols-1));
    int H2 = (int) (h2*(rows-1));

    runOnWindow(W1, H1, W2, H2, inputImage, outputName);

    waitKey(0); // Wait for a keystroke
    return(0);
}
