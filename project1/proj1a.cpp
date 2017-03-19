#include "opencv2/highgui.hpp"
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;

Mat M = (Mat_<double>(3,3) << 3.240479, -1.53715, -0.498535,
    -0.969256, 1.875991, 0.041556,
    0.055648, -0.204043, 1.057311);

Mat xyytorgb(double x, double y, double Y){
    double X = x/y;
    double Z = (1-x-y)/y;

    Mat XYZ = (Mat_<double>(3,1) << X,Y,Z);
    Mat srgb = M*XYZ;
    // cout << "Old "<<  srgb.at<double>(0) << endl;
    for(int k=0;k<3;k++){
        if(srgb.at<double>(k) < 0.00304){
            srgb.at<double>(k) = 12.92*srgb.at<double>(k);
        }
        else{
            srgb.at<double>(k) = 1.055*pow(srgb.at<double>(k),double(1/2.4)) - 0.055;
        }
    }

    // Clipping
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

Mat luvtorgb(double L, double u, double v){

    double X, Y, Z;

    // Luv to XYZ
    double u_w = (double) (4.0*0.95)/(0.95+15.0+3.0*1.09);
    double v_w = (double) (9.0)/(0.95+15.0+3.0*1.09);

    double u_prime = (double(u) + 13.0*u_w*L)/(13.0*L);
    double v_prime = (double(v) + 13.0*v_w*L)/(13.0*L);

    if(L>7.9996){
        Y = (double) pow((L + 16.0)/116.0, 3);
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
            srgb.at<double>(k) = 1.055*pow(srgb.at<double>(k),double(1/2.4)) - 0.055;
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


int main(int argc, char** argv) {
    if(argc != 3) {
        cout << argv[0] << ": "
        << "got " << argc-1 << " arguments. Expecting two: width height." 
        << endl ;
        return(-1);
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int** RED1 = new int*[height];
    int** GREEN1 = new int*[height];
    int** BLUE1 = new int*[height];
    int** RED2 = new int*[height];
    int** GREEN2 = new int*[height];
    int** BLUE2 = new int*[height];

    for(int i = 0 ; i < height ; i++) {
        RED1[i] = new int[width];
        GREEN1[i] = new int[width];
        BLUE1[i] = new int[width];
        RED2[i] = new int[width];
        GREEN2[i] = new int[width];
        BLUE2[i] = new int[width];
    }

    for(int i = 0 ; i < height ; i++)
        for(int j = 0 ; j < width ; j++){
            int r1, g1, b1;
            int r2, g2, b2;

            double x = (double)j/(double)width;
            double y = (double)i/(double)height;
            double Y = 1.0;

            double L = 90;
            double u = x * 512 - 255;
            double v = y * 512 - 255;


            /* Your code should be placed here
            It should translate xyY to byte sRGB
            and Luv to byte sRGB
            */
            
            // Find a solution for xyY
            Mat srgb = xyytorgb(x, y, Y);

            // Find Solution for Luv            
            Mat srgb2 = luvtorgb(L, u, v);

            r1 = (int) (srgb.at<double>(0) * 255);
            g1 = (int) (srgb.at<double>(1) * 255);
            b1 = (int) (srgb.at<double>(2) * 255);
            // cout << "second " << r1 <<  endl;

            r2 = (int) (srgb2.at<double>(0) * 255);
            g2 = (int) (srgb2.at<double>(1) * 255);
            b2 = (int) (srgb2.at<double>(2) * 255);

            // this is the end of your code

            RED1[i][j] = r1;
            GREEN1[i][j] = g1;
            BLUE1[i][j] = b1;
            RED2[i][j] = r2;
            GREEN2[i][j] = g2;
            BLUE2[i][j] = b2;
        }


        Mat R1(height, width, CV_8UC1);
        Mat G1(height, width, CV_8UC1);
        Mat B1(height, width, CV_8UC1);

        Mat R2(height, width, CV_8UC1);
        Mat G2(height, width, CV_8UC1);
        Mat B2(height, width, CV_8UC1);

        for(int i = 0 ; i < height ; i++)
        for(int j = 0 ; j < width ; j++) {

        R1.at<uchar>(i,j) = RED1[i][j];
        G1.at<uchar>(i,j) = GREEN1[i][j];
        B1.at<uchar>(i,j) = BLUE1[i][j];

        R2.at<uchar>(i,j) = RED2[i][j];
        G2.at<uchar>(i,j) = GREEN2[i][j];
        B2.at<uchar>(i,j) = BLUE2[i][j];
    }

    Mat xyY;
    Mat xyY_planes[] = {B1, G1, R1};

    merge(xyY_planes, 3, xyY);
    namedWindow("xyY",CV_WINDOW_AUTOSIZE);
    imshow("xyY", xyY);

    Mat Luv;
    Mat Luv_planes[] = {B2, G2, R2};
    merge(Luv_planes, 3, Luv);
    namedWindow("Luv",CV_WINDOW_AUTOSIZE);
    imshow("Luv", Luv);
    waitKey(0); // Wait for a keystroke
    return(0);
}