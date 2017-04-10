#include "opencv2/highgui.hpp"
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if(argc != 3) {
        cout << argv[0] << ": "
        << "got " << argc-1 << " arguments. Expecting two: width height." 
        << endl ;
        return(-1);
    }
}
