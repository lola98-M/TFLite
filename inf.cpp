#include <iostream>
#include <iomanip>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "opencv2/opencv.hpp"

using namespace std;
typedef cv::Point3_<float> Pixel;

const uint WIDTH = 28;
const uint HEIGHT = 28;
const uint CHANNEL = 1;
const uint OUTDIM = 1;

void normalize(Pixel &pixel){
    pixel.x = (pixel.x / 255.0);
    pixel.y = (pixel.y / 255.0);
    pixel.z = (pixel.z / 255.0);
}

int main(){
    std::vector<std::string> labels;
    auto file_name="labels.txt";
    std::ifstream input( file_name );

    for( std::string line; getline( input, line ); )
    {
        labels.push_back( line);
    }
        
    // read image file
    cv::Mat img = cv::imread("5.jpg");

    cv::Mat inputImg;
    img.convertTo(inputImg, CV_32FC3);
    cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2RGB);

    // normalize to -1 & 1
    Pixel* pixel = inputImg.ptr<Pixel>(0,0);
    const Pixel* endPixel = pixel + inputImg.cols * inputImg.rows;
    for (; pixel != endPixel; pixel++)
        normalize(*pixel);

    // resize image as model input
    cv::resize(inputImg, inputImg, cv::Size(WIDTH, HEIGHT));

    // create model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile("tf_mnist.tflite");
        
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    interpreter->AllocateTensors();

    float* inputLayer = interpreter->typed_input_tensor<float>(0);

    float* inputImg_ptr = inputImg.ptr<float>(0);
    memcpy(inputLayer, inputImg.ptr<float>(0),
            WIDTH * HEIGHT * CHANNEL * sizeof(float));

    interpreter->Invoke();

    float* outputLayer = interpreter->typed_output_tensor<float>(0);
    

    // TODO
    return 0;
}
