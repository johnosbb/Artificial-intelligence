#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

bool file_exists(const std::string &path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

int main()
{

    std::cout << cv::getBuildInformation() << std::endl;

    std::string imageFile = "image.jpg"; // Path to the local image

    // Check if the file exists
    if (!file_exists(imageFile))
    {
        std::cerr << "Error: The file '" << imageFile << "' does not exist!" << std::endl;
        return -1;
    }

    // Load the image using OpenCV
    cv::Mat frame = cv::imread(imageFile);

    if (frame.empty())
    {
        std::cerr << "Error: Could not load image file: " << imageFile << std::endl;
        std::cerr << "Possible reasons: " << std::endl;
        std::cerr << "1. The image format is not supported by OpenCV build." << std::endl;
        std::cerr << "2. The image file is corrupted." << std::endl;
        std::cerr << "3. Incorrect path to the image file." << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully!" << std::endl;
    // Continue with the rest of the code...

    return 0;
}
