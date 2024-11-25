#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <signal.h>
#include <fstream>

// Global variable for webcam capture
cv::VideoCapture cap;

// Graceful exit handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << "Interrupt signal (" << signum << ") received. Exiting..." << std::endl;
    if (cap.isOpened())
    {
        cap.release();
    }
    cv::destroyAllWindows();
    exit(signum);
}

int main()
{
    // Register signal handler
    signal(SIGINT, signalHandler);

    // Paths to YOLO files
    std::string modelConfiguration = "../data/yolov3-tiny.cfg";
    std::string modelWeights = "../data/yolov3-tiny.weights";
    std::string classesFile = "../data/coco.names";

    // Load class names
    std::vector<std::string> classes;
    std::ifstream ifs(classesFile.c_str());
    if (!ifs.is_open())
    {
        std::cerr << "Error: Could not open classes file." << classesFile.c_str() << std::endl;
        return -1;
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    // Load YOLO model
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    if (net.empty())
    {
        std::cerr << "Error: Could not load YOLO model." << std::endl;
        return -1;
    }

    // Get output layer names
    std::vector<std::string> outputLayerNames = net.getUnconnectedOutLayersNames();

    // Open webcam
    cap.open(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Cannot open webcam." << std::endl;
        return -1;
    }

    std::cout << "Press 'q' to quit." << std::endl;

    // Main processing loop
    while (true)
    {
        cv::Mat frame;
        cap >> frame; // Capture a frame
        if (frame.empty())
        {
            std::cerr << "Error: Empty frame. Exiting..." << std::endl;
            break;
        }

        // Get frame dimensions
        int frameHeight = frame.rows;
        int frameWidth = frame.cols;

        // Prepare input blob for YOLO
        cv::Mat blob = cv::dnn::blobFromImage(frame, 0.00392, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);

        // Set the input and run forward pass
        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, outputLayerNames);

        // Process YOLO outputs
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (const auto &out : outs)
        {
            auto *data = (float *)out.data;
            for (int i = 0; i < out.rows; ++i, data += out.cols)
            {
                cv::Mat scores = out.row(i).colRange(5, out.cols);
                cv::Point classIdPoint;
                double confidence;

                // Get the class ID with the highest score
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > 0.5)
                { // Detection threshold
                    int centerX = (int)(data[0] * frameWidth);
                    int centerY = (int)(data[1] * frameHeight);
                    int width = (int)(data[2] * frameWidth);
                    int height = (int)(data[3] * frameHeight);
                    int x = centerX - width / 2;
                    int y = centerY - height / 2;

                    boxes.push_back(cv::Rect(x, y, width, height));
                    confidences.push_back((float)confidence);
                    classIds.push_back(classIdPoint.x);
                }
            }
        }

        // Apply non-maximum suppression to eliminate redundant overlapping boxes
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        for (int idx : indices)
        {
            cv::Rect box = boxes[idx];
            std::string label = classes[classIds[idx]];
            float confidence = confidences[idx];

            // Draw rectangle and label on the frame
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, label + " " + cv::format("%.2f", confidence),
                        cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        // Display the resulting frame
        cv::imshow("Real-Time Object Detection", frame);

        // Exit on 'q' key press
        if (cv::waitKey(1) == 'q')
        {
            std::cout << "Exit key pressed. Closing..." << std::endl;
            break;
        }
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
