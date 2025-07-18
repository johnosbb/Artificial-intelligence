#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main()
{
    // int width = 720;
    // int height = 480;
    int width = 1920;
    int height = 1080;
    // int width = 1600;
    // int height = 900;
    int frame_counter = 0;
    bool frame_saved = false;

    // Stop conflicting services
    system("RkLunch-stop.sh");

    // OpenCV video capture
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.open(0); // default camera

    if (!cap.isOpened())
    {
        fprintf(stderr, "❌ Failed to open camera via OpenCV.\n");
        return -1;
    }

    printf("📷 Waiting for frame to save...\n");

    while (true)
    {
        cv::Mat bgr;
        cap >> bgr;

        if (bgr.empty())
        {
            fprintf(stderr, "❌ Empty frame received from camera.\n");
            continue;
        }

        if (!frame_saved && frame_counter == 10)
        {
            cv::Mat rgb;
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB); // stb expects RGB

            bool ok = stbi_write_jpg("capture.jpg", rgb.cols, rgb.rows, 3, rgb.data, 90);
            if (ok)
            {
                char cwd[256];
                getcwd(cwd, sizeof(cwd));
                printf("✅ Saved capture.jpg to %s\n", cwd);
            }
            else
            {
                fprintf(stderr, "❌ Failed to write capture.jpg\n");
            }

            frame_saved = true;
            break; // exit after saving
        }

        frame_counter++;
    }

    return 0;
}
