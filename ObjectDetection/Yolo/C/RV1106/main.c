#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> // Include time.h for measuring time
#include <darknet.h>
#include <ctype.h>
#include <sys/time.h>
#include <stb/stb_image.h> // a lite weight library for basic image manipulation

#define CLASS_FILE "coco.names"
#define DETECTION_THRESHOLD 0.3
#define NMS_THRESHOLD 0.4

// Original image size (640x424)
// int original_width = 640;
// int original_height = 424;

typedef struct
{
    float x, y, w, h;
} bounding_box;

void get_image_dimensions(const char *image_file, int *width, int *height)
{
    int channels;
    unsigned char *data = stbi_load(image_file, width, height, &channels, 0);
    if (data == NULL)
    {
        fprintf(stderr, "Error: Failed to load image %s using stb_image.\n", image_file);
        *width = 0;
        *height = 0;
        return;
    }
    stbi_image_free(data); // Free the image data after retrieving dimensions
}

// Function to convert normalized bounding box to original image size, accounting for letterboxing
void convert_bbox_to_original_size(bounding_box *bbox, int original_width, int original_height, int target_width, int target_height)
{
    // Ratio between original image and YOLO input size
    float scale_x = original_width / (float)target_width;
    float scale_y = original_height / (float)target_height;

    // Convert normalized coordinates to target image size
    float x = bbox->x * target_width;
    float y = bbox->y * target_height;
    float w = bbox->w * target_width;
    float h = bbox->h * target_height;

    // Adjust for letterboxing if aspect ratios differ
    float net_aspect_ratio = target_width / (float)target_height;
    float image_aspect_ratio = original_width / (float)original_height;

    float dx = 0, dy = 0; // Offsets due to padding
    if (net_aspect_ratio > image_aspect_ratio)
    {
        // Horizontal padding
        float pad = (target_width - (image_aspect_ratio * target_height)) / 2.0;
        dx = pad;
    }
    else if (net_aspect_ratio < image_aspect_ratio)
    {
        // Vertical padding
        float pad = (target_height - (original_width / net_aspect_ratio)) / 2.0;
        dy = pad;
    }

    // Adjust coordinates considering padding
    x = (x - dx) * scale_x;
    y = (y - dy) * scale_y;
    w = w * scale_x;
    h = h * scale_y;

    // Update bounding box
    bbox->x = x;
    bbox->y = y;
    bbox->w = w;
    bbox->h = h;
}

// Function to extract width and height from YOLO config
int parse_cfg_for_image_size(const char *cfg_file, int *width, int *height)
{
    FILE *file = fopen(cfg_file, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Failed to open cfg file %s\n", cfg_file);
        return -1;
    }

    char line[256];
    int in_net_section = 0;

    while (fgets(line, sizeof(line), file))
    {
        // Remove trailing spaces and newline characters
        char *end = line + strlen(line) - 1;
        while (end > line && isspace((unsigned char)*end))
            end--;
        *(end + 1) = '\0';

        // Detect start of [net] section
        if (strcmp(line, "[net]") == 0)
        {
            in_net_section = 1;
        }
        else if (line[0] == '[' && line[strlen(line) - 1] == ']') // End of [net] section
        {
            in_net_section = 0;
        }

        // Parse width and height only if inside [net] section
        if (in_net_section)
        {
            if (strncmp(line, "width=", 6) == 0)
            {
                *width = atoi(line + 6);
            }
            else if (strncmp(line, "height=", 7) == 0)
            {
                *height = atoi(line + 7);
            }
        }
    }

    fclose(file);

    // Validate parsed values
    if (*width <= 0 || *height <= 0)
    {
        fprintf(stderr, "Error: Invalid width or height in cfg file.\n");
        return -1;
    }

    return 0;
}

// Function to load class names
char **load_class_names(char *filename, int *num_classes)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Failed to open class names file %s\n", filename);
        perror("Reason");
        return NULL;
    }

    char **names = (char **)malloc(sizeof(char *) * 1000); // Arbitrary max number of classes
    *num_classes = 0;

    char line[256];
    while (fgets(line, sizeof(line), file))
    {
        size_t len = strlen(line);
        if (line[len - 1] == '\n')
        {
            line[len - 1] = '\0'; // Remove newline character
        }
        names[*num_classes] = strdup(line); // Copy the name
        (*num_classes)++;
    }
    fclose(file);

    return names;
}

// Function to calculate the time difference in milliseconds using timespec
double get_time_difference_ms(struct timespec start_time, struct timespec end_time)
{
    // Calculate the difference in seconds and nanoseconds
    double diff_sec = end_time.tv_sec - start_time.tv_sec;
    double diff_nsec = end_time.tv_nsec - start_time.tv_nsec;

    // If nanoseconds is negative, adjust the seconds and nanoseconds accordingly
    if (diff_nsec < 0)
    {
        diff_sec--;
        diff_nsec += 1000000000; // Add one second worth of nanoseconds
    }

    // Convert the difference into milliseconds
    return (diff_sec * 1000.0) + (diff_nsec / 1000000.0);
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        fprintf(stderr, "Usage: %s <cfg-file> <weights-file> <class-file> <image-file>\n", argv[0]);
        return 1;
    }

    char *cfg_file = argv[1];     // Path to YOLO configuration file
    char *weights_file = argv[2]; // Path to YOLO weights file
    char *class_file = argv[3];   // Path to the class names file
    char *image_file = argv[4];   // Path to the image file to analyze

    // Variables to hold width and height from cfg
    int target_width = 608; // Default fallback
    int target_height = 608;
    int original_width, original_height;
    // Parse the cfg file for the expected target width and height
    if (parse_cfg_for_image_size(cfg_file, &target_width, &target_height) != 0)
    {
        fprintf(stderr, "Error: Failed to parse cfg file for image size.\n");
        return 1;
    }

    printf("Using input size: %dx%d as per %s\n", target_width, target_height, cfg_file);

    // Start timing
    struct timespec start_time_ms, end_time_ms;

    // Record the start time

    double total_time_ms, load_network_time_ms, load_image_time_ms,
        prediction_time_ms, post_processing_time_ms, conversion_time_ms, load_classes_time_ms;
    ;

    // Measure time for loading the network
    clock_gettime(CLOCK_MONOTONIC, &start_time_ms);
    // load_network custom reads the YOLO configuration file (usually a .cfg file) to set up the architecture of the neural network.
    // The configuration specifies:
    // The network structure (e.g., number of layers, types of layers like convolutional, max pooling, etc.).
    // The input dimensions (e.g., width, height, and channels).
    // Training-specific parameters like batch size, subdivisions, learning rate, etc.
    // It Dynamically allocates memory to store the layers and parameters as defined in the .cfg file.
    // Initializes layer parameters with default or specified values.
    // Reads the binary weights file (usually a .weights file) and updates the network's weights.
    // The argument train is typically used to indicate this:
    // train = 0 → Inference mode (disable gradients).
    // train = 1 → Training mode (enable gradients for backpropagation).

    network *net = load_network_custom(cfg_file, weights_file, 0, 1); // For inference, batch size is usually set to 1, meaning the network processes one image at a time.
    if (!net)
    {
        fprintf(stderr, "Error: Failed to load YOLO network from %s and %s.\n", cfg_file, weights_file);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time_ms);
    load_network_time_ms = get_time_difference_ms(start_time_ms, end_time_ms);
    printf("Network loaded with %d layers.\n", net->n);

    set_batch_network(net, 1); // Set batch size to 1 for inference

    // Measure time for loading the image
    clock_gettime(CLOCK_MONOTONIC, &start_time_ms);
    image im = load_image_color(image_file, target_width, target_height);
    if (!im.data)
    {
        fprintf(stderr, "Error: Failed to load image %s.\n", image_file);
        free_network_ptr(net); // Use free_network_ptr for pointer
        return 1;
    }

    get_image_dimensions(image_file, &original_width, &original_height);

    if (original_width == 0 || original_height == 0)
    {
        fprintf(stderr, "Error: Could not determine original image dimensions.\n");
        return 1;
    }

    printf("Original image dimensions: %dx%d\n", original_width, original_height);
    clock_gettime(CLOCK_MONOTONIC, &end_time_ms);
    load_image_time_ms = get_time_difference_ms(start_time_ms, end_time_ms);

    // Measure time for loading class names
    clock_gettime(CLOCK_MONOTONIC, &start_time_ms);
    int num_classes = 0;
    char **class_names = load_class_names(CLASS_FILE, &num_classes);
    clock_gettime(CLOCK_MONOTONIC, &end_time_ms);
    load_classes_time_ms = get_time_difference_ms(start_time_ms, end_time_ms);
    if (!class_names)
    {
        fprintf(stderr, "Error loading class names.\n");
        return 1;
    }

    // Measure time for running YOLO prediction
    clock_gettime(CLOCK_MONOTONIC, &start_time_ms);
    network_predict_ptr(net, im.data); // Use network_predict_ptr
    clock_gettime(CLOCK_MONOTONIC, &end_time_ms);
    prediction_time_ms = get_time_difference_ms(start_time_ms, end_time_ms);

    layer output_layer = net->layers[net->n - 1];                     // The output layer
    int num_boxes = output_layer.w * output_layer.h * output_layer.n; // Number of detections

    // Measure time for decoding detections
    clock_gettime(CLOCK_MONOTONIC, &start_time_ms);

    detection *dets = get_network_boxes(net, im.w, im.h, DETECTION_THRESHOLD, 0.5, 0, 1, &num_boxes, 0); // Pass 0 or 1 for `letter`
    if (!dets)
    {
        fprintf(stderr, "Error: Failed to get network detections.\n");
        free_image(im);
        free_network_ptr(net);
        return 1;
    }

    // Non-maximal suppression to remove redundant detections
    do_nms_sort(dets, num_boxes, output_layer.classes, NMS_THRESHOLD);
    // The YOLO model outputs bounding box coordinates as relative values between 0 and 1,
    // normalized to the size of the input image.

    // The bounding box coordinates are given in the format [x, y, w, h], where:
    //  x: x-coordinate of the box's center (normalized).
    //  y: y-coordinate of the box's center (normalized).
    //  w: Width of the bounding box (normalized).
    //  h: Height of the bounding box (normalized).
    for (int i = 0; i < num_boxes; i++)
    {
        for (int j = 0; j < output_layer.classes; j++)
        {
            if (dets[i].prob[j] > 0.5)
            { // Detection threshold
                // Print the detected object class and probability before the transformation
                printf("Detected object: Class %s, Probability %.2f, Box with normalised locations [Center: (%.2f, %.2f) Width: %.2f Height: %.2f]\n",
                       class_names[j], dets[i].prob[j], dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);

                // Create a bounding box struct for conversion
                bounding_box bbox = {dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h};

                // Convert the bounding box from normalized YOLO output to original image size
                convert_bbox_to_original_size(&bbox, original_width, original_height, target_width, target_height);

                // Print the converted bounding box coordinates in the original image size
                printf("Box in original image size pixel locations: Box [Center: (%.2f, %.2f) Width: %.2f Height: %.2f]\n",
                       bbox.x, bbox.y, bbox.w, bbox.h);
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time_ms);
    post_processing_time_ms = get_time_difference_ms(start_time_ms, end_time_ms);

    // Free resources
    free_detections(dets, num_boxes);
    free_image(im);
    free_network_ptr(net);

    // Free the class names
    for (int i = 0; i < num_classes; i++)
    {
        free(class_names[i]);
    }
    free(class_names);

    // Output timing statistics
    total_time_ms = load_network_time_ms + load_image_time_ms + prediction_time_ms + post_processing_time_ms + load_classes_time_ms;
    printf("\nTiming statistics:\n");
    printf("Network loading time (ms): %.2f\n", load_network_time_ms);
    printf("Image loading time (ms): %.2f \n", load_image_time_ms);
    printf("Class names loading time: (ms) %.2f \n", load_classes_time_ms);
    printf("Prediction time (ms): %.2f\n", prediction_time_ms);
    printf("post_processing time (ms): %.2f seconds\n", post_processing_time_ms);
    printf("Total time: (ms) %.2f \n", total_time_ms);
    return 0;
}
