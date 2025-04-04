#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> // Include time.h for measuring time
#include <darknet.h>
#include <ctype.h>
#include <math.h>
#define CLASS_FILE "../data/coco.names"
#define DETECTION_THRESHOLD 0.3
#define NMS_THRESHOLD 0.4

// Original image size (640x424)
int original_width = 640;
int original_height = 424;

typedef struct
{
    float x, y, w, h;
} bounding_box;

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

void show_anchor_placements(layer l)
{
    printf("\nLayer Grid: %dx%d\n", l.w, l.h);
    for (int i = 0; i < l.n; i++)
    { // Loop through anchors
        printf("  Anchor %d: Width: %.2f, Height: %.2f\n",
               i, l.biases[2 * i], l.biases[2 * i + 1]);
    }

    printf("\nGrid Cell and Anchor Placement Example:\n");
    for (int grid_y = 0; grid_y < l.h; grid_y++)
    {
        for (int grid_x = 0; grid_x < l.w; grid_x++)
        {
            printf("Cell (%d, %d):\n", grid_x, grid_y);
            for (int a = 0; a < l.n; a++)
            {
                printf("  Anchor %d -> Center at Grid (%d, %d), Size: (%.2f, %.2f)\n",
                       a, grid_x, grid_y, l.biases[2 * a], l.biases[2 * a + 1]);
            }
        }
    }
}
void show_anchorbox_information(network *net)
{
    // Print anchor box dimensions for each YOLO layer with mask interpretation
    for (int i = 0; i < net->n; i++)
    {
        layer l = net->layers[i];
        if (l.type == YOLO)
        {
            printf("\nLayer %d (YOLO Layer): Grid Size: %dx%d, Number of Anchors: %d\n", i, l.w, l.h, l.n);
            printf("Image Input Size: %dx%d\n", net->w, net->h);
            printf("Anchors (based on mask):\n");

            for (int j = 0; j < l.n; j++)
            {
                int anchor_index = l.mask[j]; // Use mask to determine the anchor index
                float anchor_width = l.biases[2 * anchor_index];
                float anchor_height = l.biases[2 * anchor_index + 1];

                printf("  Anchor %d: Width: %.2f, Height: %.2f\n", j, anchor_width, anchor_height);
            }
        }
    }
}

// Function to calculate the bounding box in pixel space
void calculate_bounding_box(int grid_x, int grid_y, float x, float y, float w, float h,
                            float anchor_w, float anchor_h, int grid_width, int grid_height,
                            int image_width, int image_height, float *px, float *py, float *pw, float *ph)
{
    // Calculate the center coordinates (in pixels) of the bounding box
    *px = (grid_x + x) * (image_width / grid_width);
    *py = (grid_y + y) * (image_height / grid_height);

    // Calculate the width and height (in pixels) of the bounding box
    *pw = w * anchor_w;
    *ph = h * anchor_h;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <cfg-file> <weights-file> <image-file>\n", argv[0]);
        return 1;
    }

    char *cfg_file = argv[1];     // Path to YOLO configuration file
    char *weights_file = argv[2]; // Path to YOLO weights file
    char *image_file = argv[3];   // Path to the image file to analyze

    // Variables to hold width and height from cfg
    int target_width = 608; // Default fallback
    int target_height = 608;

    // Parse the cfg file for width and height
    if (parse_cfg_for_image_size(cfg_file, &target_width, &target_height) != 0)
    {
        fprintf(stderr, "Error: Failed to parse cfg file for image size.\n");
        return 1;
    }

    printf("Using input size: %dx%d as per %s\n", target_width, target_height, cfg_file);

    // Start timing
    clock_t start_time, end_time;
    double total_time, load_network_time, load_image_time, prediction_time, detection_time, conversion_time;

    // Measure time for loading the network
    start_time = clock();
    network *net = load_network_custom(cfg_file, weights_file, 0, 1); // Use load_network_custom
    if (!net)
    {
        fprintf(stderr, "Error: Failed to load YOLO network from %s and %s.\n", cfg_file, weights_file);
        return 1;
    }
    show_anchorbox_information(net);
    // // Verify anchor placements
    // printf("== Anchor Placements for Layer 16 ==\n");
    // show_anchor_placements(net->layers[16]);

    // printf("== Anchor Placements for Layer 23 ==\n");
    // show_anchor_placements(net->layers[23]);
    end_time = clock();
    load_network_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Network loaded with %d layers.\n", net->n);

    set_batch_network(net, 1); // Set batch size to 1 for inference

    // Measure time for loading the image
    start_time = clock();
    image im = load_image_color(image_file, target_width, target_height);
    if (!im.data)
    {
        fprintf(stderr, "Error: Failed to load image %s.\n", image_file);
        free_network_ptr(net); // Use free_network_ptr for pointer
        return 1;
    }

    printf("Original image dimensions: %dx%d\n", original_width, original_height);
    end_time = clock();
    load_image_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Measure time for loading class names
    start_time = clock();
    int num_classes = 0;
    char **class_names = load_class_names(CLASS_FILE, &num_classes);
    end_time = clock();
    detection_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    if (!class_names)
    {
        fprintf(stderr, "Error loading class names.\n");
        return 1;
    }

    // Measure time for running YOLO prediction
    start_time = clock();
    network_predict_ptr(net, im.data); // Use network_predict_ptr
    end_time = clock();
    prediction_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    layer l = net->layers[net->n - 1]; // The output layer
    int num_boxes = l.w * l.h * l.n;   // Number of detections
    // show_raw_predictions_for_yolo_layers(net, target_width, target_height);
    //  Measure time for decoding detections
    start_time = clock();

    detection *dets = get_network_boxes(net, im.w, im.h, DETECTION_THRESHOLD, 0.5, 0, 1, &num_boxes, 0); // Pass 0 or 1 for `letter`
    if (!dets)
    {
        fprintf(stderr, "Error: Failed to get network detections.\n");
        free_image(im);
        free_network_ptr(net);
        return 1;
    }
    end_time = clock();
    detection_time += ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    layer last_layer = net->layers[net->n - 1]; // Get the last layer of the network

    // Measure time for bounding box conversion
    start_time = clock();
    // Non-maximal suppression to remove redundant detections
    do_nms_sort(dets, num_boxes, l.classes, NMS_THRESHOLD);
    // The YOLO model outputs bounding box coordinates as relative values between 0 and 1,
    // normalized to the size of the input image.

    // The bounding box coordinates are given in the format [x, y, w, h], where:
    //  x: x-coordinate of the box's center (normalized).
    //  y: y-coordinate of the box's center (normalized).
    //  w: Width of the bounding box (normalized).
    //  h: Height of the bounding box (normalized).
    for (int i = 0; i < num_boxes; i++)
    {
        for (int j = 0; j < l.classes; j++)
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
    end_time = clock();
    conversion_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

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
    total_time = load_network_time + load_image_time + prediction_time + detection_time + conversion_time;
    printf("\nTiming statistics:\n");
    printf("Network loading time: %.2f seconds\n", load_network_time);
    printf("Image loading time: %.2f seconds\n", load_image_time);
    printf("Class names loading time: %.2f seconds\n", detection_time);
    printf("Prediction time: %.2f seconds\n", prediction_time);
    printf("Detection and conversion time: %.2f seconds\n", conversion_time);
    printf("Total time: %.2f seconds\n", total_time);

    printf("Detection complete.\n");
    return 0;
}
