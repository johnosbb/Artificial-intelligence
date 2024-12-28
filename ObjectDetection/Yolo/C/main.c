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
    // Print anchor box dimensions for each YOLO layer
    for (int i = 0; i < net->n; i++)
    {
        layer l = net->layers[i];
        if (l.type == YOLO)
        {
            printf("\nLayer %d (YOLO Layer): Grid Size: %dx%d, Number of Anchors: %d\n", i, l.w, l.h, l.n);
            printf("Anchors:\n");
            for (int j = 0; j < l.n; j++)
            {
                printf("  Anchor %d: Width: %.2f, Height: %.2f\n", j, l.biases[2 * j], l.biases[2 * j + 1]);
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

void show_raw_predictions(layer l, int image_width, int image_height)
{
    // Iterate through the grid cells and anchors
    for (int i = 0; i < l.w * l.h; i++)
    {
        for (int j = 0; j < l.n; j++) // Iterate over the anchors (l.n is the number of anchors)
        {
            int index = i * l.n + j; // Access each anchor in each grid cell

            // Get the raw prediction for this grid cell and anchor
            float x = l.output[index + 0];   // Center x
            float y = l.output[index + 1];   // Center y
            float w = l.output[index + 2];   // Width
            float h = l.output[index + 3];   // Height
            float obj = l.output[index + 4]; // Objectness score

            // Calculate the grid cell position (grid_x, grid_y)
            int grid_x = i % l.w;
            int grid_y = i / l.w;

            // Output the raw prediction details
            printf("\nGrid Cell %d, Anchor %d for grid size %dx%d\n", i, j, l.w, l.h);
            printf("  Raw Prediction x: %.2f, y: %.2f, w: %.2f, h: %.2f, obj: %.2f\n", x, y, w, h, obj);

            // Access the anchor sizes from the biases array
            float anchor_w = l.biases[2 * j];     // Width of the anchor box
            float anchor_h = l.biases[2 * j + 1]; // Height of the anchor box

            // Variables to hold pixel-based values for bounding box
            float px, py, pw, ph;

            // Call the function to calculate the bounding box in pixels
            calculate_bounding_box(grid_x, grid_y, x, y, w, h, anchor_w, anchor_h, l.w, l.h, image_width, image_height, &px, &py, &pw, &ph);

            // Output the calculated bounding box in pixels
            printf("  Bounding Box (in pixels):\n");
            printf("    Center: (%.2f, %.2f)\n", px, py);
            printf("    Width: %.2f, Height: %.2f\n", pw, ph);
        }
    }
}

// Function to dump raw predictions to a file
void dump_raw_predictions(layer l, char *filename, int image_width, int image_height)
{
    // Open the file to write the predictions
    FILE *file = fopen(filename, "w");

    // Check if the file opened successfully
    if (file == NULL)
    {
        printf("Failed to open file for writing\n");
        return;
    }

    // Write the header to the CSV file
    fprintf(file, "grid_width,grid_height,grid_cell,anchor_number,x,y,w,h,objectness,px,py,pw,ph\n");

    // Iterate through the grid cells and anchors
    for (int i = 0; i < l.w * l.h; i++)
    {
        for (int j = 0; j < l.n; j++)
        {                            // Iterate over the anchors (l.n is the number of anchors)
            int index = i * l.n + j; // Access each anchor in each grid cell

            // Get the raw prediction for this grid cell and anchor
            float x = l.output[index + 0];   // Center x
            float y = l.output[index + 1];   // Center y
            float w = l.output[index + 2];   // Width
            float h = l.output[index + 3];   // Height
            float obj = l.output[index + 4]; // Objectness score

            // Calculate the grid cell position (grid_x, grid_y)
            int grid_x = i % l.w;
            int grid_y = i / l.w;

            // Access the anchor sizes from the biases array
            float anchor_w = l.biases[2 * j];     // Width of the anchor box
            float anchor_h = l.biases[2 * j + 1]; // Height of the anchor box

            // Variables to hold pixel-based values
            float px, py, pw, ph;

            // Calculate the bounding box in pixels
            calculate_bounding_box(grid_x, grid_y, x, y, w, h, anchor_w, anchor_h, l.w, l.h, image_width, image_height, &px, &py, &pw, &ph);

            // Write the data for this raw prediction to the CSV file
            fprintf(file, "%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                    l.w, l.h, i, j, x, y, w, h, obj, px, py, pw, ph);
        }
    }

    // Close the file after writing
    fclose(file);

    printf("Raw predictions written to %s\n", filename);
}

void show_raw_predictions_for_yolo_layers(network *net, int target_width, int target_height)
{
    // Iterate over all layers of the network
    for (int i = 0; i < net->n; i++)
    {
        layer l = net->layers[i];

        // Check if the layer is a YOLO layer
        if (l.type == YOLO)
        {
            printf("\n== Raw Predictions for YOLO Layer %d ==\n", i);
            show_raw_predictions(l, target_width, target_height); // Call the function to show raw predictions for this YOLO layer
            char filename[150];
            sprintf(filename, "../data/raw_predictions_layer_%d.csv", i); // Create a unique filename based on the layer index
            dump_raw_predictions(l, filename, target_width, target_height);
        }
    }
}

/**
 * @brief Display grid cell responsibility for detections based on YOLO output.
 *
 * @param dets Pointer to an array of detections.
 * @param num_boxes Number of detected boxes.
 * @param target_width Width of the YOLO network input.
 * @param target_height Height of the YOLO network input.
 * @param grid_width Number of grid cells along the width in the output layer.
 * @param grid_height Number of grid cells along the height in the output layer.
 * @param detection_threshold Confidence threshold for filtering detections.
 */
void show_grid_cell_responsibility(detection *dets, int num_boxes,
                                   int target_width, int target_height,
                                   int grid_width, int grid_height,
                                   float detection_threshold)
{
    printf("\nGrid Cell Responsibility\n-----------------------\n");
    printf("Grid Width: %d, Grid Height: %d\n", grid_width, grid_height);
    for (int i = 0; i < num_boxes; i++)
    {
        detection det = dets[i];
        if (det.prob[0] > detection_threshold) // Assuming class 0 for simplicity
        {
            int grid_x = (int)(det.bbox.x * target_width) / (target_width / grid_width);
            int grid_y = (int)(det.bbox.y * target_height) / (target_height / grid_height);

            printf("\nDetection %d:\n", i);
            printf("  Class: %d, Confidence: %.2f\n", 0, det.prob[0]);
            printf("  Responsible Grid Cell: (%d, %d)\n", grid_x, grid_y);
            printf("  Bounding Box Center: (%.2f, %.2f)\n", det.bbox.x * target_width, det.bbox.y * target_height);
        }
    }
}

/**
 * @brief Calculate Intersection over Union (IoU) between two bounding boxes.
 *
 * @param box1 The first bounding box.
 * @param box2 The second bounding box.
 * @return float IoU value between two bounding boxes.
 */
float calculate_iou(bounding_box box1, box box2)
{
    float x1 = fmax(box1.x - box1.w / 2, box2.x - box2.w / 2);
    float y1 = fmax(box1.y - box1.h / 2, box2.y - box2.h / 2);
    float x2 = fmin(box1.x + box1.w / 2, box2.x + box2.w / 2);
    float y2 = fmin(box1.y + box1.h / 2, box2.y + box2.h / 2);

    float intersection = fmax(0, x2 - x1) * fmax(0, y2 - y1);
    float area1 = box1.w * box1.h;
    float area2 = box2.w * box2.h;
    float union_area = area1 + area2 - intersection;

    return union_area > 0 ? intersection / union_area : 0;
}

/**
 * @brief Display IoU between anchors and detected bounding boxes.
 *
 * @param dets Pointer to an array of detections.
 * @param num_boxes Number of detected boxes.
 * @param last_layer The YOLO output layer containing anchor biases.
 */
void show_iou(detection *dets, int num_boxes, layer last_layer)
{
    for (int i = 0; i < last_layer.n; i++)
    {
        // Retrieve anchor box dimensions from layer biases
        bounding_box anchor = {0, 0, last_layer.biases[2 * i], last_layer.biases[2 * i + 1]};

        printf("\nAnchor %d [Width: %.2f, Height: %.2f]\n", i, anchor.w, anchor.h);

        for (int j = 0; j < num_boxes; j++)
        {
            float iou = calculate_iou(anchor, dets[j].bbox);
            printf("  Detection %d - IoU with Anchor %d: %.2f\n", j, i, iou);
        }
    }
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
    // Verify anchor placements
    printf("== Anchor Placements for Layer 16 ==\n");
    show_anchor_placements(net->layers[16]);

    printf("== Anchor Placements for Layer 23 ==\n");
    show_anchor_placements(net->layers[23]);
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
    show_raw_predictions_for_yolo_layers(net, target_width, target_height);
    // Measure time for decoding detections
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

    show_grid_cell_responsibility(
        dets,               // Detections array
        num_boxes,          // Number of detected boxes
        target_width,       // Network input width
        target_height,      // Network input height
        last_layer.w,       // Grid width from last layer
        last_layer.h,       // Grid height from last layer
        DETECTION_THRESHOLD // Minimum detection threshold
    );

    // Measure time for bounding box conversion
    start_time = clock();
    // Non-maximal suppression to remove redundant detections
    do_nms_sort(dets, num_boxes, l.classes, NMS_THRESHOLD);
    // The YOLO model outputs bounding box coordinates as relative values between 0 and 1,
    // normalized to the size of the input image.

    // Show IoU for anchors and detections
    show_iou(dets, num_boxes, last_layer);

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
