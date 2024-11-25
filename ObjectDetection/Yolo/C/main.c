#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> // Include time.h for measuring time
#include <darknet.h>

#define CLASS_FILE "./coco.names"
#define DETECTION_THRESHOLD 0.3
#define NMS_THRESHOLD 0.4

// Original image size (640x424)
int original_width = 640;
int original_height = 424;

// Target image size (608x608)
int target_width = 608;
int target_height = 608;

typedef struct
{
    float x, y, w, h;
} bounding_box;

// Function to convert normalized bounding box to original image size
void convert_bbox_to_original_size(bounding_box *bbox, int original_width, int original_height, int target_width, int target_height)
{
    // Convert normalized coordinates to the original image size
    float x_original = bbox->x * target_width;
    float y_original = bbox->y * target_height;
    float width_original = bbox->w * target_width;
    float height_original = bbox->h * target_height;

    // Rescale to the original image size
    bbox->x = x_original * (original_width / (float)target_width);
    bbox->y = y_original * (original_height / (float)target_height);
    bbox->w = width_original * (original_width / (float)target_width);
    bbox->h = height_original * (original_height / (float)target_height);
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

    // Measure time for bounding box conversion
    start_time = clock();
    // Non-maximal suppression to remove redundant detections
    do_nms_sort(dets, num_boxes, l.classes, NMS_THRESHOLD);

    for (int i = 0; i < num_boxes; i++)
    {
        for (int j = 0; j < l.classes; j++)
        {
            if (dets[i].prob[j] > 0.5)
            { // Detection threshold
                // Print the detected object class and probability before the transformation
                printf("Detected object: Class %s, Probability %.2f, Box [%.2f, %.2f, %.2f, %.2f]\n",
                       class_names[j], dets[i].prob[j], dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);

                // Create a bounding box struct for conversion
                bounding_box bbox = {dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h};

                // Convert the bounding box from 608x608 normalized to original image size (640x424)
                convert_bbox_to_original_size(&bbox, original_width, original_height, target_width, target_height);

                // Print the converted bounding box coordinates in the original image size
                printf("Converted Box in original image size (640x424): [%.2f, %.2f, %.2f, %.2f]\n",
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
