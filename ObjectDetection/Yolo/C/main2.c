#include <darknet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int target_width = 608;
int target_height = 608;

void print_usage(char *program_name)
{
    printf("Usage: %s <cfg_file> <weight_file> <names_file> <image_file> [threshold]\n", program_name);
    printf("Example: %s cfg/yolov3.cfg yolov3.weights data/coco.names test.jpg 0.5\n", program_name);
    printf("\nArguments:\n");
    printf("  cfg_file     - Path to YOLO configuration file\n");
    printf("  weight_file  - Path to YOLO weights file\n");
    printf("  names_file   - Path to class names file (e.g., coco.names)\n");
    printf("  image_file   - Path to image file to process\n");
    printf("  threshold    - Detection threshold (optional, default: 0.5)\n");
}

int file_exists(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file)
    {
        fclose(file);
        return 1;
    }
    return 0;
}

char **load_class_names(char *filename, int *num_classes)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Failed to open names file: %s\n", filename);
        return NULL;
    }

    char **names = calloc(1000, sizeof(char *)); // Using calloc for safer initialization
    if (!names)
    {
        fprintf(stderr, "Error: Memory allocation failed for names array\n");
        fclose(file);
        return NULL;
    }

    char line[256];
    int count = 0;

    while (fgets(line, sizeof(line), file))
    {
        int len = strlen(line);
        if (len > 0 && line[len - 1] == '\n')
        {
            line[len - 1] = '\0';
        }

        names[count] = strdup(line); // Using strdup for safer string duplication
        if (!names[count])
        {
            fprintf(stderr, "Error: Memory allocation failed for name at index %d\n", count);
            free_class_names(names, count);
            fclose(file);
            return NULL;
        }
        count++;
    }

    fclose(file);
    *num_classes = count;
    return names;
}

void free_class_names(char **names, int count)
{
    if (names)
    {
        for (int i = 0; i < count; i++)
        {
            free(names[i]);
        }
        free(names);
    }
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        print_usage(argv[0]);
        return 1;
    }

    char *cfg_file = argv[1];
    char *weight_file = argv[2];
    char *names_file = argv[3];
    char *filename = argv[4];
    float thresh = (argc > 5) ? atof(argv[5]) : 0.5;
    float nms_thresh = 0.45; // Moved declaration here

    // File validation
    if (!file_exists(cfg_file) || !file_exists(weight_file) ||
        !file_exists(names_file) || !file_exists(filename))
    {
        fprintf(stderr, "Error: One or more input files not found\n");
        return 1;
    }

    // Load class names
    int num_classes = 0;
    char **class_names = load_class_names(names_file, &num_classes);
    if (!class_names)
    {
        return 1;
    }

    printf("Loading network...\n");

    // Force CPU mode and initialize darknet
    gpu_index = -1;

    // Load the network with error checking
    network *net = load_network(cfg_file, weight_file, 0);
    if (!net)
    {
        fprintf(stderr, "Error: Failed to load network\n");
        free_class_names(class_names, num_classes);
        return 1;
    }

    // Configure network for CPU inference
    net->batch = 1;
    net->gpu_index = -1;

    // Load and process image with error checking
    image im = load_image_color(filename, target_width, target_height);
    if (!im.data || im.w <= 0 || im.h <= 0)
    {
        fprintf(stderr, "Error: Failed to load image or invalid dimensions\n");
        free_class_names(class_names, num_classes);
        free_network(*net);
        free(net);
        return 1;
    }

    // Resize image to network size with error checking
    image sized = letterbox_image(im, net->w, net->h);
    if (!sized.data)
    {
        fprintf(stderr, "Error: Failed to resize image\n");
        free_image(im);
        free_class_names(class_names, num_classes);
        free_network(*net);
        free(net);
        return 1;
    }

    layer l = net->layers[net->n - 1];

    // Prepare network prediction
    float *X = sized.data;
    double time = what_time_is_it_now();

    // Set network to inference mode
    net->train = 0;
    net->delta = 0;

    // Perform prediction with error checking
    if (network_predict(*net, X) < 0)
    {
        fprintf(stderr, "Error: Failed during network prediction\n");
        free_image(sized);
        free_image(im);
        free_class_names(class_names, num_classes);
        free_network(*net);
        free(net);
        return 1;
    }

    printf("Predicted in %f seconds.\n", what_time_is_it_now() - time);

    // Get detections with error checking
    int nboxes = 0;
    float hier_thresh = 0.5;
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, 1);
    if (!dets)
    {
        fprintf(stderr, "Error: Failed to get network boxes\n");
        free_image(sized);
        free_image(im);
        free_class_names(class_names, num_classes);
        free_network(*net);
        free(net);
        return 1;
    }

    // Apply NMS if threshold is positive
    if (nms_thresh > 0)
    {
        do_nms_sort(dets, nboxes, l.classes, nms_thresh);
    }

    // Print detections
    printf("\nObjects detected:\n");
    int detection_count = 0;

    for (int i = 0; i < nboxes; ++i)
    {
        for (int j = 0; j < l.classes; ++j)
        {
            if (dets[i].prob && dets[i].prob[j] > thresh)
            {
                detection_count++;
                box b = dets[i].bbox;
                printf("- Object %d: %s (%.1f%%)\n",
                       detection_count,
                       class_names[j],
                       dets[i].prob[j] * 100);
                printf("  Box: x=%d, y=%d, w=%d, h=%d\n",
                       (int)(b.x * im.w),
                       (int)(b.y * im.h),
                       (int)(b.w * im.w),
                       (int)(b.h * im.h));
            }
        }
    }

    if (detection_count == 0)
    {
        printf("No objects detected above threshold %.2f\n", thresh);
    }

    // Clean up
    free_detections(dets, nboxes);
    free_image(sized);
    free_image(im);
    free_network(*net);
    free(net);
    free_class_names(class_names, num_classes);

    return 0;
}