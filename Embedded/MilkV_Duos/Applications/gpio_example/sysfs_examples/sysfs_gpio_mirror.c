#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "sysfs_gpio.h"

#define INPUT_GPIO 498
#define OUTPUT_GPIO 500

int main()
{
    // Validate GPIOs
    if (!is_valid_gpio(INPUT_GPIO) || !is_valid_gpio(OUTPUT_GPIO))
    {
        fprintf(stderr, "Invalid GPIO configuration.\n");
        return 1;
    }

    // Export and configure GPIOs
    export_gpio(INPUT_GPIO);
    export_gpio(OUTPUT_GPIO);

    set_gpio_direction(INPUT_GPIO, "in");
    set_gpio_direction(OUTPUT_GPIO, "out");

    while (1) // Continuous loop to mirror the input value
    {
        int value = read_gpio_value(INPUT_GPIO);
        if (value == -1)
        {
            fprintf(stderr, "Failed to read GPIO %d\n", INPUT_GPIO);
            break;
        }

        printf("GPIO %d value: %d -> Setting GPIO %d\n", INPUT_GPIO, value, OUTPUT_GPIO);
        set_gpio_value(OUTPUT_GPIO, value);

        usleep(500000); // Sleep for 500ms to reduce CPU usage
    }

    // Cleanup
    unexport_gpio(INPUT_GPIO);
    unexport_gpio(OUTPUT_GPIO);

    return 0;
}
