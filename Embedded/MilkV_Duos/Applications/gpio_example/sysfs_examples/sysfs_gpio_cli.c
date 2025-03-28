#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include "sysfs_gpio.h" // Include the GPIO module

int main(int argc, char *argv[])
{
    if (argc != 3 && argc != 4) // Ensure either 3 or 4 arguments
    {
        fprintf(stderr, "Invalid command. Usage: %s <GPIO_PIN> <VALUE (0 or 1)> [read]\n", argv[0]);
        return 1;
    }

    int gpio_pin = atoi(argv[1]); // Convert GPIO pin argument to integer
#ifdef ENABLE_DEBUG
    printf("Requested GPIO pin: %d\n", gpio_pin); // Debug
#endif

    if (!is_valid_gpio(gpio_pin)) // Check if GPIO is valid
    {
        fprintf(stderr, "Invalid GPIO pin: %d is not allowed.\n", gpio_pin);
        return 1;
    }

    // If there's a "read" argument, handle reading GPIO value
    if (argc == 3 && strcmp(argv[2], "read") == 0) // Read operation (pin and 'read')
    {
#ifdef ENABLE_DEBUG
        printf("Reading GPIO %d value...\n", gpio_pin); // Debug
#endif

        export_gpio(gpio_pin);              // Export the GPIO pin if necessary
        set_gpio_direction(gpio_pin, "in"); // Set direction to input for reading

        // Read GPIO value
        int value = read_gpio_value(gpio_pin);
        if (value == -1)
        {
            fprintf(stderr, "Failed to read GPIO %d value.\n", gpio_pin);
            return 1;
        }

        printf("GPIO %d current value: %d\n", gpio_pin, value);
    }
    else if (argc == 4) // Set operation (pin and value)
    {
        int value = atoi(argv[2]); // Get the value (0 or 1) for setting the pin
#ifdef ENABLE_DEBUG
        printf("Setting GPIO %d to value: %d\n", gpio_pin, value); // Debug
#endif

        if (value != 0 && value != 1)
        {
            fprintf(stderr, "Invalid value: Must be 0 or 1.\n");
            return 1;
        }

        // Set GPIO value
        export_gpio(gpio_pin);               // Export the GPIO pin if necessary
        set_gpio_direction(gpio_pin, "out"); // Set the direction to output
        set_gpio_value(gpio_pin, value);     // Set the value (0 or 1)
        printf("Setting GPIO %d to %d...\n", gpio_pin, value);
    }
    else
    {
        fprintf(stderr, "Invalid command. Usage: %s <GPIO_PIN> <VALUE (0 or 1)> [read|write]\n", argv[0]);
        fprintf(stderr, "Example Write. Usage: %s  500 0 write\n", argv[0]);
        fprintf(stderr, "Example Read. Usage: %s   498 read\n", argv[0]);
        return 1;
    }

    return 0;
}
