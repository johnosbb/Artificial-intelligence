#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include "sysfs_gpio.h" // Include the GPIO module

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <GPIO_PIN> <VALUE (0 or 1)>\n", argv[0]);
        return 1;
    }

    if (!is_valid_integer(argv[1]) || !is_valid_integer(argv[2]))
    {
        fprintf(stderr, "Invalid input: GPIO pin and value must be numbers.\n");
        return 1;
    }

    int gpio_pin = atoi(argv[1]);
    int value = atoi(argv[2]);

    if (value != 0 && value != 1)
    {
        fprintf(stderr, "Invalid value: Must be 0 or 1.\n");
        return 1;
    }

    if (!is_valid_gpio(gpio_pin))
    {
        fprintf(stderr, "Invalid GPIO pin: %d is not allowed.\n", gpio_pin);
        return 1;
    }

    // Initialize GPIO
    export_gpio(gpio_pin);
    set_gpio_direction(gpio_pin, "out");

    // Set GPIO value
    printf("Setting GPIO %d to %d...\n", gpio_pin, value);
    set_gpio_value(gpio_pin, value);

    return 0;
}
