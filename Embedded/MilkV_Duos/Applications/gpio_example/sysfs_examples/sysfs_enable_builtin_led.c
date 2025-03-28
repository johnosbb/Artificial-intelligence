#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "sysfs_gpio.h" // Include the GPIO module

#define BUILTIN_LED 509 // GPIO_PIN_29

int main()
{
    int gpio_pin = BUILTIN_LED;
    if (!is_valid_gpio(gpio_pin)) // Check if GPIO is valid
    {
        fprintf(stderr, "Invalid GPIO pin: %d is not allowed.\n", gpio_pin);
        return 1;
    }
    // Initialize GPIO
    export_gpio(gpio_pin);
    set_gpio_direction(gpio_pin, "out");

    printf("Turning LED ON...\n");
    set_gpio_value(gpio_pin, 1);
    sleep(2);

    // Cleanup
    printf("Unexporting GPIO...\n");
    unexport_gpio(gpio_pin);

    return 0;
}
