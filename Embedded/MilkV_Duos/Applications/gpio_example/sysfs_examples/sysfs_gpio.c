#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include "sysfs_gpio.h" // Include the header file

// List of valid GPIOs
const int valid_gpios[VALID_GPIO_COUNT] = {
    500, 508, 509, 497, 496, 509, 510, 498};

// Function to check if a given pin is valid
int is_valid_gpio(int pin)
{
    for (int i = 0; i < VALID_GPIO_COUNT; i++)
    {
        if (valid_gpios[i] == pin)
        {
            return 1; // Valid GPIO
        }
    }
    return 0; // Invalid GPIO
}

void write_to_sysfs(const char *path, const char *value)
{
#ifdef ENABLE_DEBUG
    printf("Writing to %s with value: %s\n", path, value); // Debug
#endif
    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        perror("Failed to open file");
        return;
    }
    fprintf(fp, "%s", value);
    fclose(fp);
}

void unexport_gpio(int pin)
{
#ifdef ENABLE_DEBUG
    printf("Unexporting GPIO %d\n", pin); // Debug
#endif
    char pin_str[8];
    snprintf(pin_str, sizeof(pin_str), "%d", pin);
    write_to_sysfs("/sys/class/gpio/unexport", pin_str);
}

void export_gpio(int pin)
{
    char gpio_path[128];
    snprintf(gpio_path, sizeof(gpio_path), "/sys/class/gpio/gpio%d", pin);

    if (access(gpio_path, F_OK) == -1) // GPIO not exported
    {
#ifdef ENABLE_DEBUG
        printf("Exporting GPIO %d\n", pin); // Debug
#endif
        char pin_str[8];
        snprintf(pin_str, sizeof(pin_str), "%d", pin);
        write_to_sysfs("/sys/class/gpio/export", pin_str);
        usleep(200000); // Allow system to register export
    }
    else
    {
#ifdef ENABLE_DEBUG
        printf("GPIO %d is already exported\n", pin); // Debug
#endif
    }
}

void set_gpio_direction(int pin, const char *direction)
{
    char gpio_path[128];
    snprintf(gpio_path, sizeof(gpio_path), "/sys/class/gpio/gpio%d/direction", pin);
#ifdef ENABLE_DEBUG
    printf("Setting GPIO %d direction to: %s\n", pin, direction); // Debug
#endif
    write_to_sysfs(gpio_path, direction);
}

void set_gpio_value(int pin, int value)
{
    char gpio_path[128];
    snprintf(gpio_path, sizeof(gpio_path), "/sys/class/gpio/gpio%d/value", pin);

    char val_str[2];
    snprintf(val_str, sizeof(val_str), "%d", value);
#ifdef ENABLE_DEBUG
    printf("Setting GPIO %d value to: %d\n", pin, value); // Debug
#endif
    write_to_sysfs(gpio_path, val_str);
}

int read_gpio_value(int pin)
{
    char gpio_value_path[128];
    snprintf(gpio_value_path, sizeof(gpio_value_path), "/sys/class/gpio/gpio%d/value", pin);

    FILE *fp = fopen(gpio_value_path, "r");
    if (fp == NULL)
    {
        perror("Failed to read GPIO value");
        return -1;
    }

    char value;
    fscanf(fp, "%c", &value); // Read the value (0 or 1)
    fclose(fp);

    return value - '0'; // Convert char to int ('0' -> 0, '1' -> 1)
}

int is_valid_integer(const char *str)
{
    while (*str)
    {
        if (!isdigit(*str))
            return 0; // Not a number
        str++;
    }
    return 1;
}
