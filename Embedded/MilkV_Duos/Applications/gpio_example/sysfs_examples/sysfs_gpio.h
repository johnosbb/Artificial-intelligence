#ifndef SYSFS_GPIO_H
#define SYSFS_GPIO_H

// Uncomment the next line to enable debug messages
// #define ENABLE_DEBUG

#define VALID_GPIO_COUNT 8 // Total number of valid GPIOs

// List of valid GPIOs
extern const int valid_gpios[VALID_GPIO_COUNT];

// Function to check if a given pin is valid
int is_valid_gpio(int pin);

// Function to write to a sysfs path
void write_to_sysfs(const char *path, const char *value);

// Function to export a GPIO pin
void export_gpio(int pin);

// Function to set the direction of a GPIO pin
void set_gpio_direction(int pin, const char *direction);

// Function to set the value of a GPIO pin
void set_gpio_value(int pin, int value);

// Function to read the value of a GPIO pin
int read_gpio_value(int pin);

// Function to validate if a string is a valid integer
int is_valid_integer(const char *str);

// Function to unexport a GPIO Pin
void unexport_gpio(int pin);

#endif // SYSFS_GPIO_H
