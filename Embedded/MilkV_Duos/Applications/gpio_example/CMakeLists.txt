cmake_minimum_required(VERSION 3.10)
project(GPIO C)

# Set the C standard to use
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Wiringx examples
add_executable(gpio_example wiringx_examples/gpio_example.c)
add_executable(gpio_example_input wiringx_examples/gpio_example_input.c)
add_executable(gpio_example_output wiringx_examples/gpio_example_output.c)
add_executable(gpio_enable_led wiringx_examples/gpio_enable_led.c)
add_executable(gpio_enable_builtin_led wiringx_examples/gpio_enable_builtin_led.c)


# sysfs examples
add_executable(sysfs_enable_builtin_led sysfs_examples/sysfs_enable_builtin_led.c sysfs_examples/sysfs_gpio.c)
add_executable(sysfs_set_pin_value sysfs_examples/sysfs_set_pin_value.c sysfs_examples/sysfs_gpio.c)
add_executable(sysfs_gpio_cli sysfs_examples/sysfs_gpio_cli.c sysfs_examples/sysfs_gpio.c)
add_executable(sysfs_gpio_mirror sysfs_examples/sysfs_gpio_mirror.c sysfs_examples/sysfs_gpio.c)

# Ensure CMake looks in the right place for libraries
set(CMAKE_LIBRARY_PATH ${CMAKE_SYSROOT}/usr/lib)
set(CMAKE_INCLUDE_PATH ${CMAKE_SYSROOT}/usr/include)



target_link_libraries(gpio_example wiringx)
target_link_libraries(gpio_example_output wiringx)
target_link_libraries(gpio_example_input wiringx)
target_link_libraries(gpio_enable_led wiringx)
target_link_libraries(gpio_enable_builtin_led wiringx)

# Include wiringX headers
include_directories(${CMAKE_INCLUDE_PATH})

# Install target (optional)
install(TARGETS gpio_example_output gpio_enable_led gpio_example_input gpio_example RUNTIME DESTINATION bin)
