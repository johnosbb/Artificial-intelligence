cmake_minimum_required(VERSION 3.10)

# Project name
project(YOLO_C_EXAMPLE)







# Add the executable
add_executable(yolo_example main.c)

# Add the executable
add_executable(yolo_example2 main2.c)



# Link OpenCV libraries
target_link_libraries(yolo_example darknet)

# Link OpenCV libraries
target_link_libraries(yolo_example2 darknet lm)


