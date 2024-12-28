cmake_minimum_required(VERSION 3.10)

# Project name
project(YOLO_C_EXAMPLE)







# Add the executable
add_executable(yolo_example main.c)





# Link Darknet
target_link_libraries(yolo_example darknet m)




