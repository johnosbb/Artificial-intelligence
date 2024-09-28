# Description:
#   TensorFlow Lite preventative_maintenance example.

load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_binary(
    name = "preventative_maintenance",
    srcs = [
        "preventative_maintenance.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)
