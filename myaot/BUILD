load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

tf_library(
    name = 'test_graph',
    config = 'test_graph.config.pbtxt',
    cpp_class = 'TestGraph',
    graph = 'test_graph.pb',
)

cc_binary(
    name = "my_code",
    srcs = [
        "my_code.cpp"
    ],
    deps = [
        ":test_graph",
        "//third_party/eigen3"
    ],
    linkopts = [
        "-lpthread"
    ],
)

