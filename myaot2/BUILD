load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

tf_library(
	name = 'mnist_graph',
	config = 'mnist_graph.config.pbtxt',
	cpp_class = 'MnistGraph',
	graph = 'mnist_graph.pb',
)

cc_binary(
    name = "app_code",
    srcs = [
        "app_code.cpp"
    ],
    deps = [
        ":mnist_graph",
        "//third_party/eigen3"
    ],
    linkopts = [
        "-lpthread"
    ],
)