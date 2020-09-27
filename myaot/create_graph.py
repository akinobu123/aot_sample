#-------------------------------------------------
# Kerasサンプルのモデルを取得します
#-------------------------------------------------
import tensorflow as tf
model = tf.keras.applications.ResNet50()

#-------------------------------------------------
# グラフを保存します (test_graph.pb)
#-------------------------------------------------
ss = tf.keras.backend.get_session()
output_node_names = [node.op.name for node in model.outputs]
graphdef = tf.graph_util.convert_variables_to_constants(ss, ss.graph_def, output_node_names)
tf.train.write_graph(graphdef, '.', 'test_graph.pb', as_text=False)


#-------------------------------------------------
# configファイルを作成します
# (test_graph.config.pbtxt)
#-------------------------------------------------
import tf2xla_pb2

# configの取得
config = tf2xla_pb2.Config()

# feed (入力形式)の指定
batch_size = 1
for x in model.inputs:
    # shapeを[1,Dim(224),Dim(224),Dim(3)] (ResNet50の入力形式)にセット
    x.set_shape([batch_size] + list(x.shape)[1:])
    feed = config.feed.add()
    feed.id.node_name = x.op.name
    feed.shape.MergeFrom(x.shape.as_proto())

# fetch (出力形式)の指定
for x in model.outputs:
    fetch = config.fetch.add()
    fetch.id.node_name = x.op.name

# configファイルの保存
with open('test_graph.config.pbtxt', 'w') as fo:
    out_txt = str(config)
    # print(out_txt) # for display
    fo.write(out_txt)
