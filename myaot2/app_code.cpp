#include <iostream>
#include <fstream>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "myaot2/mnist_graph.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// ビットマップ画像をロードするクラス
class RawImage
{
public:
    RawImage(){}

    // 入力画像の前処理済みのデータ
    float data[1][784];

    bool loadImage(std::string filename){
        // ファイル読み込み
        std::ifstream fin( filename.c_str(), std::ios::in | std::ios::binary );
        if (!fin){
            std::cout << "cannot open file:" << filename << std::endl;
            return false;
        }

        // ビットマップファイルのヘッダを読み飛ばします(54byte)
        for(int i=0; i<54; i++){
            char dmy;
            fin.read(&dmy, sizeof(dmy) );
        }

        // 画像データを1byteずつ読み込みます
        // ResNet50モデルの入力に使えるように前処理も行います

        // ビットマップ形式ではY軸は上下反転で格納されているため
        // 画像情報を左下から読み込んで、順番に格納していきます
        int idx=0;
        for(int y=27; y>=0; y--){
            for(int x=0; x<28; x++ ){
                if(!fin.eof()){
                    // 1byteずつ読み込みます
                    unsigned char v;
                    fin.read( (char*)&v, sizeof(v) );

                    // (学習時と同じ前処理を実施)
                    float vv = (float)v;
                    vv = vv / 255.0;
                    data[0][idx++] = vv;
                }
            }
        }
        fin.close();
    }
};


// 推論実行関数
int run(const float *input1, const float *input2, float *output, int output_size) {
    Eigen::ThreadPool tp(std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());
    MnistGraph mnist_graph;

    mnist_graph.set_thread_pool(&device);

    // 入力値のセット
    mnist_graph.set_arg0_data((void*)input1);
    mnist_graph.set_arg1_data((void*)input2);

    // 推論実行
    auto success = mnist_graph.Run();
    if(not success){
      return -1;
    }

    // 推論結果を取得
    std::copy(mnist_graph.result0_data(), mnist_graph.result0_data() + output_size/sizeof(float), output);
    return 0;
}


// 配列内の最大値となる添え字を返す
int max(float* array, int len){
    int max_i = 0;
    for(int i=0; i<len; i++){
        if(array[i] > array[max_i] ){
            max_i = i;
        }
    }
    return max_i;
}



// Main関数
int main(){
    RawImage rawimg;        // ビットマップ画像
    float result[1][10];  // 結果取得用

    // 28x28のビットマップイメージを取得
    rawimg.loadImage("./2.bmp");
    const float *input1 = (const float*)rawimg.data;
    const float input2[10] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0};

    // MNISTの推論モデルに取得したビットマップ画像to seikaiを入力し結果を取得する
    int ret = run(input1, input2, (float*)result, sizeof(result) );

    // 最大のスコアを持つIDを検索する(ResNet50の分類クラスのどれにヒットしたか確認)
    int max_i = max( &(result[0][0]), 10 );

    // 結果出力
    std::cout << "result max_i is " << max_i << std::endl;
    return 0;
}
