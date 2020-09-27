#include <iostream>
#include <fstream>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "myaot/test_graph.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// ビットマップ画像をロードするクラス
class RawImage
{
    public:
    RawImage(){}

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

        // ビットマップ形式ではY軸は上下反転、カラー情報はBGR形式で格納されているため
        // 画像情報を左下から読み込んで、順番に格納していきます
        for(int y=223; y>=0; y--){
            for(int x=0; x<224; x++ ){
                for(int c=0; c<3; c++){
                    if(!fin.eof()){
                        // 1byteずつB->G->Rの順番で読み込みます
                        unsigned char v;
                        fin.read( (char*)&v, sizeof(v) );

                        // 前処理として画像の画素値よりVGG16の平均画素値を減算します
                        // (学習時と同じ前処理を実施)
                        float vv = (float)v;
                        const float mean[3] = {103.939,116.779,123.68}; // BGRのそれぞれの平均値
                        vv = vv - mean[c];
                        data[0][y][x][c] = vv;
                    }
                }
            }
        }
        fin.close();
    }

    // 入力画像の前処理済みのデータ
    float data[1][224][224][3];
};


// 推論実行関数
int run(const float *p_input, int input_size, float *p_output, int output_size) {
    Eigen::ThreadPool tp(std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());
    TestGraph computation;

    computation.set_thread_pool(&device);

    // 入力値のセット
    computation.set_arg0_data((void*)p_input);

    // 推論実行
    auto success = computation.Run();
    if(not success){
      return -1;
    }

    // 推論結果を取得
    std::copy(computation.result0_data(), computation.result0_data() + output_size/sizeof(float), p_output);
    return 0;
}


// 配列内の最大値となる添え字を返す
int max( float* array, int len ){
    int max_i = 0;
    for(int i=0; i<len; i++){
        if( array[i] > array[max_i] ){
            max_i = i;
        }
    }
    return max_i;
}



// Main関数
int main(){
    RawImage rawimg;        // ビットマップ画像
    float result[1][1000];  // 結果取得用

    // 24bit color 224x224のビットマップイメージを取得
    rawimg.loadImage("./cat224.bmp");

    // ResNet50の推論モデルに取得したビットマップ画像を入力し結果を取得する
    int ret = run( (const float*)rawimg.data, sizeof(rawimg.data), (float*) result, sizeof(result) );

    // 最大のスコアを持つIDを検索する(ResNet50の分類クラスのどれにヒットしたか確認)
    int max_i = max( &(result[0][0]), 1000 );

    // 結果出力
    std::cout << "result max_i is " << max_i << std::endl;
    return 0;
}

