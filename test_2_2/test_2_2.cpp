#include "../source/mlp.h"

using namespace std;

int main() {
    vector<int> layer_1{784, 256, 10};
    double lr = 1e-2;
    int epoch = 500;
    string data_filename = "../dataset/train-images.idx3-ubyte";
    string label_filename = "../dataset/train-labels.idx1-ubyte";
    string log_filename_1 = "./trainLog_1.txt";
    string loss_filename_1 = "./loss_1.txt";
    string log_filename_2 = "./trainLog_2.txt";
    string loss_filename_2 = "./loss_2.txt";


    //SGD
    MLP mlp_1(layer_1, lr);
    mlp_1.initLogger(log_filename_1, loss_filename_1);
    mlp_1.initDataLoader(data_filename, label_filename, 1);
    mlp_1.SGD_train(epoch);

    //BGD
    MLP mlp_2(layer_1, lr);
    mlp_2.initLogger(log_filename_2, loss_filename_2);
    mlp_2.initDataLoader(data_filename, label_filename, 1);
    mlp_2.basic_batch_train(epoch);
    return 0;
}