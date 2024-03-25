#include "../source/mlp.h"

using namespace std;

int main() {
    vector<int> layer{784, 512, 128, 10};
    double lr = 1e-1;
    int epoch = 50;
    string data_filename = "./dataset/train-images.idx3-ubyte";
    string label_filename = "./dataset/train-labels.idx1-ubyte";
    string log_filename = "./trainLog.txt";
    string loss_filename = "./loss.txt";


    MLP mlp(layer, lr);
    mlp.initLogger(log_filename, loss_filename);
    mlp.initDataLoader(data_filename, label_filename, 1);
    mlp.basic_batch_train(50);
    return 0;
}