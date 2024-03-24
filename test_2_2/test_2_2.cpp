#include "../source/mlp.h"

using namespace std;

int main() {
    vector<int> layer{784, 256, 10};
    double lr = 1e-1;
    int epoch = 50;
    string data_filename = "../dataset/train-images.idx3-ubyte";
    string label_filename = "../dataset/train-labels.idx1-ubyte";
    string log_filename = "./trainLog.txt";
    string loss_filename = "./loss.txt";


    
    return 0;
}