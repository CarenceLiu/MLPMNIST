#include "mlp.h"
using namespace std;


// default construct function: no hidden layer
MLP::MLP(): lr(1e-3), layerSize{784, 10}, matrixSize{array<int, 2>{10, 784}},
     weightMatrix{}, forwardLayerData{}, backwardLayerDelta{}, dataLoader(nullptr), logger(nullptr) {
        layerNum = layerSize.size();
        auto weight = vector<vector<double>>(748, vector<double>(10, 0));
        weightMatrix.push_back(move(weight));
        auto bias = vector<double>(10,0);
        biasMatrix.push_back(move(bias));
}

//input each layer's size
MLP::MLP(vector<int> layerSize_, double lr_): lr(lr_), layerSize(layerSize_), weightMatrix{}, forwardLayerData{},
 backwardLayerDelta{}, dataLoader(nullptr), logger(nullptr) {
    assert(layerSize.front() == 784);
    assert(layerSize.back() == 10);
    
    int layerNum = layerSize.size();
    for(int i = 1; i < layerNum; ++i) {
        matrixSize.push_back(array<int, 2>{layerSize[i], layerSize[i-1]});
        auto weight = vector<vector<double>>(layerSize[i], vector<double>(layerSize[i-1], 0));
        weightMatrix.push_back(move(weight));
        auto bias = vector<double>(layerSize[i],0);
        biasMatrix.push_back(move(bias));
    }
}

void MLP::initDataLoader(const string & data_filename, const string & label_filename, int trainValidationRatio = 10) {
    dataLoader = make_shared<DataLoader>(new DataLoader(data_filename, label_filename, trainValidationRatio));
}

void MLP::initLogger(const string log_filename, const string loss_filename) {
    logger = make_shared<Logger>(new Logger(log_filename, loss_filename));
}

// random init weight
void MLP::random_init() {
    random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
 
    // 生成并输出一个随机数
    double random_double = dis(gen);
    for(int t = 0; t < matrixSize.size(); ++t) {
        auto [x, y] = matrixSize[t];
        for(int i = 0; i < x; ++i) {
            for(int j = 0; j < y; ++j) {
                weightMatrix[t][i][j] = dis(gen);
                if(weightMatrix[t][i][j] == 0) {
                    weightMatrix[t][i][j] = 1e-5;
                }
            }
        }
        for(int i = 0; i < x; ++i) {
            biasMatrix[t][i] = dis(gen);
            if(biasMatrix[t][i] == 0) {
                biasMatrix[t][i] = 1e-5;
            }
        }
    }
}

void MLP::singleSoftMax(const vector<double> & output, vector<double> & result) {
    double sum = 0;
    for(int i = 0; i < 10; ++i) {
        sum += exp(output[i]);
    }
    for(int i = 0; i < 10; ++i) {
        result[i] = exp(output[i])/sum;
    }
    return;
}

vector<double> MLP::single_forward(const array<unsigned char, 784> & input_data, unsigned char label) {
    vector<double> output(10,0);
    int forwardT = matrixSize.size();

    //init forwardLayerData and delta
    forwardLayerData = vector<vector<vector<double>>>{vector<vector<double>>{}};
    backwardLayerDelta = vector<vector<vector<double>>>{vector<vector<double>>{}};
    for(int i = 0; i < layerNum; ++i) {
        backwardLayerDelta[0].push_back(vector<double>(layerSize[i], 0));
    }
    
    vector<double> layer_input = vector<double>(784, 0);
    for(int i = 0; i < 784; ++i) {
        layer_input[i] = input_data[i];
    }

    forwardLayerData[0].push_back(layer_input); 
    //for convenience
    forwardLayerData[0].push_back(layer_input);

    //forward 
    for(int t = 0; t < forwardT; ++t) {
        auto output = vector<double>(matrixSize[t][0], 0);

        // z = sigmoid(wx+b)
        for(int i = 0; i < matrixSize[t][0]; ++i) {
            for(int j = 0; j < matrixSize[t][1]; ++j) {
                output[i] += weightMatrix[t][i][j]*layer_input[j];
            }
            output[i] += biasMatrix[t][i];
        }
        
        forwardLayerData[0].push_back(output);
        

        //sigmoid
        if(t != forwardT-1) {
            for(int i = 0; i < matrixSize[t][0]; ++i) {
                output[i] = 1/(1+exp(-output[i]));
            }
            forwardLayerData[0].push_back(output);
        }

        layer_input = move(output);
    }

    singleSoftMax(layer_input, output);
    forwardLayerData[0].push_back(output);

    return move(output);
}

//for validation, not for train
vector<double> MLP::single_forward(const array<unsigned char, 784> & input_data) {
    vector<double> output(10,0);
    int forwardT = matrixSize.size();
    
    vector<double> layer_input = vector<double>(784, 0);
    for(int i = 0; i < 784; ++i) {
        layer_input[i] = input_data[i];
    }
    //forward 
    for(int t = 0; t < forwardT; ++t) {
        auto output = vector<double>(matrixSize[t][0], 0);

        // z = sigmoid(wx+b)
        for(int i = 0; i < matrixSize[t][0]; ++i) {
            for(int j = 0; j < matrixSize[t][1]; ++j) {
                output[i] += weightMatrix[t][i][j]*layer_input[j];
            }
            output[i] += biasMatrix[t][i];
        }

        //sigmoid
        if(t != forwardT-1) {
            for(int i = 0; i < matrixSize[t][0]; ++i) {
                output[i] = 1/(1+exp(-output[i]));
            }
        }

        layer_input = move(output);
    }

    singleSoftMax(layer_input, output);
    return move(output);
}


double MLP::singleLoss(const vector<double> & forward_output, const vector<double> & ideal_output) {
    double loss = 0;
    
    for(int i = 0; i < 10; ++i) {
        loss += (ideal_output[i] - forward_output[i])*(ideal_output[i] - forward_output[i]);
    }

    loss *= 0.5;

    return loss;
}


void MLP::single_backward(const vector<double> & output, unsigned char label) {
    vector<double> label_arr(10,0);
    label_arr[(size_t)label] = 1.0;

    double loss = singleLoss(output, label_arr);

    for(int i = 0; i < 10; ++i) {
        backwardLayerDelta[0][layerNum-1][i] = (output[i] - label_arr[i])*forwardLayerData[0][2*layerNum-2][i]*(1-forwardLayerData[0][2*layerNum-2][i]);
    }


    //calculate delta
    for(int t = layerNum-2; t >= 1; t--) {
        for(int i = 0; i < layerSize[t]; ++i) {
            for(int k = 0; k < layerSize[t+1]; ++k) {
                //delta j(l) += delta k(l+1)* w kj(l+1)*f'(z j(l))
                backwardLayerDelta[0][t][i] += backwardLayerDelta[0][t+1][k]*weightMatrix[t][k][i]*forwardLayerData[0][2*t-2][i]*(1-forwardLayerData[0][2*t-2][i]);
            }
        }
    }


    // update w, b 
    for(int t = 1; t < layerNum; ++t) {
        for(int i = 0; i < matrixSize[t-1][0]; ++i) {
            for(int j = 0; j < matrixSize[t-1][1]; ++j) {
                weightMatrix[t-1][i][j] -= lr*backwardLayerDelta[0][t][i]*forwardLayerData[0][2*t-2][j];
            }
        }
        for(int i = 0; i < matrixSize[t-1][0]; ++i) {
            biasMatrix[t-1][i] -= lr*backwardLayerDelta[0][t][i];
        }
    }
}

pair<double, double> MLP::validation(const vector<array<unsigned char, 784>> & validation_data, const vector<unsigned char> & validation_label) {
    double total_loss = 0;
    int sz = validation_data.size();
    double hit = 0.0;
    for(int i = 0; i < sz; ++i) {
        vector<double> label_arr(10,0);
        label_arr[(size_t)validation_label[i]] = 1.0;
        auto output = single_forward(validation_data[i]);
        auto max_it = max_element(validation_data[i].begin(), validation_data[i].end());
        auto index = distance(validation_data[i].begin(), max_it);
        if((int)index == (int)validation_label[i]) {
            hit += 1.0;
        }
        total_loss += singleLoss(output, label_arr);
    }
    
    double average_loss = total_loss/(double)sz;
    double accuracy = hit/(double)sz;

    return make_pair(average_loss, accuracy);
}

void MLP::basic_single_train(int epoch) {
    int t = 0;
    assert(logger != nullptr);
    assert(dataLoader != nullptr);

    //pre init
    random_init();
    string info = "Basic Train: random init w, epoch: " + to_string(epoch) + ", learning rate: " + to_string(lr) + ", layer size: ";
    for(auto layer:layerSize) {
        info += to_string(layer);
        info += " ";
    }
    logger->log(info);

    //begin train 
    for(int t = 0; t < epoch; ++t) {
        dataLoader->dataShuffle();
        int index = -1;

        //SGD
        while((index = dataLoader->getNextDataIndex()) != -1) {
            auto output = single_forward(dataLoader->train_data[index], dataLoader->train_label[index]);
            single_backward(output, dataLoader->train_label[index]);
        }

        //validation 
        auto [loss, accuracy] = validation(dataLoader->validation_data, dataLoader->validation_label);

        logger->log(t, loss, accuracy);
        logger->lossPrint(loss);
    }


}

