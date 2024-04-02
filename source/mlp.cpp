#include "mlp.h"
using namespace std;

// default construct function: no hidden layer
MLP::MLP(): lr(1e-3), layerSize{784, 10}, matrixSize{array<int, 2>{10, 784}},
     weightMatrix{}, forwardLayerData{}, backwardLayerDelta{}, dataLoader(nullptr), logger(nullptr) {
        layerNum = layerSize.size();
        auto weight = vector<vector<double>>(784, vector<double>(10, 0));
        weightMatrix.push_back(move(weight));
        auto bias = vector<double>(10,0);
        biasMatrix.push_back(move(bias));
}

//input each layer's size
MLP::MLP(vector<int> layerSize_, double lr_): lr(lr_), layerSize(layerSize_), weightMatrix{}, forwardLayerData{},
 backwardLayerDelta{}, dataLoader(nullptr), logger(nullptr) {
    assert(layerSize.front() == 784);
    assert(layerSize.back() == 10);
    
    layerNum = layerSize.size();
    for(int i = 1; i < layerNum; ++i) {
        // cout << layerSize[i] << " " << layerSize[i-1] << endl;
        matrixSize.push_back(array<int, 2>{layerSize[i], layerSize[i-1]});
        auto weight = vector<vector<double>>(layerSize[i], vector<double>(layerSize[i-1], 0));
        weightMatrix.push_back(move(weight));
        auto bias = vector<double>(layerSize[i],0);
        biasMatrix.push_back(move(bias));
    }
}

void MLP::initDataLoader(const string & data_filename, const string & label_filename, int trainValidationRatio) {
    dataLoader = make_shared<DataLoader>(data_filename, label_filename, trainValidationRatio);
}

void MLP::initLogger(const string log_filename, const string loss_filename) {
    logger = make_shared<Logger>(log_filename, loss_filename);
}

// random init weight
void MLP::random_init() {
    random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);
 
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

#if DEBUG_MODE
    // for(int t = 0; t < matrixSize.size(); ++t) {
    //     auto [x, y] = matrixSize[t];
    //     for(int i = 0; i < x; ++i) {
    //         for(int j = 0; j < y; ++j) {
    //             cout <<weightMatrix[t][i][j] <<" ";
    //         }
    //         cout << endl;
    //     }
    //     for(int i = 0; i < x; ++i) {
    //         cout <<biasMatrix[t][i] <<" ";
    //     }
    //     cout <<endl << endl;
    // }
    // exit(0);
#endif
}

void MLP::xavier_uniform_init() {
    random_device rd;
    std::mt19937 gen(rd());

    for(int t = 0; t < matrixSize.size(); ++t) {
        auto [x, y] = matrixSize[t];
        double threshold = sqrt(6/(double)(x+y));
        std::uniform_real_distribution<> dis(-threshold, threshold);
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

void MLP::xavier_normal_init() {
    random_device rd;
    std::mt19937 gen(rd());

    for(int t = 0; t < matrixSize.size(); ++t) {
        auto [x, y] = matrixSize[t];
        double threshold = 2/(double)(x+y);
        std::normal_distribution<double> dis(0, threshold);
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

    double max_v = *max_element(layer_input.begin(), layer_input.end());
    for(auto & e: layer_input) {
        e -= max_v;
    }

    singleSoftMax(layer_input, output);
    forwardLayerData[0].push_back(output);

#if DEBUG_MODE
    // for(auto & v: forwardLayerData[0]) {
    //     cout<<v.size() <<" ";
    // }
    // cout <<endl;
    // for(auto & v: forwardLayerData[0]) {
    //     for(auto & e: v) {
    //         cout << e <<" ";
    //     }
    //     cout << endl;
    // }
#endif

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

    //avoid overflow
    double max_v = *max_element(layer_input.begin(), layer_input.end());
    for(auto & e: layer_input) {
        e -= max_v;
    }

    singleSoftMax(layer_input, output);
    return move(output);
}

// vector<vector<double>> MLP::batch_forward(const vector<array<unsigned char, 784>> & input_data, vector<unsigned char> label) {
//     int batch_size = input_data.size();
//     vector<vector<double>> output(batch_size, vector<double>(10,0));
//     int forwardT = matrixSize.size();

//     forwardLayerData = vector<vector<vector<double>>>(batch_size, vector<vector<double>>{});
//     backwardLayerDelta = vector<vector<vector<double>>>(batch_size, vector<vector<double>>{});

//     for(int m = 0; m < batch_size; ++m) {
//         for(int i = 0; i < layerNum; ++i) {
//             backwardLayerDelta[m].push_back(vector<double>(layerSize[i], 0));
//         }
//     }

//     vector<vector<double>> layer_input = vector<vector<double>>(batch_size, vector<double>(784, 0));
//     for(int m = 0; m < batch_size; ++m) {
//         for(int i = 0; i < 784; ++i) {
//             layer_input[m][i] = input_data[m][i];
//         }
//     }

//     for(int m = 0; m < batch_size; ++m) {
//         forwardLayerData[m].push_back(layer_input[m]); 
//         //for convenience
//         forwardLayerData[m].push_back(layer_input[m]);
//     }
//     //forward 
//     for(int t = 0; t < forwardT; ++t) {
//         auto output = vector<vector<double>>(batch_size, vector<double>(matrixSize[t][0], 0));
//         for(int m = 0; m < batch_size; ++m) {

//             // z = sigmoid(wx+b)
//             for(int i = 0; i < matrixSize[t][0]; ++i) {
//                 for(int j = 0; j < matrixSize[t][1]; ++j) {
//                     output[m][i] += weightMatrix[t][i][j]*layer_input[m][j];
//                 }
//                 output[m][i] += biasMatrix[t][i];
//             }
            
//             for(int m = 0; m < batch_size; ++m) {
//                 forwardLayerData[m].push_back(output[m]); 
//             }
            
//             //sigmoid
//             if(t != forwardT-1) {
//                 for(int m = 0; m < batch_size; ++m) {
//                     for(int i = 0; i < matrixSize[t][0]; ++i) {
//                         output[m][i] = 1/(1+exp(-output[m][i]));
//                     }
//                     forwardLayerData[m].push_back(output[m]);
//                 }
//             }
//         }
//         layer_input = move(output);
//     }
//     for(int m = 0; m < batch_size; ++m) {
//         double max_v = *max_element(layer_input[m].begin(), layer_input[m].end());
//         for(auto & e: layer_input[m]) {
//             e -= max_v;
//         }
//         singleSoftMax(layer_input[m], output[m]);
//         forwardLayerData[m].push_back(output[m]);
//     }
//     return move(output);
// }


double MLP::singleMSELoss(const vector<double> & forward_output, const vector<double> & ideal_output) {
    double loss = 0;
    
    for(int i = 0; i < 10; ++i) {
        loss += (ideal_output[i] - forward_output[i])*(ideal_output[i] - forward_output[i]);
    }

    loss *= 0.5;

    return loss;
}

double MLP::singleCrossEntropyLoss(const vector<double> & forward_output, const vector<double> & ideal_output) {
    double loss = 0;
    
    for(int i = 0; i < 10; ++i) {
        loss += -ideal_output[i] * log(forward_output[i]+1e-7);
    }


    return loss;
}

double MLP::singleCrossEntropyLossL1Norm(const vector<double> & forward_output, const vector<double> & ideal_output, double lambda) {
    double loss = 0;
    double total_w = 0;
    
    for(int i = 0; i < 10; ++i) {
        loss += -ideal_output[i] * log(forward_output[i]+1e-7);
    }

    for(auto &m:weightMatrix) {
        for(auto & line: m) {
            for(auto & w: line) {
                total_w += abs(w);
            }
        }
    }

    loss += lambda*total_w;

    return loss;
}

double MLP::singleCrossEntropyLossL2Norm(const vector<double> & forward_output, const vector<double> & ideal_output, double lambda) {
    double loss = 0;
    double total_w = 0;
    
    for(int i = 0; i < 10; ++i) {
        loss += -ideal_output[i] * log(forward_output[i]+1e-7);
    }

    for(auto &m:weightMatrix) {
        for(auto & line: m) {
            for(auto & w: line) {
                total_w += w*w;
            }
        }
    }

    loss += 0.5*lambda*total_w;

    return loss;
}

double MLP::single_backward(const vector<double> & output, unsigned char label) {
    vector<double> label_arr(10,0);
    label_arr[(size_t)label] = 1.0;

    double loss = singleCrossEntropyLoss(output, label_arr);
    // cout << loss<< endl;

    for(int i = 0; i < 10; ++i) {
        // softmax+MSE
        // backwardLayerDelta[0][layerNum-1][i] = (output[i] - label_arr[i])*forwardLayerData[0][2*layerNum-1][i]*(1-forwardLayerData[0][2*layerNum-1][i]);
        
        //softmax+crossEntropy
        backwardLayerDelta[0][layerNum-1][i] = output[i] - label_arr[i];
#if DEBUG_MODE
        cout<< layerNum-1 <<" " << output[i] <<" " <<label_arr[i]<< endl;
#endif
    }


    //calculate delta
    for(int t = layerNum-2; t >= 1; t--) {
        for(int i = 0; i < layerSize[t]; ++i) {
            for(int k = 0; k < layerSize[t+1]; ++k) {
                //delta j(l) += delta k(l+1)* w kj(l+1)*f'(z j(l))
                backwardLayerDelta[0][t][i] += backwardLayerDelta[0][t+1][k]*weightMatrix[t][k][i]*forwardLayerData[0][2*t+1][i]*(1-forwardLayerData[0][2*t+1][i]);
#if DEBUG_MODE
                // if(backwardLayerDelta[0][t+1][k]*weightMatrix[t][k][i]*forwardLayerData[0][2*t-1][i]*(1-forwardLayerData[0][2*t-1][i]) != 0) {
                //     cout<< t <<" " << backwardLayerDelta[0][t+1][k]<<" " << weightMatrix[t][k][i]<<" " <<forwardLayerData[0][2*t+1][i] << " " << forwardLayerData[0][2*t-1][i]*(1-forwardLayerData[0][2*t-1][i])<< endl;   
                // }
#endif
            }
        }
    }

#if DEBUG_MODE

    // for(auto & b: backwardLayerDelta[0]) {
    //     if(b.size() != 784) {
    //         for(auto & e: b) {
    //             cout << e <<" ";
    //         }
    //         cout <<endl;
    //     }
    // }

    // cout <<endl<<endl<<endl<<endl;

#endif

    // update w, b 
    for(int t = 1; t < layerNum; ++t) {
        for(int i = 0; i < matrixSize[t-1][0]; ++i) {
            for(int j = 0; j < matrixSize[t-1][1]; ++j) {
#if DEBUG_MODE
            // if(backwardLayerDelta[0][t][i] != 0 && forwardLayerData[0][2*t-1][j] != 0 && t-1 == 0) {
            //     cout<< t-1 <<" " << weightMatrix[t-1][i][j] <<" " <<backwardLayerDelta[0][t][i] << " " << forwardLayerData[0][2*t-1][j]  << " " << lr*backwardLayerDelta[0][t][i]*forwardLayerData[0][2*t-1][j] << endl;
            // }
#endif

                weightMatrix[t-1][i][j] -= lr*backwardLayerDelta[0][t][i]*forwardLayerData[0][2*t-1][j];
            }
        }
        for(int i = 0; i < matrixSize[t-1][0]; ++i) {
            biasMatrix[t-1][i] -= lr*backwardLayerDelta[0][t][i];
        }
    }

    return loss;
}

double MLP::single_backward(const vector<double> & output, unsigned char label, vector<vector<vector<double>>> & weightMatrixUpdate, vector<vector<double>> & biasMatrixUpdate) {
    vector<double> label_arr(10,0);
    label_arr[(size_t)label] = 1.0;

    double loss = singleCrossEntropyLoss(output, label_arr);

    for(int i = 0; i < 10; ++i) {
        backwardLayerDelta[0][layerNum-1][i] = output[i] - label_arr[i];
    }

    //calculate delta
    for(int t = layerNum-2; t >= 1; t--) {
        for(int i = 0; i < layerSize[t]; ++i) {
            for(int k = 0; k < layerSize[t+1]; ++k) {
                //delta j(l) += delta k(l+1)* w kj(l+1)*f'(z j(l))
                backwardLayerDelta[0][t][i] += backwardLayerDelta[0][t+1][k]*weightMatrix[t][k][i]*forwardLayerData[0][2*t+1][i]*(1-forwardLayerData[0][2*t+1][i]);
            }
        }
    }

    // update w, b 
    for(int t = 1; t < layerNum; ++t) {
        for(int i = 0; i < matrixSize[t-1][0]; ++i) {
            for(int j = 0; j < matrixSize[t-1][1]; ++j) {
                weightMatrixUpdate[t-1][i][j] -= lr*backwardLayerDelta[0][t][i]*forwardLayerData[0][2*t-1][j];
            }
        }
        for(int i = 0; i < matrixSize[t-1][0]; ++i) {
            biasMatrixUpdate[t-1][i] -= lr*backwardLayerDelta[0][t][i];
        }
    }

    return loss;
}

double MLP::single_backward(const vector<double> & output, unsigned char label, vector<vector<vector<double>>> & weightMatrixUpdate, vector<vector<double>> & biasMatrixUpdate, int mode, double momentum_beta) {
    vector<double> label_arr(10,0);
    label_arr[(size_t)label] = 1.0;

    double loss = singleCrossEntropyLoss(output, label_arr);

    for(int i = 0; i < 10; ++i) {
        backwardLayerDelta[0][layerNum-1][i] = output[i] - label_arr[i];
    }

    //calculate delta
    for(int t = layerNum-2; t >= 1; t--) {
        for(int i = 0; i < layerSize[t]; ++i) {
            for(int k = 0; k < layerSize[t+1]; ++k) {
                //delta j(l) += delta k(l+1)* w kj(l+1)*f'(z j(l))
                backwardLayerDelta[0][t][i] += backwardLayerDelta[0][t+1][k]*weightMatrix[t][k][i]*forwardLayerData[0][2*t+1][i]*(1-forwardLayerData[0][2*t+1][i]);
            }
        }
    }

    // update w, b 
    for(int t = 1; t < layerNum; ++t) {
        for(int i = 0; i < matrixSize[t-1][0]; ++i) {
            for(int j = 0; j < matrixSize[t-1][1]; ++j) {
                if(mode == 1) {
                    weightMatrixUpdate[t-1][i][j] *= momentum_beta;
                    weightMatrixUpdate[t-1][i][j] += lr*backwardLayerDelta[0][t][i]*forwardLayerData[0][2*t-1][j];
                }
                else if(mode == 2) {
                    weightMatrixUpdate[t-1][i][j] *= momentum_beta;
                    weightMatrixUpdate[t-1][i][j] += lr*(backwardLayerDelta[0][t][i]*forwardLayerData[0][2*t-1][j]+momentum_beta*weightMatrixUpdate[t-1][i][j]);
                }
                weightMatrix[t-1][i][j] -= weightMatrixUpdate[t-1][i][j];
            }
        }
        for(int i = 0; i < matrixSize[t-1][0]; ++i) {
            if(mode == 1) {
                biasMatrixUpdate[t-1][i] *= momentum_beta;
                biasMatrixUpdate[t-1][i] += lr*backwardLayerDelta[0][t][i];
            }
            else if(mode == 2) {
                biasMatrixUpdate[t-1][i] *= momentum_beta;
                biasMatrixUpdate[t-1][i] += lr*(backwardLayerDelta[0][t][i]+momentum_beta*biasMatrixUpdate[t-1][i]);
            }
            biasMatrix[t-1][i] -= biasMatrixUpdate[t-1][i];
        }
    }

    return loss;
}


double MLP::single_backwardNorm(const vector<double> & output, unsigned char label, int mode, double lambda) {
    vector<double> label_arr(10,0);
    label_arr[(size_t)label] = 1.0;

    double loss = 0;
    // if(mode == 1) {
    //     loss = singleCrossEntropyLossL1Norm(output, label_arr, lambda);
    // }
    // else if(mode == 2) {
    //     loss = singleCrossEntropyLossL1Norm(output, label_arr, lambda);
    // }
    // else{
        loss = singleCrossEntropyLoss(output, label_arr);
    // }

    for(int i = 0; i < 10; ++i) {
        backwardLayerDelta[0][layerNum-1][i] = output[i] - label_arr[i];
    }


    //calculate delta
    for(int t = layerNum-2; t >= 1; t--) {
        for(int i = 0; i < layerSize[t]; ++i) {
            for(int k = 0; k < layerSize[t+1]; ++k) {
                //delta j(l) += delta k(l+1)* w kj(l+1)*f'(z j(l))
                backwardLayerDelta[0][t][i] += backwardLayerDelta[0][t+1][k]*weightMatrix[t][k][i]*forwardLayerData[0][2*t+1][i]*(1-forwardLayerData[0][2*t+1][i]);
            }
        }
    }

    // update w, b 
    for(int t = 1; t < layerNum; ++t) {
        for(int i = 0; i < matrixSize[t-1][0]; ++i) {
            for(int j = 0; j < matrixSize[t-1][1]; ++j) {
                if(mode == 1) {     //L1 normalization
                    if(weightMatrix[t-1][i][j] >= 0) {
                        weightMatrix[t-1][i][j] -= lr*lambda;
                    }
                    else {
                        weightMatrix[t-1][i][j] += lr*lambda;
                    }
                }
                else if(mode == 2) {    //l2 normalization
                    weightMatrix[t-1][i][j] -= lr*lambda*weightMatrix[t-1][i][j];
                }
                weightMatrix[t-1][i][j] -= lr*backwardLayerDelta[0][t][i]*forwardLayerData[0][2*t-1][j];
            }
        }
        for(int i = 0; i < matrixSize[t-1][0]; ++i) {
            biasMatrix[t-1][i] -= lr*backwardLayerDelta[0][t][i];
        }
    }

    return loss;
}

// void MLP::batch_backward(const vector<vector<double>> & output, vector<unsigned char> label) {
//     int batch_size = output.size();
//     vector<vector<double>> label_arr(batch_size, vector<double>(10,0));
//     vector<double> loss(batch_size, 0);


//     for(int m = 0; m < batch_size; ++m) {
//         label_arr[m][(size_t)label[m]] = 1.0;
//         loss[m] = singleCrossEntropyLoss(output[m], label_arr[m]);
//     }

//     for(int m = 0; m < batch_size; ++m) {
//         for(int i = 0; i < 10; ++i) {
//             //softmax+crossEntropy
//             backwardLayerDelta[m][layerNum-1][i] = output[m][i] - label_arr[m][i];
//         }
//     }

//     //calculate delta
//     for(int m = 0; m < batch_size; ++m) {
//         for(int t = layerNum-2; t >= 1; t--) {
//             for(int i = 0; i < layerSize[t]; ++i) {
//                 for(int k = 0; k < layerSize[t+1]; ++k) {
//                     //delta j(l) += delta k(l+1)* w kj(l+1)*f'(z j(l))
//                     backwardLayerDelta[m][t][i] += backwardLayerDelta[m][t+1][k]*weightMatrix[t][k][i]*forwardLayerData[m][2*t-1][i]*(1-forwardLayerData[m][2*t-1][i]);
//                 }
//             }
//         }
//     }

//     //BGD update w, b 
//     for(int t = 1; t < layerNum; ++t) {
//         for(int i = 0; i < matrixSize[t-1][0]; ++i) {
//             for(int j = 0; j < matrixSize[t-1][1]; ++j) {
//                 double total_w_update = 0;
//                 for(int m = 0; m < batch_size; ++m) {
//                     total_w_update += lr*backwardLayerDelta[m][t][i]*forwardLayerData[m][2*t-1][j];
//                 }
//                 weightMatrix[t-1][i][j] -= total_w_update/batch_size;
//             }
//         }
//         for(int i = 0; i < matrixSize[t-1][0]; ++i) {
//             double total_w_update = 0;
//             for(int m = 0; m < batch_size; ++m) {
//                 total_w_update += lr*backwardLayerDelta[m][t][i];
//             }
//             biasMatrix[t-1][i] -= total_w_update/batch_size;
//         }
//     }
// }


pair<double, double> MLP::validation(const vector<array<unsigned char, 784>> & validation_data, const vector<unsigned char> & validation_label) {
    double total_loss = 0;
    int sz = validation_data.size();
    double hit = 0.0;
    for(int i = 0; i < sz; ++i) {
        vector<double> label_arr(10,0);
        label_arr[(size_t)validation_label[i]] = 1.0;
        auto output = single_forward(validation_data[i]);
        auto max_it = max_element(output.begin(), output.end());
        int index = distance(output.begin(), max_it);
        // cout << index <<" " << (int)validation_label[i]<<endl;
        if(index == (int)validation_label[i]) {
            // cout << "hit !"<<endl;
            hit += 1.0;
        }
        double loss = singleCrossEntropyLoss(output, label_arr);
        total_loss += loss;

#if DEBUG_MODE
        // for(auto e: output) {
        //     cout << e <<" ";
        // }
        // cout <<endl;
        // for(auto e: label_arr) {
        //     cout << e<< " ";
        // }
        // cout << endl;
        // cout <<"Loss: " << loss<<endl;
#endif

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
    string info = "Basic SGD Train: random init w, epoch: " + to_string(epoch) + ", learning rate: " + to_string(lr) + ", layer size: ";
    for(auto layer:layerSize) {
        info += to_string(layer);
        info += " ";
    }
    logger->log(info);

    //begin train 
    for(int t = 0; t < epoch; ++t) {
        dataLoader->dataShuffle();
        int index = -1;
        double train_hit = 0;
        double total_loss = 0;
        // int s = 0;

        //SGD
        while((index = dataLoader->getNextDataIndex()) != -1) {
            // index = index % 4;
            // cout << index <<endl;
            auto output = single_forward(dataLoader->train_data[index], dataLoader->train_label[index]);
            auto max_it = max_element(output.begin(), output.end());
            int predict_index = distance(output.begin(), max_it);
            if(predict_index == (int)dataLoader->train_label[index]) {
                train_hit += 1.0;
            }
#if DEBUG_MODE
            // s++;
            // if(s > 10) {
            //     exit(0);
            // }
            // cout << predict_index << " " << (int)dataLoader->train_label[index] << endl;
            // for(auto & e: output) {
            //     cout << e<< " ";
            // }
            // cout <<endl;
            // // cout<<(int)dataLoader->train_label[index]<<endl;
            // if(index > 12) {
            //     exit(0);
            // }
#endif

            total_loss += single_backward(output, dataLoader->train_label[index]);

#if DEBUG_MODE
            // for(auto & v: weightMatrix) {
            //     for(auto & e: v[0]) {
            //         cout << e<< ",";
            //     }
            //     cout << endl;
            //     cout <<endl << endl <<endl;
            // }
            if(index > 2) {
                exit(0);
            }
#endif
        }

        //validation 
        auto [loss, accuracy] = validation(dataLoader->validation_data, dataLoader->validation_label);

        // cout << train_hit<<endl;
        // logger->log(t, loss, train_hit/(double)dataLoader->train_data.size(), accuracy);
        logger->log(t, total_loss/(double)dataLoader->train_data.size(), loss, train_hit/(double)dataLoader->train_data.size(), accuracy);
        logger->lossPrint(total_loss/(double)dataLoader->train_data.size());

    }


}

void MLP::basic_batch_train(int epoch) {
    int t = 0;
    assert(logger != nullptr);
    assert(dataLoader != nullptr);

    //pre init
    random_init();
    string info = "Basic BGD Train: random init w, epoch: " + to_string(epoch) + ", learning rate: " + to_string(lr) + ", layer size: ";
    for(auto layer:layerSize) {
        info += to_string(layer);
        info += " ";
    }
    logger->log(info); 

    for(int t = 0; t < epoch; ++t) {
        dataLoader->dataShuffle();
        int index = -1;
        double train_hit = 0;
        int batch_size = dataLoader->train_data.size();
        vector<vector<vector<double>>> weightMatrixUpdate{};
        vector<vector<double>> biasMatrixUpdate{};
        for(int i = 1; i < layerNum; ++i) {
            // cout << layerSize[i] << " " << layerSize[i-1] << endl;
            auto weight = vector<vector<double>>(layerSize[i], vector<double>(layerSize[i-1], 0));
            weightMatrixUpdate.push_back(move(weight));
            auto bias = vector<double>(layerSize[i],0);
            biasMatrixUpdate.push_back(move(bias));
        }

        //BGD
        double total_loss = 0;
        while((index = dataLoader->getNextDataIndex()) != -1) {

            auto output = single_forward(dataLoader->train_data[index], dataLoader->train_label[index]);
            auto max_it = max_element(output.begin(), output.end());
            int predict_index = distance(output.begin(), max_it);
            if(predict_index == (int)dataLoader->train_label[index]) {
                train_hit += 1.0;
            }

            total_loss += single_backward(output, dataLoader->train_label[index], weightMatrixUpdate, biasMatrixUpdate);
        }

        //BGD update
        for(int i = 0; i < weightMatrix.size(); ++i){
            for(int j = 0; j < weightMatrix[i].size(); ++j) {
                for(int k = 0; k < weightMatrix[i][j].size(); ++k) {
                    weightMatrix[i][j][k] += weightMatrixUpdate[i][j][k]/(double)batch_size;
                }
            }
        }
        for(int i = 0; i < biasMatrix.size(); ++i) {
            for(int j = 0; j < biasMatrix[i].size(); ++j) {
                biasMatrix[i][j] += biasMatrixUpdate[i][j]/(double)batch_size;
            }
        }

        auto [loss, accuracy] = validation(dataLoader->validation_data, dataLoader->validation_label);

        logger->log(t, total_loss/(double)batch_size, loss, train_hit/(double)batch_size, accuracy);
        logger->lossPrint(total_loss/(double)batch_size);
    }
}


void MLP::SGD_train(int epoch) {
    int t = 0;
    assert(logger != nullptr);
    assert(dataLoader != nullptr);

    //pre init
    random_init();
    string info = "Basic SGD Train: random init w, epoch: " + to_string(epoch) + ", learning rate: " + to_string(lr) + ", layer size: ";
    for(auto layer:layerSize) {
        info += to_string(layer);
        info += " ";
    }
    logger->log(info);

    //begin train 
    for(int t = 0; t < epoch; ++t) {
        dataLoader->dataShuffle();
        double train_hit = 0;
        // int s = 0;

        //SGD;
        auto output = single_forward(dataLoader->train_data[0], dataLoader->train_label[0]);
        auto max_it = max_element(output.begin(), output.end());
        int predict_index = distance(output.begin(), max_it);
        if(predict_index == (int)dataLoader->train_label[0]) {
            train_hit += 1.0;
        }
        double train_loss = single_backward(output, dataLoader->train_label[0]);

        //validation 
        // auto [loss, accuracy] = validation(dataLoader->validation_data, dataLoader->validation_label);

        // cout << train_hit<<endl;
        // logger->log(t, loss, train_hit/(double)dataLoader->train_data.size(), accuracy);
        logger->log(t, train_loss, 0, 0, 0);
        logger->lossPrint(train_loss);

    }


}

void MLP::SGD_Momentum_train(int epoch) {
    int t = 0;
    assert(logger != nullptr);
    assert(dataLoader != nullptr);

    vector<vector<vector<double>>> weightMatrixUpdate{};
    vector<vector<double>> biasMatrixUpdate{};
    for(int i = 1; i < layerNum; ++i) {
        // cout << layerSize[i] << " " << layerSize[i-1] << endl;
        auto weight = vector<vector<double>>(layerSize[i], vector<double>(layerSize[i-1], 0));
        weightMatrixUpdate.push_back(move(weight));
        auto bias = vector<double>(layerSize[i],0);
        biasMatrixUpdate.push_back(move(bias));
    }

    //pre init
    random_init();
    string info = "Momentum SGD Train: random init w, epoch: " + to_string(epoch) + ", learning rate: " + to_string(lr) + ", layer size: ";
    for(auto layer:layerSize) {
        info += to_string(layer);
        info += " ";
    }
    logger->log(info);

    //begin train 
    for(int t = 0; t < epoch; ++t) {
        dataLoader->dataShuffle();
        double train_hit = 0;
        // int s = 0;

        //SGD;
        auto output = single_forward(dataLoader->train_data[0], dataLoader->train_label[0]);
        auto max_it = max_element(output.begin(), output.end());
        int predict_index = distance(output.begin(), max_it);
        if(predict_index == (int)dataLoader->train_label[0]) {
            train_hit += 1.0;
        }
        double train_loss = single_backward(output, dataLoader->train_label[0], weightMatrixUpdate, biasMatrixUpdate, 1, 0.9);

        //validation 
        // auto [loss, accuracy] = validation(dataLoader->validation_data, dataLoader->validation_label);

        // cout << train_hit<<endl;
        // logger->log(t, loss, train_hit/(double)dataLoader->train_data.size(), accuracy);
        logger->log(t, train_loss, 0, 0, 0);
        logger->lossPrint(train_loss);

    }


}

void MLP::SGD_Nesterov_train(int epoch) {
    int t = 0;
    assert(logger != nullptr);
    assert(dataLoader != nullptr);

    vector<vector<vector<double>>> weightMatrixUpdate{};
    vector<vector<double>> biasMatrixUpdate{};
    for(int i = 1; i < layerNum; ++i) {
        // cout << layerSize[i] << " " << layerSize[i-1] << endl;
        auto weight = vector<vector<double>>(layerSize[i], vector<double>(layerSize[i-1], 0));
        weightMatrixUpdate.push_back(move(weight));
        auto bias = vector<double>(layerSize[i],0);
        biasMatrixUpdate.push_back(move(bias));
    }

    //pre init
    random_init();
    string info = "Nesterov SGD Train: random init w, epoch: " + to_string(epoch) + ", learning rate: " + to_string(lr) + ", layer size: ";
    for(auto layer:layerSize) {
        info += to_string(layer);
        info += " ";
    }
    logger->log(info);

    //begin train 
    for(int t = 0; t < epoch; ++t) {
        dataLoader->dataShuffle();
        double train_hit = 0;
        // int s = 0;

        //SGD;
        auto output = single_forward(dataLoader->train_data[0], dataLoader->train_label[0]);
        auto max_it = max_element(output.begin(), output.end());
        int predict_index = distance(output.begin(), max_it);
        if(predict_index == (int)dataLoader->train_label[0]) {
            train_hit += 1.0;
        }
        double train_loss = single_backward(output, dataLoader->train_label[0], weightMatrixUpdate, biasMatrixUpdate, 2, 0.9);

        //validation 
        // auto [loss, accuracy] = validation(dataLoader->validation_data, dataLoader->validation_label);

        // cout << train_hit<<endl;
        // logger->log(t, loss, train_hit/(double)dataLoader->train_data.size(), accuracy);
        logger->log(t, train_loss, 0, 0, 0);
        logger->lossPrint(train_loss);

    }


}



void MLP::SGD_train(int epoch, int mode) {
    int t = 0;
    assert(logger != nullptr);
    assert(dataLoader != nullptr);

    //pre init
    if(mode == 1) {
        xavier_uniform_init();
    }
    else if(mode == 2) {
        xavier_normal_init();
    }
    else {
        random_init();
    }
    string info = "Basic SGD Train: random init w, epoch: " + to_string(epoch) + ", learning rate: " + to_string(lr) + ", layer size: ";
    for(auto layer:layerSize) {
        info += to_string(layer);
        info += " ";
    }
    logger->log(info);

    //begin train 
    for(int t = 0; t < epoch; ++t) {
        dataLoader->dataShuffle();
        double train_hit = 0;
        // int s = 0;

        //SGD;
        auto output = single_forward(dataLoader->train_data[0], dataLoader->train_label[0]);
        auto max_it = max_element(output.begin(), output.end());
        int predict_index = distance(output.begin(), max_it);
        if(predict_index == (int)dataLoader->train_label[0]) {
            train_hit += 1.0;
        }

        double train_loss = 0;
        if(mode == 3) {     //L1 normalization
            train_loss = single_backwardNorm(output, dataLoader->train_label[0], 1, 0.1);
        }
        else if(mode == 4) {    //l2 normalization
            train_loss = single_backwardNorm(output, dataLoader->train_label[0], 2, 0.1);
        }
        else {
            train_loss = single_backward(output, dataLoader->train_label[0]);
        }
        //validation 
        // auto [loss, accuracy] = validation(dataLoader->validation_data, dataLoader->validation_label);

        // cout << train_hit<<endl;
        // logger->log(t, loss, train_hit/(double)dataLoader->train_data.size(), accuracy);
        logger->log(t, train_loss, 0, 0, 0);
        logger->lossPrint(train_loss);

    }

    auto [loss, accuracy] = validation(dataLoader->validation_data, dataLoader->validation_label);

    logger->log(t, 0, loss, 0, accuracy);
}