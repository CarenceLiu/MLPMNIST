#include "config.h"
#include "dataLoader.h"
#include "logger.h"
#include <array>
#include <vector>
#include <cmath>
#include <memory>
#include <utility>
#include <random>
#include <cassert>
#include <string>

namespace std{
    class MLP{
        public:
            MLP();
            MLP(vector<int> layerSize_, double lr_);
            void initDataLoader(const string & data_filename, const string & label_filename, int trainValidationRatio = 10);
            void initLogger(const string log_filename, const string loss_filename);
            void random_init();
            void xavier_uniform_init();
            void xavier_normal_init();
            //for train 
            vector<double> single_forward(const array<unsigned char, 784> & input_data, unsigned char label);
            //for validation
            vector<double> single_forward(const array<unsigned char, 784> & input_data);

            //for train, but memory cost
            // vector<vector<double>> batch_forward(const vector<array<unsigned char, 784>> & input_data, vector<unsigned char> label);
            // vector<array<double, 10>> batch_forward(const vector<array<unsigned char, 784>> & input_data, int batch_size);
            double single_backward(const vector<double> & output, unsigned char label);
            double single_backward(const vector<double> & output, unsigned char label, vector<vector<vector<double>>> & weightMatrixUpdate, vector<vector<double>> & biasMatrixUpdate);
            double single_backward(const vector<double> & output, unsigned char label, vector<vector<vector<double>>> & weightMatrixUpdate, vector<vector<double>> & biasMatrixUpdate, int mode, double momentum_beta);
            double single_backwardNorm(const vector<double> & output, unsigned char label, int mode, double lambda);
            //memory cost 
            // void batch_backward(const vector<vector<double>> & output, vector<unsigned char> label);

            void basic_single_train(int epoch);
            void basic_batch_train(int epoch);
            void SGD_train(int epoch);
            void SGD_train(int epoch, int mode);
            void SGD_Momentum_train(int epoch);
            void SGD_Nesterov_train(int epoch);


            //validation return <loss, accuracy>
            pair<double, double> validation(const vector<array<unsigned char, 784>> & validation_data, const vector<unsigned char> & validation_label);

        private:
            void singleSoftMax(const vector<double> & output, vector<double> & result);
            double singleMSELoss(const vector<double> & forward_output, const vector<double> & ideal_output);
            double singleCrossEntropyLoss(const vector<double> & forward_output, const vector<double> & ideal_output);
            double singleCrossEntropyLossL1Norm(const vector<double> & forward_output, const vector<double> & ideal_output, double lambda);
            double singleCrossEntropyLossL2Norm(const vector<double> & forward_output, const vector<double> & ideal_output, double lambda);
            int layerNum;
            double lr;
            vector<int> layerSize;
            vector<array<int, 2>> matrixSize;
            vector<vector<vector<double>>> weightMatrix;
            vector<vector<double>> biasMatrix;

            //each layer's data, forward
            vector<vector<vector<double>>> forwardLayerData;

            //each layer's data, backward
            vector<vector<vector<double>>> backwardLayerDelta;
            shared_ptr<Logger> logger;
            shared_ptr<DataLoader> dataLoader;
    };

}