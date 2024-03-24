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
            //for train 
            vector<double> single_forward(const array<unsigned char, 784> & input_data, unsigned char label);
            //for validation
            vector<double> single_forward(const array<unsigned char, 784> & input_data);
            // vector<array<double, 10>> batch_forward(const vector<array<unsigned char, 784>> & input_data, int batch_size);
            void single_backward(const vector<double> & output, unsigned char label);

            void basic_single_train(int epoch);

            //validation return <loss, accuracy>
            pair<double, double> validation(const vector<array<unsigned char, 784>> & validation_data, const vector<unsigned char> & validation_label);

        private:
            void singleSoftMax(const vector<double> & output, vector<double> & result);
            double MLP::singleLoss(const vector<double> & forward_output, const vector<double> & ideal_output);

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