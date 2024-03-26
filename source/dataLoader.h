#include "config.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

namespace std{
    class DataLoader{
        public:
            DataLoader(const string & data_filename, const string & label_filename, int trainValidationRatio = 10);
            void dataShuffle();
            int getNextDataIndex();
            vector<pair<array<unsigned char, 784>, unsigned char>> data;
            vector<array<unsigned char, 784>> train_data;
            vector<unsigned char> train_label;
            vector<array<unsigned char, 784>> validation_data;
            vector<unsigned char> validation_label;
        
        private:
            int trainValidationRatio;
            int trainDataIndex;
    };
}