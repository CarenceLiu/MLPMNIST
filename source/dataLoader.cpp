#include "dataLoader.h"
#include <cassert>
#include <algorithm>
#include <random>
using namespace std;


uint32_t bigLitteEndian(const array<unsigned char, 4> & buf) {
    uint32_t res = 0;
    for(int i = 3, k = 1; i >= 0; --i, k*=256) {
        res += k*buf[i];
    }
    return res;
}


DataLoader::DataLoader(const string & data_filename, const string & label_filename, int trainValidationRatio_ = 10): data{},
 train_data{}, validation_data{}, trainValidationRatio(trainValidationRatio_) {
    ifstream data_file(data_filename, ios::binary);
    ifstream label_file(data_filename, ios::binary);
    assert(!data_file);
    assert(!label_file);
    array<unsigned char, 4> buf;
    array<unsigned char, 784> image_buf;
    unsigned char label_buf;

    //read data
    uint32_t type, items, row, col, labelNum;
    data_file.seekg(0, ios::beg);
    data_file.read(reinterpret_cast<char *>(&buf), 4);
    type = bigLitteEndian(buf);
    assert(type == 2501);
    data_file.read(reinterpret_cast<char *>(&buf), 4);
    items = bigLitteEndian(buf);
    data_file.read(reinterpret_cast<char *>(&buf), 4);
    row = bigLitteEndian(buf);
    data_file.read(reinterpret_cast<char *>(&buf), 4);
    col = bigLitteEndian(buf);
    assert(row == 28);
    assert(col == 28);
    
    //read label
    label_file.seekg(0, ios::beg);
    data_file.read(reinterpret_cast<char *>(&buf), 4);
    type = bigLitteEndian(buf);
    assert(type == 2049);
    data_file.read(reinterpret_cast<char *>(&buf), 4);
    labelNum = bigLitteEndian(buf);
    assert(labelNum == items);

    for(int i = 0; i < items; ++i) {
        data_file.read(reinterpret_cast<char *>(&image_buf), 784);
        label_file.read(reinterpret_cast<char *>(&label_buf), 1);
        data.push_back(make_pair(image_buf, label_buf));
    }


    dataShuffle();
}

void DataLoader::dataShuffle() {
    int valBound = data.size()/(1+trainValidationRatio);
    shuffle(data.begin(), data.end(), default_random_engine(random_device()()));
    train_data.clear();
    validation_data.clear();
    train_label.clear();
    validation_label.clear();

    for(int i = 0; i < valBound; ++i) {
        auto [image, label] = data[i];
        validation_data.push_back(move(image));
        validation_label.push_back(label);
    }
    for(int i = valBound; i < data.size(); ++i) {
        auto [image, label] = data[i];
        train_data.push_back(move(image));
        train_label.push_back(label);
    }
    trainDataIndex = 0;
}


int DataLoader::getNextDataIndex() {
    if(trainDataIndex < train_data.size()) {
        return trainDataIndex++;
    }
    else {
        return -1;
    }
}

