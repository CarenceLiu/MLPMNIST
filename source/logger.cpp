#include "logger.h"
#include <iostream>
#include <cassert>
using namespace std;

Logger::Logger(const string log_filename, const string loss_filename): 
    logFile(log_filename), lossResultFile(loss_filename) {
        assert(logFile.is_open());
        assert(lossResultFile.is_open());
    }

void Logger::log(int epoch, double train_loss, double validation_loss, double train_acc, double validation_acc) {
    logFile << "[Train Log] epoch " << epoch << ", train loss: " << train_loss << ", test loss: " << validation_loss <<", train accuracy: " << train_acc << ", test accuracy: " << validation_acc << endl; 
    cout << "[Train Log] epoch " << epoch << ", train loss: " << train_loss << ", test loss: " << validation_loss <<", train accuracy: " << train_acc << ", test accuracy: " << validation_acc << endl;
}

void Logger::log(const string & line) {
    logFile << line << endl;
    cout << line << endl;
}

void Logger::lossPrint(double loss) {
    lossResultFile << loss << endl; 
}