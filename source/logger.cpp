#include "logger.h"
#include <iostream>
using namespace std;

Logger::Logger(const string log_filename, const string loss_filename): 
    logFile(log_filename), lossResultFile(loss_filename) {}

void Logger::log(int epoch, double loss, double acc) {
    logFile << "[Train Log] epoch " << epoch << ", loss " << loss <<", prediction accuracy: " << acc << endl; 
    cout << "[Train Log] epoch " << epoch << ", loss " << loss <<", prediction accuracy: " << acc << endl; 
}

void Logger::log(const string & line) {
    logFile << line << endl;
    cout << line << endl;
}

void Logger::lossPrint(double loss) {
    lossResultFile << loss << endl;
}