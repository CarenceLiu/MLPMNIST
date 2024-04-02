#include <string>
#include <fstream>

namespace std{
    class Logger {
        public:
            Logger() = delete;
            Logger(const string log_filename, const string loss_filename);
            void log(int epoch, double train_loss, double validation_loss, double train_acc, double validation_acc);
            void log(const string & line);
            void lossPrint(double loss);
        private:
            ofstream logFile;
            ofstream lossResultFile;
    };
}