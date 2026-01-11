#pragma once

#include <torch/torch.h>
#include <memory>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;

struct BaseNet : torch::nn::Module {
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual ~BaseNet() = default;
public:
    std::string model_name = "";
protected:
    void set_model_name(std::string name) { model_name = name; }
};

class ModelTrainer {
private:
    int batch_size = 128;
    int num_classes = 10;
    double learning_rate = 0.00025;
    int num_epochs = 50;

    bool loaded_pretrained_weights = false;
    
    std::string model_name = "";
    std::string pretrained_file = "";
    std::string pretrained_path = fs::current_path().string();

    torch::Device device_{torch::kCPU};

    std::shared_ptr<BaseNet> model_;

    void check_for_gpu();

public:
    ModelTrainer();
    ~ModelTrainer();
    ModelTrainer(std::shared_ptr<BaseNet> model);

    void set_batch_size(int value);
    void set_num_classes(int value);
    void set_learning_rate(double value);
    void set_num_epochs(int value);
    void set_model(std::shared_ptr<BaseNet> model);
    void print_current_parameters();
    void ensure_torch_can_run();
    void train_model();
    void test_model();
};

