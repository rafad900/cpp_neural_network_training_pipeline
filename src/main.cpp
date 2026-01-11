#include <iostream>
#include <mongocxx/instance.hpp>
#include "MongoManager.hpp"
#include "rest_client.hpp"
#include <torch/torch.h>
#include <torch/nn/functional/dropout.h>
#include "ModelTrainer.hpp"
#include <memory>
#include <string>

struct Net1 : public BaseNet {
    Net1(int input_channels, int output_dim) {
        set_model_name("Net1");
        conv1 = register_module("conv1", torch::nn::Conv2d(input_channels, 32, 3));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 5).padding(2)));
        conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 7).padding(3)));
        
        fc1 = register_module("fc1", torch::nn::Linear(512, 256));
        fc2 = register_module("fc2", torch::nn::Linear(256, 128));
        fc3 = register_module("fc3", torch::nn::Linear(128, output_dim));

        linear_dropout_options = torch::nn::functional::DropoutFuncOptions().p(0.5).training(is_training());
    }
    torch::Tensor forward(torch::Tensor x) override {
        x = torch::relu(conv1->forward(x));
        x = torch::nn::functional::dropout2d(x, torch::nn::functional::Dropout2dFuncOptions().p(0.2).training(is_training()));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(conv2->forward(x));
        x = torch::nn::functional::dropout2d(x, torch::nn::functional::Dropout2dFuncOptions().p(0.2).training(is_training()));
        x = torch::relu(conv3->forward(x));
        x = torch::nn::functional::dropout2d(x, torch::nn::functional::Dropout2dFuncOptions().p(0.2).training(is_training()));
        x = torch::relu(conv4->forward(x));
        x = torch::nn::functional::dropout2d(x, torch::nn::functional::Dropout2dFuncOptions().p(0.2).training(is_training()));
        x = torch::relu(conv5->forward(x));

        x = torch::nn::functional::adaptive_avg_pool2d(x, 
            torch::nn::functional::AdaptiveAvgPool2dFuncOptions({1,1}));
        x = x.view({x.size(0), -1});

        x = fc1->forward(x);
        x = torch::nn::functional::dropout(x, linear_dropout_options);
        x = fc2->forward(x);
        x = torch::nn::functional::dropout(x, linear_dropout_options);
        x = fc3->forward(x);
        return x;

    }

    torch::nn::functional::DropoutFuncOptions linear_dropout_options;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
};


int main() {
    mongocxx::instance instance{};
    MongoManager db("mongodb://root:password123@localhost:27017");
    db.saveData("https://google.com", "Hello world", 200);
    std::cout << "Saved the data to the database\n";

    std::unique_ptr<CustomClient> client = std::make_unique<CustomClient>();
    client->create_client("https://www.thecocktaildb.com/api/json/v1/1/search.php?s=margarita");
    std::cout << "Created the client\n";
    
    std::unique_ptr<ModelTrainer> trainer = std::make_unique<ModelTrainer>();
    trainer->ensure_torch_can_run();

    auto model = std::make_shared<Net1>(3,10);
    trainer->set_model(model);

    trainer->train_model();
    // trainer->test_model();

    return 0;
}
