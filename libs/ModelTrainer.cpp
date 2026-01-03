#include "ModelTrainer.hpp"
#include "CIFAR10Loader.hpp"
#include "CustomDataset.hpp"

void look_for_pt_file(std::string& final_path, std::string& final_filename) {
    while (true) {
        std::cout << "Path to pretrained weights [if none, leave empty]: ";
        std::string temporary_path;
        std::getline(std::cin, temporary_path);
        fs::path path = temporary_path;

        if (temporary_path == "") {
            break;
        }
        if (!fs::exists(path)) {
            std::cout << "The path does not exist\n";
            continue;
        }
        if (!fs::is_regular_file(path)) {
            std::cout << "The path is not a file. Give full path not just directory\n";
            continue;
        }
        if (path.extension() != ".pt") {
            std::cout << "The file is not a '.pt' file. Please try again\n";
            continue;
        }
        final_filename  = path.filename();
        final_path      = path.parent_path().string();
        break;
    }
    std::cout << "Path: " << final_path << std::endl << "Filename: " << final_filename << std::endl;
}

std::string get_current_time_str() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    // Use put_time to format into the stream
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d_%H:%M:%S");
    return ss.str();
}

ModelTrainer::ModelTrainer() {
    std::cout << "Intializing Model Trainer\n";
    print_current_parameters();
    check_for_gpu();
    look_for_pt_file(pretrained_path, pretrained_file);
}

ModelTrainer::ModelTrainer(std::shared_ptr<BaseNet> model) {
    std::cout << "Initializing Model Trainer with custom model\n";
    print_current_parameters();
    check_for_gpu();
    look_for_pt_file(pretrained_path, pretrained_file);
    set_model(model);
}

ModelTrainer::~ModelTrainer() {}

void ModelTrainer::print_current_parameters() {
    std::cout << "Current hyper-parameters: "
        << "batch_size = " << batch_size
        << "num_classes = " << num_classes
        << "learning_rate = " << learning_rate
        << "num_epochs = " << num_epochs 
        << std::endl;
}

void ModelTrainer::train_model() {
    if (model_ == nullptr) {
        std::cerr << "NO MODEL SET UP!\n";
        return;
    }
    if (!loaded_pretrained_weights && !pretrained_path.empty() && !pretrained_file.empty()) {
        std::cout << "Loaded the pretrained weights\n";
        loaded_pretrained_weights = true;
        fs::path full_path = pretrained_path;
        fs::path filename = pretrained_file;
        full_path = full_path / filename;
        torch::load(model_, full_path.string() );
    }

    model_->train();
    model_->to(device_);
    torch::optim::Adam optimizer = torch::optim::Adam(model_->parameters(), torch::optim::AdamOptions(learning_rate));

    CIFAR10 all_data = CIFAR10("/home/rafad900/Data/cifar-10-batches-bin/train");
    auto train_dataset  = CustomDataset(all_data.get_images(), all_data.get_labels()).map(torch::data::transforms::Stack<>());
    auto train_loader   = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);
    std::cout << "Data loader created with size: " << train_dataset.size().value() << std::endl;
    
    for (int i = 0; i < num_epochs; i++) {
        auto gpu_epoch_loss = torch::zeros({}, device_);
        auto gpu_correct_total = torch::zeros({}, torch::TensorOptions().device(device_).dtype(torch::kInt64));
        int64_t total_samples = 0;

        for (auto& batch : *train_loader) {       

            auto data   = batch.data.to(device_);
            auto target = batch.target.to(device_);
            
            optimizer.zero_grad();

            // Forward pass
            auto output = model_->forward(data);
            auto loss   = torch::nn::functional::cross_entropy(output, target);
            
            // Backward pass
            loss.backward();
            optimizer.step();

            // Track Metrics
            gpu_epoch_loss += loss;
            auto prediction = output.argmax(1);
            gpu_correct_total += prediction.eq(target).sum();
            total_samples += data.size(0);
        }

        double current_accuracy = (double)gpu_correct_total.item<int64_t>() / total_samples * 100;

        // Print summary after each Epoch
        std::cout << "Epoch [" << (i + 1) << "/" << num_epochs << "] "
                << "Loss: " << gpu_epoch_loss.item<float>() / total_samples << " "
                << "Accuracy: " << current_accuracy << "%" 
                << std::endl;
    }
}

void ModelTrainer::test_model() {
    if (model_ == nullptr) {
        std::cerr << "NO MODEL SET UP!\n";
        return;
    }
    
    if (!loaded_pretrained_weights && !pretrained_path.empty() && !pretrained_file.empty()) {
        std::cout << "Loaded the pretrained weights\n";
        loaded_pretrained_weights = true;
        fs::path full_path = pretrained_path;
        fs::path filename = pretrained_file;
        full_path = full_path / filename;
        torch::load(model_, full_path.string() );
    }

    CIFAR10 all_data = CIFAR10("/home/rafad900/Data/cifar-10-batches-bin/test");
    auto test_dataset   = CustomDataset(all_data.get_images(), all_data.get_labels()).map(torch::data::transforms::Stack<>());
    auto options = torch::data::DataLoaderOptions()
        .batch_size(64)
        .drop_last(false)   // This ensures the last partial batch is kept
        .workers(4);        // Optional: helps with speed!
    auto test_loader    = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), options);
    auto gpu_correct_total = torch::zeros({}, torch::TensorOptions().device(device_).dtype(torch::kInt64));
    int64_t total_samples = 0;
    model_->eval();
    torch::NoGradGuard no_grad; 
    for (auto& batch : *test_loader) {       
        auto data   = batch.data.to(device_);
        auto target = batch.target.to(device_);
        
        // Forward pass
        auto output = model_->forward(data);
        auto loss   = torch::nn::functional::cross_entropy(output, target);
        
        auto prediction = output.argmax(1);
        gpu_correct_total += prediction.eq(target).sum();
        total_samples += data.size(0);
    }

    double current_accuracy = (double)gpu_correct_total.item<int64_t>() / total_samples * 100;

    std::cout << "TEST RUN --- Accuracy: " << current_accuracy << "%" << std::endl;

    std::string decision = "";
    std::cout << "Save the current weights? [yes | no]: ";
    std::getline(std::cin, decision);
    if (decision == "Yes" || decision == "yes" || decision == "y") {
        std::string saved_path = pretrained_path + "/" + model_name + "_" + get_current_time_str() + ".pt";
        torch::save(model_, saved_path);
        std::cout << "Weights saved to: " << saved_path << std::endl;
    }
}

void ModelTrainer::ensure_torch_can_run() {
    std::cout << "--- Starting Torch Integration Test ---" << std::endl;
    auto test_model = torch::nn::Linear(784, 10);
    test_model->to(device_); 

    double test_lr = 1e-3;
    torch::optim::Adam test_optimizer(test_model->parameters(), torch::optim::AdamOptions(test_lr));

    auto data = torch::randn({1, 784}).to(device_);
    auto target = torch::tensor({5}, torch::kLong).to(device_);

    test_optimizer.zero_grad();
    
    auto output = test_model->forward(data);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    
    loss.backward();
    test_optimizer.step();

    std::cout << "Test Complete. Loss value: " << loss.item<float>() << std::endl;
    std::cout << "---------------------------------------" << std::endl;
}

void ModelTrainer::check_for_gpu() {
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device_ = torch::Device(torch::kCUDA);
    } else {
        std::cout << "CUDA not found. Falling back to CPU." << std::endl;
    }
}

void ModelTrainer::set_model(std::shared_ptr<BaseNet> model) {
    model_ = model;
    model_name = model->model_name;
}
