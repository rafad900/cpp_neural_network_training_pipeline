#pragma once

#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;


class CIFAR10 {
private:  
    torch::Tensor images, labels;
public:
    CIFAR10(const std::string& path) {

        std::vector<torch::Tensor> all_images;
        std::vector<torch::Tensor> all_labels;

        int file_count = 0;

        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                std::string path = entry.path().string();
                std::ifstream file(path, std::ios::binary);

                if (!file.is_open()) 
                    std::cout << "FILE AT PATH NOT OPEN: " << path << std::endl;

                const int num_images = 10000;
                const int image_size = 3072;
                std::vector<char> buffer((image_size + 1) * num_images);
                file.read(buffer.data(), buffer.size());

                auto file_images = torch::empty({num_images, 3, 32, 32}, torch::kUInt8);
                auto file_labels = torch::empty({num_images}, torch::kInt64);

                for (int i = 0; i < num_images; ++i) {
                    int start_index = i * (image_size + 1);
                    file_labels[i] = static_cast<unsigned char>(buffer[start_index]);
                    auto raw_img = torch::from_blob(&buffer[start_index + 1], {3, 32, 32}, torch::kUInt8);
                    file_images[i] = raw_img.clone();
                }
                
                all_images.push_back(file_images);
                all_labels.push_back(file_labels);
                std::cout << "Loaded: " << entry.path().filename() << std::endl;
                file_count++;
            }
        }

        images = torch::cat(all_images, 0).to(torch::kFloat32).div(255.0);
        labels = torch::cat(all_labels, 0);
        
        std::cout << "Full Tensor Shape: " << images.sizes() << std::endl;
    }

    torch::Tensor get_images() { return images; }
    torch::Tensor get_labels() { return labels; }
};