#include <torch/torch.h>

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
private:
    torch::Tensor images_, labels_;

public:
    CustomDataset(torch::Tensor images, torch::Tensor labels)
        : images_(images), labels_(labels) {}
    
    torch::data::Example<> get(size_t index) override {
    
        return {images_[index], labels_[index]};
    }
    
    torch::optional<size_t> size() const override {
        return images_.size(0);
    }
};