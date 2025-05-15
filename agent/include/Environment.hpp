#pragma once
#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>

#include "RNN.hpp"

template<std::size_t I, std::size_t O>
struct Sample {
    LinearLib::Matrix<I, 1, float> input;
    LinearLib::Matrix<O, 1, float> label;

    Sample(LinearLib::Matrix<I, 1, float> input, LinearLib::Matrix<O, 1, float> label) : input(input), label(label) {}
};

template<std::size_t I, std::size_t H, std::size_t O>
struct Environment {
    unsigned int nEpochs;
    unsigned int patience;
    unsigned int currentEpoch = 0;
    int seed;
    RNN<I, H, O> model = RNN<I, H, O>(0.05f, 10.0f);

    explicit Environment(const unsigned int nEpochs, const int patience = 20, const int seed = 42) {
        this->nEpochs = nEpochs;
        this->patience = patience;
        this->seed = seed;
    }

    void train(const std::vector<Sample<I, O>>& input) {

        std::cout << "Beginning Training..." << std::endl;

        float minLoss = std::numeric_limits<float>::max();
        unsigned int patienceCounter = 0;

        for (uint i = 0; i < nEpochs; i++) {
            currentEpoch++;
            std::cout << "Starting Epoch " << currentEpoch << std::endl;
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            float loss = 0;

            for (std::size_t j = 0; j < input.size(); j++) {

                Sample<I, O> sample = input[j];

                LinearLib::Matrix<1, 1, float> y = model.forward(sample.input);

                loss += mse(sample, y);

                const LinearLib::Matrix<1, 1, float> d_y = sample.label - y;

                model.backward(d_y);

                model.clearHistory();
            }

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Epoch " << currentEpoch << " Completed. Elapsed Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms Value Loss: " << loss << " Loss: " << loss / static_cast<float>(input.size()) << std::endl;
            save();

            if (loss >= minLoss) {
                patienceCounter++;
                if (patienceCounter >= this->patience) {
                    std::cout << "Patience exceeded, early stopping." << std::endl;
                    break;
                }
            } else {
                minLoss = loss;
                patienceCounter = 0;
                std::cout << "New record." << std::endl;
            }
        }

        std::cout << "Training complete" << std::endl;
    }

    void validate(const std::vector<Sample<I, O>>& input) {

        std::cout << "Beginning validating..." << std::endl;

        float loss = 0;

        for (Sample sample: input) {
            const LinearLib::Matrix<1, 1, float> y = model.forward(sample.input);

            loss += mse(sample, y);
        }

        std::cout << "Value Loss: " << loss << " Loss: " << loss / static_cast<float>(input.size()) << std::endl;

        std::cout << "Validation complete" << std::endl;
    }

    float predict(const LinearLib::Matrix<I, O, float> &input) {
        return model.forward(input)[0][0];
    }

    void save() {
        std::cout << "Saving model..." << std::endl;
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

        std::filesystem::create_directories("data");

        const auto filePath = std::format("data/model_{}_{}.qnt", currentEpoch, now.time_since_epoch().count());

        std::cout << "Outputting file: " << filePath.c_str() << std::endl;

        std::ofstream file(filePath.c_str());

        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filePath.c_str() << std::endl;
            return;
        }

        file << model.serialize();

        file.close();
    }

    void load(std::string const& path) {
        std::ifstream file;
        file.open(path.c_str());
        this->model = RNN<I, H, O>::import(file.rdbuf());
    }

    static float mse(const Sample<I, O>& sample, LinearLib::Matrix<O, 1, float> pred) {
        return std::pow(sample.label[0][0] - pred[0][0], 2) / 2;
    }
};
