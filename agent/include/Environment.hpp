#pragma once
#include <iostream>

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
    int seed;
    RNN<I, H, O> model = RNN<I, H, O>(0.001f, 10.0f);

    explicit Environment(const unsigned int nEpochs, const int seed = 42) {
        this->nEpochs = nEpochs;
        this->seed = seed;
    }

    void train(const std::vector<Sample<I, O>>& input) {

        std::cout << "Beginning training..." << std::endl;

        for (uint i = 0; i < nEpochs; i++) {
            std::cout << "Starting epoch: " << i + 1 << std::endl;

            float loss = 0;

            for (int j = 0; j < input.size(); j++) {

                Sample<I, O> sample = input[j];

                LinearLib::Matrix<1, 1, float> y = model.forward(sample.input);

                loss += mse(sample, y);

                const LinearLib::Matrix<1, 1, float> d_y = sample.label - y;

                model.backward(d_y);
            }

            std::cout << "Value Loss: " << loss << " Loss: " << loss / static_cast<float>(input.size()) << std::endl;
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

    static float mse(const Sample<I, O>& sample, LinearLib::Matrix<O, 1, float> pred) {
        return std::pow(sample.label[0][0] - pred[0][0], 2) / 2;
    }
};
