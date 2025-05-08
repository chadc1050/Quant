#pragma once
#include <iostream>
#include <sys/types.h>

#include "RNN.hpp"

template<std::size_t I, std::size_t O>
struct Sample {
    LinearLib::Matrix<I, 1, float> input;
    LinearLib::Matrix<O, 1, float> label;

    Sample(LinearLib::Matrix<I, 1, float> input, LinearLib::Matrix<O, 1, float> label) : input(input), label(label) {}
};

struct Environment {
    unsigned int nEpochs;
    int seed;
    RNN<30, 64, 1> model = RNN<30, 64, 1>(0.001f, 10.0f);

    Environment(unsigned int nEpochs, int seed = 42) {
        this->nEpochs = nEpochs;
        this->seed = seed;
    }

    void train() {
        LinearLib::Matrix<30, 1, float> x = LinearLib::Matrix<30, 1, float>::random(1, 100, this->seed);

        std::vector samples = {
            Sample(x, LinearLib::Matrix<1, 1, float>{{30}})
        };

        for (uint i = 0; i < nEpochs; i++) {
            std::cout << "Starting epoch: " << i + 1 << std::endl;

            float loss = 0;

            for (Sample<30, 1> &sample: samples) {
                LinearLib::Matrix<1, 1, float> y = model.forward(sample.input);

                loss += std::pow(sample.label[0][0] - y[0][0], 2) / 2;

                const LinearLib::Matrix<1, 1, float> d_y = sample.label - y;

                model.backward(d_y);
            }

            std::cout << "Value Loss: " << loss << " Loss: " << loss / static_cast<float>(samples.size()) << std::endl;
        }
    }

    void validate() {

    }
};
