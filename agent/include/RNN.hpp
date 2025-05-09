#pragma once

#include "LinearLib/Matrix.hpp"
#include <vector>
#include <cmath>

template<std::size_t I, std::size_t H, std::size_t O>
struct RNN {
    LinearLib::Matrix<H, I, float> w_i_h;
    LinearLib::Matrix<O, H, float> w_h_o;

    LinearLib::Matrix<H, 1, float> b_i_h;
    LinearLib::Matrix<O, 1, float> b_h_o;

    std::vector<LinearLib::Matrix<H, 1, float>> history = {};

    float learning_rate;
    float clip;
    int seed;

    RNN(float learning_rate, float clip, int seed = 42) {
        this->learning_rate = learning_rate;
        this->clip = clip;
        this->seed = seed;

        w_i_h = LinearLib::Matrix<H, I, float>::random(0, 1, seed);
        w_h_o = LinearLib::Matrix<O, H, float>::random(0, 1, seed);

        b_i_h = LinearLib::Matrix<H, 1, float>::zeros();
        b_h_o = LinearLib::Matrix<O, 1, float>::zeros();
    }

    LinearLib::Matrix<O, 1, float> forward(const LinearLib::Matrix<I, 1, float>& x) {

        LinearLib::Matrix<H, 1, float> h = LinearLib::Matrix<H, 1, float>::zeros();

        history.push_back(h);

        for (std::size_t i = 0; i < I; i++) {
            h = (this->w_i_h & x) + this->b_i_h;
            h.forEach([this](float& val) {
                val = std::tanh(val);
            });
            history.push_back(h);
        }

        LinearLib::Matrix<1, O, float> y = (this->w_h_o & h) + this->b_h_o;

        return y;
    }

    void backward(LinearLib::Matrix<O, 1, float> d_y) {

        // Init derivatives
        LinearLib::Matrix<O, H, float> d_w_h_o = LinearLib::Matrix<O, H, float>::zeros();
        LinearLib::Matrix<H, I, float> d_w_i_h = LinearLib::Matrix<H, I, float>::zeros();

        LinearLib::Matrix<O, 1, float> d_b_h_o = LinearLib::Matrix<O, 1, float>::zeros();
        LinearLib::Matrix<H, 1, float> d_b_i_h = LinearLib::Matrix<H, 1, float>::zeros();


        LinearLib::Matrix<H, 1, float> d_h = this->w_h_o.transpose() & d_y;

        // Backprop through time
        for (std::size_t t = this->history.size(); t > 0; --t) {

            LinearLib::Matrix<H, 1, float> h_sq = LinearLib::Matrix<H, 1, float>::zeros();
            h_sq.forEach([this](float& val) {
                val = static_cast<float>(std::pow(val, 2));
            });

            LinearLib::Matrix<H, 1, float> d_l_h = (LinearLib::Matrix<H, 1, float>::ones() - h_sq) * d_h;

            d_b_i_h = d_b_i_h + d_l_h;
            d_b_h_o = d_b_h_o + (this->w_h_o & d_l_h);

            // For hidden to output weights
            d_w_h_o = d_w_h_o + (d_y & this->history[t-1].transpose());
        }

        // Apply gradient clipping
        const auto clipGradient = [this](float& val) {
            if (val > this->clip) {
                val = this->clip;
            } else if (val < -this->clip) {
                val = -this->clip;
            }
        };

        // Clip
        d_w_h_o.forEach(clipGradient);
        d_w_i_h.forEach(clipGradient);
        d_b_h_o.forEach(clipGradient);
        d_b_i_h.forEach(clipGradient);

        // Apply updates (missing from your original code)
        this->w_h_o = this->w_h_o + d_w_h_o * this->learning_rate;
        this->w_i_h = this->w_i_h + d_w_i_h * this->learning_rate;
        this->b_h_o = this->b_h_o + d_b_h_o * this->learning_rate;
        this->b_i_h = this->b_i_h + d_b_i_h * this->learning_rate;
    }


    LinearLib::Matrix<H, 1, float> get_hidden_state() {
        return history.back();
    }

};
