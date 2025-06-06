#include "Model/MACD.hpp"

#include <exception>
#include <cmath>

MACD::MACD(std::shared_ptr<DataStore>& data) {
    this->data = data;
}

float MACD::signal(const std::string &symbol, const std::chrono::year_month_day date) const {

    const std::vector<StraddleDerived> history = data->getStraddleDerivedHistory(symbol, date, 20);

    if (history.size() < 20) {
        throw std::runtime_error("Less than 20 derivation records!");
    }

    if (history.front().ema.ema32 == 0.0f) {
        throw std::runtime_error("No EMA32 value!");
    }

    float X = 0.0f;

    for (int i = 0; i < 3; i++) {
        std::array<float, 20> macd_norms = {};

        int counter = 0;
        // Calculate prior 20 normalized MACDs
        for (const StraddleDerived& derived : history) {

            std::array ema_S = {derived.ema.ema2, derived.ema.ema4, derived.ema.ema8};
            std::array ema_L = {derived.ema.ema8, derived.ema.ema16, derived.ema.ema32};

            float macd =  ema_S[i] - ema_L[i];
            macd_norms[counter] = macd / derived.std.std5;
            counter++;
        }

        // Get the standard deviation of the last 20 normalized MACDs
        float sum = 0.0f;
        for (const float macd_norm : macd_norms) {
            sum += macd_norm;
        }

        const float mean_macd_norms = sum / 20;

        float std_macd_norms = 0.0f;
        for (const float macd_norm : macd_norms) {
            std_macd_norms += (macd_norm - mean_macd_norms) * (macd_norm - mean_macd_norms);
        }

        std_macd_norms = std::sqrt(std_macd_norms / 20);

        const float Y = macd_norms.front() / std_macd_norms;

        X += Y * static_cast<float>(std::exp(-1 * (Y * Y) / 4) / 0.89 / 3);
    }

    return X;
}
