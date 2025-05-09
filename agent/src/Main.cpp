#include "Environment.hpp"
#include "Data.hpp"

int main() {

    Environment<30, 64, 1> env(10);

    auto data = Data();

    const std::vector<VixData> vix = data.getVixData();

    std::vector<Sample<30, 1>> samples;

    for (int i = 0; i < vix.size() - 30; i++) {
        LinearLib::Matrix<30, 1, float> input = LinearLib::Matrix<30, 1, float>::zeros();
        for (int j = 0; j < 30; j++) {
            input[j][0] = static_cast<float>(vix[i + j].vix);
        }

        const auto label = LinearLib::Matrix<1, 1, float>{{static_cast<float>(vix[i + 30].vix)}};

        samples.emplace_back(input, label);
    }

    env.train(samples);

    env.validate(samples);

    return 0;
}

