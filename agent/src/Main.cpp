#include "Environment.hpp"
#include "Data.hpp"

#include <algorithm>
#include <random>

template<std::size_t I, std::size_t O>
std::vector<Sample<I, O>> generateSamples(const std::vector<VixData>& vix) {

    std::vector<Sample<I, O>> samples;

    for (std::size_t i = 0; i < vix.size() - I; i++) {
        LinearLib::Matrix<I, 1, float> input = LinearLib::Matrix<I, 1, float>::zeros();
        for (std::size_t j = 0; j < I; j++) {
            input[j][0] = static_cast<float>(vix[i + j].vix);
        }

        const auto label = LinearLib::Matrix<O, 1, float>{{static_cast<float>(vix[i + I].vix)}};

        samples.emplace_back(input, label);
    }

    std::shuffle(samples.begin(), samples.end(), std::default_random_engine(42));

    return samples;
}

int main() {

    Environment<32, 512, 1> env(1000);

    const auto data = Data();

    const std::vector<VixData> vix = data.getVixData();

    std::vector<Sample<32, 1>> samples = generateSamples<32, 1>(vix);

    const std::size_t trainingEndIdx = static_cast<size_t>(samples.size() * 0.8);
    const std::size_t validationStartIdx = trainingEndIdx;

    const std::vector trainingSamples(samples.begin(), samples.begin() + trainingEndIdx);
    const std::vector validationSamples(samples.begin() + validationStartIdx, samples.end());

    env.train(trainingSamples);

    env.validate(validationSamples);

    return 0;
}

