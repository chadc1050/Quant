#pragma once

#include <memory>
#include <chrono>

#include "Signal.hpp"
#include "Data/DataStore.hpp"

class MACD final : public Signal {
    public:
        explicit MACD(std::shared_ptr<DataStore>& data);
        [[nodiscard]] float signal(const std::string &symbol, std::chrono::year_month_day date) const override;

    private:
        std::shared_ptr<DataStore> data;
};
