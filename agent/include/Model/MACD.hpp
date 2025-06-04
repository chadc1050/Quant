#pragma once

#include <memory>
#include <chrono>

#include "Signal.hpp"
#include "Data/DataStore.hpp"

class MACD : public Signal {
    public:
        explicit MACD(std::shared_ptr<DataStore>& data);
        float signal(const std::string &symbol, std::chrono::year_month_day date) const override;

    private:
        std::shared_ptr<DataStore> data;
};
