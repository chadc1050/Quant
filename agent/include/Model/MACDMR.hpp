#pragma once

#include "MACD.hpp"
#include "Signal.hpp"
#include "Data/DataStore.hpp"

class MACDMR final : public Signal {
public:
    explicit MACDMR(std::shared_ptr<DataStore>& data);
    [[nodiscard]] float signal(const std::string &symbol, std::chrono::year_month_day date) const override;

private:
    std::shared_ptr<MACD> macd;
};
