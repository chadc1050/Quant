#pragma once
#include "Signal.hpp"

class Short final : public Signal {
public:
    Short() = default;
    [[nodiscard]] float signal(const std::string &symbol, std::chrono::year_month_day date) const override { return -1;}
};
