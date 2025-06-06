#pragma once
#include "Signal.hpp"

class Long final : public Signal {
    public:
        Long() = default;
        [[nodiscard]] float signal(const std::string &symbol, std::chrono::year_month_day date) const override { return 1;}
};
