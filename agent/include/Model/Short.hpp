#pragma once
#include "Signal.hpp"

class Short final : public Signal {
public:
    Short() = default;
    [[nodiscard]] float signal(const int straddle_id, std::chrono::year_month_day date) const override { return -1;}
};
