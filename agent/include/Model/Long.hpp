#pragma once
#include "Signal.hpp"

class Long final : public Signal {
    public:
        Long() = default;
        [[nodiscard]] float signal(const int straddle_id, std::chrono::year_month_day date) const override { return 1;}
};
