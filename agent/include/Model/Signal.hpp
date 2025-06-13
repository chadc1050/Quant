#pragma once

#include <chrono>
#include <string>
#include <concepts>

class Signal {
public:
    virtual ~Signal() = default;
    [[nodiscard]] virtual float signal(int straddle_id, std::chrono::year_month_day date) const = 0;
};