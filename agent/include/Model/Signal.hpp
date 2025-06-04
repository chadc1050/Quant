#pragma once

#include <chrono>
#include <string>
#include <concepts>

class Signal {
public:
    virtual ~Signal() = default;
    virtual float signal(const std::string& symbol, std::chrono::year_month_day date) const = 0;
};