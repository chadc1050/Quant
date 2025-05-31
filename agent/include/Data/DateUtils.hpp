#pragma once

#include <chrono>

[[nodiscard]] std::chrono::year_month_day parseDate(const std::string& dateStr);

[[nodiscard]] std::string formatDate(const std::chrono::year_month_day &date);

bool isThirdFriday(const std::chrono::year_month_day &date);

std::chrono::year_month_day nextThirdFriday(const std::chrono::year_month_day &date);