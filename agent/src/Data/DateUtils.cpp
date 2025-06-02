#include "Data/DateUtils.hpp"

std::chrono::year_month_day parseDate(const std::string& dateStr) {
    int year, month, day;
    std::sscanf(dateStr.c_str(), "%d-%d-%d", &year, &month, &day);

    return std::chrono::year_month_day{
        std::chrono::year(year),
        std::chrono::month(month),
        std::chrono::day(day)
    };
}

std::string formatDate(const std::chrono::year_month_day& date) {
    char buffer[11];
    sprintf(buffer, "%04d-%02d-%02d",
            static_cast<int>(date.year()),
            static_cast<unsigned>(date.month()),
            static_cast<unsigned>(date.day()));
    return std::string(buffer);
}

bool isThirdFriday(const std::chrono::year_month_day &date) {
    const auto sys_days = std::chrono::sys_days(date);

    if (const auto wday = std::chrono::weekday(sys_days); wday != std::chrono::Friday) {
        return false;
    }

    const unsigned day = static_cast<unsigned>(date.day());

    return (day - 1) / 7 + 1 == 3;
}

std::chrono::year_month_day nextThirdFriday(const std::chrono::year_month_day &date) {

    const auto sysDays = std::chrono::sys_days(date);

    // Get the current year and month
    const std::chrono::year year = date.year();
    const std::chrono::month month = date.month();

    // Calculate the first day of the current month
    const std::chrono::year_month_day firstOfMonth{year, month, std::chrono::day{1}};
    const auto firstDay = std::chrono::sys_days{firstOfMonth};

    // Calculate what weekday the 1st falls on (0 = Sunday, 6 = Saturday)
    const std::chrono::weekday firstWeekday{firstDay};

    // Calculate how many days to add to get to the first Friday
    // If 1st is Friday, add 0; if 1st is Saturday, add 6 days; etc.
    int daysUntilFirstFriday = (std::chrono::Friday - firstWeekday).count();
    if (daysUntilFirstFriday < 0) {
        daysUntilFirstFriday += 7;
    }

    // The third Friday is the first Friday plus 14 days
    const std::chrono::sys_days thirdFriday = firstDay + std::chrono::days{daysUntilFirstFriday + 14};
    std::chrono::year_month_day thirdFridayYmd{thirdFriday};

    // If the current date is after or equal to the third Friday of the current month,
    // we need to find the third Friday of the next month
    const std::chrono::year_month current_ym{year, month};
    if (sysDays >= thirdFriday) {
        // Move to the first day of next month
        std::chrono::year_month nextYm;
        if (current_ym.month() == std::chrono::month{12}) {
            nextYm = std::chrono::year_month{current_ym.year() + std::chrono::years{1}, std::chrono::month{1}};
        } else {
            nextYm = std::chrono::year_month{current_ym.year(), current_ym.month() + std::chrono::months{1}};
        }

        const std::chrono::year_month_day firstOfNextMonth{nextYm.year(), nextYm.month(), std::chrono::day{1}};
        const auto nextFirstDay = std::chrono::sys_days{firstOfNextMonth};

        // Calculate weekday of the first day of next month
        const std::chrono::weekday nextFirstWeekday{nextFirstDay};

        // Calculate days until first Friday of next month
        int nextDaysUntilFirstFriday = (std::chrono::Friday - nextFirstWeekday).count();
        if (nextDaysUntilFirstFriday < 0) {
            nextDaysUntilFirstFriday += 7;
        }

        // The third Friday is the first Friday plus 14 days
        const std::chrono::sys_days nextThirdFriday = nextFirstDay + std::chrono::days{nextDaysUntilFirstFriday + 14};
        thirdFridayYmd = std::chrono::year_month_day{nextThirdFriday};
    }

    return thirdFridayYmd;
}