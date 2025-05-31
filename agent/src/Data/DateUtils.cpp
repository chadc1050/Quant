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

    const auto sys_days = std::chrono::sys_days(date);

    // Get the current year and month
    const std::chrono::year year = date.year();
    const std::chrono::month month = date.month();

    // Calculate the first day of the current month
    const std::chrono::year_month_day first_of_month{year, month, std::chrono::day{1}};
    const std::chrono::sys_days first_day = std::chrono::sys_days{first_of_month};

    // Calculate what weekday the 1st falls on (0 = Sunday, 6 = Saturday)
    const std::chrono::weekday first_weekday{first_day};

    // Calculate how many days to add to get to the first Friday
    // If 1st is Friday, add 0; if 1st is Saturday, add 6 days; etc.
    int days_until_first_friday = (std::chrono::Friday - first_weekday).count();
    if (days_until_first_friday < 0) {
        days_until_first_friday += 7;
    }

    // The third Friday is the first Friday plus 14 days
    std::chrono::sys_days third_friday = first_day + std::chrono::days{days_until_first_friday + 14};
    std::chrono::year_month_day third_friday_ymd{third_friday};

    // If the current date is after or equal to the third Friday of the current month,
    // we need to find the third Friday of the next month
    const std::chrono::year_month current_ym{year, month};
    if (sys_days >= third_friday) {
        // Move to the first day of next month
        std::chrono::year_month next_ym;
        if (current_ym.month() == std::chrono::month{12}) {
            next_ym = std::chrono::year_month{current_ym.year() + std::chrono::years{1}, std::chrono::month{1}};
        } else {
            next_ym = std::chrono::year_month{current_ym.year(), current_ym.month() + std::chrono::months{1}};
        }

        std::chrono::year_month_day first_of_next_month{next_ym.year(), next_ym.month(), std::chrono::day{1}};
        std::chrono::sys_days next_first_day = std::chrono::sys_days{first_of_next_month};

        // Calculate weekday of the first day of next month
        std::chrono::weekday next_first_weekday{next_first_day};

        // Calculate days until first Friday of next month
        int next_days_until_first_friday = (std::chrono::Friday - next_first_weekday).count();
        if (next_days_until_first_friday < 0) {
            next_days_until_first_friday += 7;
        }

        // The third Friday is the first Friday plus 14 days
        std::chrono::sys_days next_third_friday = next_first_day + std::chrono::days{next_days_until_first_friday + 14};
        third_friday_ymd = std::chrono::year_month_day{next_third_friday};
    }

    return third_friday_ymd;

}