#pragma once

#include <chrono>
#include <cstring>

#include "ConnectionPool.hpp"
#include "DateUtils.hpp"
#include "Models.hpp"

#include "cppconn/prepared_statement.h"
#include "cppconn/statement.h"
#include "cppconn/resultset.h"
#include "mysql_connection.h"

struct DataStore {
    std::shared_ptr<ConnectionPool<10>> pool = std::make_shared<ConnectionPool<10>>();

    [[nodiscard]] std::chrono::year_month_day getStartDate() const;

    [[nodiscard]] std::optional<std::chrono::year_month_day> getNextDate(std::chrono::year_month_day const& prevDate) const;

    [[nodiscard]] std::vector<OptionChain> getOptionChain(std::chrono::year_month_day const& date) const;

    [[nodiscard]] std::vector<OptionChain> getOptionChain(std::chrono::year_month_day const& date, std::chrono::year_month_day const& expiration) const;

    [[nodiscard]] std::vector<OptionChain> getOptionChain(std::string const& symbol, std::chrono::year_month_day const& date) const;

    [[nodiscard]] std::vector<Straddle> getStraddles(const std::chrono::year_month_day &date, const std::chrono::year_month_day &expiration) const;

    [[nodiscard]] std::vector<Stock> getStocks(std::chrono::year_month_day const& date) const;

    [[nodiscard]] std::vector<std::string> getSAP100Stocks(std::chrono::year_month_day const& date) const;

    [[nodiscard]] std::vector<VixData> getVix() const;

    [[nodiscard]] SimpleMovingAverages getSimpleMovingAverages(const std::string& symbol, std::chrono::year_month_day date) const;

    [[nodiscard]] ExpMovingAverages getExpMovingAverages(const std::string& symbol, std::chrono::year_month_day date) const;

    [[nodiscard]] StandardDeviation getStandardDeviation(const std::string& symbol, std::chrono::year_month_day date) const;

    [[nodiscard]] StraddleDerived getStraddleDerived(int straddle_id, std::chrono::year_month_day date) const;

    [[nodicard]] std::vector<StraddleDerived> getStraddleDerivedHistory(int straddle_id, std::chrono::year_month_day from, int days) const;
};
