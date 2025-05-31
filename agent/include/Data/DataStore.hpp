#pragma once

#include <chrono>
#include <cstring>

#include "ConnectionPool.hpp"
#include "DateUtils.hpp"

#include "cppconn/prepared_statement.h"
#include "cppconn/statement.h"
#include "cppconn/resultset.h"
#include "mysql_connection.h"

struct VixData {
    std::string date;
    double vix = 0.0f;
};

enum class OptionType {
    CALL,
    PUT
};

struct OptionId {
    std::string symbol;
    std::chrono::year_month_day expiration;
    float strike = 0.0;

    [[nodiscard]] bool equals(const OptionId& other) const {
        return this->symbol == other.symbol
        && this->expiration == other.expiration
        && this->strike == other.strike;
    }

    bool operator==(const OptionId &other) const {
        return this->equals(other);
    }
};

template <>
struct std::hash<OptionId> {
    size_t operator()(const OptionId& id) const noexcept {
        // Hash the symbol
        const size_t symbolHash = std::hash<std::string>{}(id.symbol);

        // Hash the expiration date
        const auto sysDays = std::chrono::sys_days(id.expiration);
        const long epoch = sysDays.time_since_epoch().count();
        const size_t dateHash = std::hash<int>{}(epoch);

        // Hash the strike price
        uint32_t strikeAsInt;
        std::memcpy(&strikeAsInt, &id.strike, sizeof(float));
        const size_t strikeHash = std::hash<uint32_t>{}(strikeAsInt);

        // Combine the hashes
        size_t result = symbolHash;
        result = result * 31 + dateHash;
        result = result * 31 + strikeHash;

        return result;
    }
};


struct Option {
    OptionId id;
    OptionType type = OptionType::CALL;

    [[nodiscard]] bool equals(const Option& other) const {
        return this->id.equals( other.id)
        && this->type == other.type;
    }

    bool operator==(const Option &other) const {
        return this->equals(other);
    }
};

struct OptionValues {
    float bid = 0.0;
    float ask = 0.0;
    float delta = 0.0;
};

struct OptionChain {
    std::chrono::year_month_day date;
    Option option;
    OptionValues value;

    OptionChain() = default;

    static OptionChain fromResult(const sql::ResultSet* result) {
        OptionChain option;
        option.date = parseDate(result->getString("date"));
        option.option.id.symbol = result->getString("symbol");
        option.option.id.expiration = parseDate(result->getString("expiration"));
        option.option.id.strike = static_cast<float>(result->getDouble("strike"));
        option.option.type = static_cast<OptionType>(result->getInt("option_type"));
        option.value.bid = static_cast<float>(result->getDouble("bid"));
        option.value.ask = static_cast<float>(result->getDouble("ask"));
        option.value.delta = static_cast<float>(result->getDouble("delta"));

        return option;
    }
};

struct Stock {
    std::chrono::year_month_day date;
    std::string symbol;
    float open;
    float high;
    float low;
    float close;
    float volume;

    static Stock fromResult(const sql::ResultSet* result) {
        Stock stock;
        stock.date = parseDate(result->getString("date"));
        stock.symbol = result->getString("symbol");
        stock.open = static_cast<float>(result->getDouble("open"));
        stock.high = static_cast<float>(result->getDouble("high"));
        stock.low = static_cast<float>(result->getDouble("low"));
        stock.close = static_cast<float>(result->getDouble("close"));
        stock.volume = static_cast<float>(result->getDouble("volume"));

        return stock;
    }
};

struct DataStore {
    std::shared_ptr<ConnectionPool<10>> pool = std::make_shared<ConnectionPool<10>>();

    [[nodiscard]] std::chrono::year_month_day getStartDate() const;

    [[nodiscard]] std::optional<std::chrono::year_month_day> getNextDate(std::chrono::year_month_day const& prevDate) const;

    [[nodiscard]] std::vector<OptionChain> getOptionChain(std::chrono::year_month_day const& date) const;

    [[nodiscard]] std::vector<OptionChain> getOptionChain(std::chrono::year_month_day const& date, std::chrono::year_month_day const& expiration) const;

    [[nodiscard]] std::vector<OptionChain> getOptionChain(std::string const& symbol, std::chrono::year_month_day const& date) const;

    [[nodiscard]] std::vector<Stock> getStocks(std::chrono::year_month_day const& date) const;

    [[nodiscard]] std::vector<std::string> getSAP100Stocks(std::chrono::year_month_day const& date) const;

    [[nodiscard]] std::vector<VixData> getVix() const;
};
