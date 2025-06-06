#pragma once
#include "DateUtils.hpp"
#include "cppconn/resultset.h"

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

template<>
struct std::hash<std::chrono::year_month_day> {
    size_t operator()(const std::chrono::year_month_day& ymd) const noexcept {
        const auto y = static_cast<int>(ymd.year());
        const auto m = static_cast<unsigned>(ymd.month());
        const auto d = static_cast<unsigned>(ymd.day());
        return hash<int>()(y) ^ hash<unsigned>()(m) << 1 ^ hash<unsigned>()(d) << 2;
    }
};

template <>
struct std::hash<OptionId> {
    size_t operator()(const OptionId& id) const noexcept {
        // Hash the symbol
        const size_t symbolHash = std::hash<std::string>{}(id.symbol);

        // Hash the expiration date
        const size_t dateHash = hash<std::chrono::year_month_day>{}(id.expiration);

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

    float midpoint() const {
        return (bid + ask) / 2.0f;
    }
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

struct SimpleMovingAverages {
    float sma2;
    float sma4;
    float sma8;
    float sma16;
    float sma32;

    static SimpleMovingAverages fromResult(const sql::ResultSet* result) {
        SimpleMovingAverages sma;
        sma.sma2 = static_cast<float>(result->getDouble("sma2"));
        sma.sma4 = static_cast<float>(result->getDouble("sma4"));
        sma.sma8 = static_cast<float>(result->getDouble("sma8"));
        sma.sma16 = static_cast<float>(result->getDouble("sma16"));
        sma.sma32 = static_cast<float>(result->getDouble("sma32"));
        return sma;
    }
};

struct ExpMovingAverages {
    float ema2;
    float ema4;
    float ema8;
    float ema16;
    float ema32;

    static ExpMovingAverages fromResult(const sql::ResultSet* result) {
        ExpMovingAverages ema;
        ema.ema2 = static_cast<float>(result->getDouble("ema2"));
        ema.ema4 = static_cast<float>(result->getDouble("ema4"));
        ema.ema8 = static_cast<float>(result->getDouble("ema8"));
        ema.ema16 = static_cast<float>(result->getDouble("ema16"));
        ema.ema32 = static_cast<float>(result->getDouble("ema32"));
        return ema;
    }
};

struct StandardDeviation {
    float std5;
    float std10;
    float std15;
    float std20;
    float std25;
    float std30;

    static StandardDeviation fromResult(const sql::ResultSet* result) {
        StandardDeviation std;
        std.std5 = static_cast<float>(result->getDouble("std5"));
        std.std10 = static_cast<float>(result->getDouble("std10"));
        std.std15 = static_cast<float>(result->getDouble("std15"));
        std.std20 = static_cast<float>(result->getDouble("std20"));
        std.std25 = static_cast<float>(result->getDouble("std25"));
        std.std30 = static_cast<float>(result->getDouble("std30"));
        return std;
    }
};

struct StraddleDerived {
    std::chrono::year_month_day date;
    SimpleMovingAverages sma;
    ExpMovingAverages ema;
    StandardDeviation std;

    static StraddleDerived fromResult(const sql::ResultSet* result) {
        StraddleDerived straddle;
        straddle.date = parseDate(result->getString("date"));
        straddle.sma = SimpleMovingAverages::fromResult(result);
        straddle.ema = ExpMovingAverages::fromResult(result);
        straddle.std = StandardDeviation::fromResult(result);
        return straddle;
    }
};