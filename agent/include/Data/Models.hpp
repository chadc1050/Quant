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

    std::partial_ordering operator <=>(const OptionId &other) const {
        if (symbol != other.symbol) {
            return symbol <=> other.symbol;
        }

        if (expiration != other.expiration) {
            return expiration <=> other.expiration;
        }

        return strike <=> other.strike;
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

    [[nodiscard]] float midpoint() const {
        return (bid + ask) / 2.0f;
    }
};

struct OptionChain {
    std::chrono::year_month_day date;
    Option option;
    OptionValues value;

    static OptionChain fromResult(const sql::ResultSet* result) {
        return OptionChain{
            parseDate(result->getString("date")),
            Option{
                OptionId{
                    result->getString("symbol"),
                    parseDate(result->getString("expiration")),
                    static_cast<float>(result->getDouble("strike"))
                },
                static_cast<OptionType>(result->getInt("option_type"))
            },
            OptionValues{
                static_cast<float>(result->getDouble("bid")),
                static_cast<float>(result->getDouble("ask")),
                static_cast<float>(result->getDouble("delta"))
            }
        };
    }
};

struct Straddle {
    int straddle_id;
    OptionId id;
    OptionValues call;
    OptionValues put;

    static Straddle fromResult(const sql::ResultSet* result) {
        return Straddle{
            result->getInt("option_straddle_id"),
            OptionId{
                result->getString("symbol"),
                parseDate(result->getString("expiration")),
                static_cast<float>(result->getDouble("strike"))
            },
            OptionValues{
            static_cast<float>(result->getDouble("call_bid")),
                static_cast<float>(result->getDouble("call_ask")),
                static_cast<float>(result->getDouble("call_delta"))
            },
            OptionValues{
                static_cast<float>(result->getDouble("put_bid")),
                static_cast<float>(result->getDouble("put_ask")),
                static_cast<float>(result->getDouble("put_delta"))
            }
        };
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
        return Stock{
            parseDate(result->getString("date")),
            result->getString("symbol"),
            static_cast<float>(result->getDouble("open")),
            static_cast<float>(result->getDouble("high")),
            static_cast<float>(result->getDouble("low")),
            static_cast<float>(result->getDouble("close")),
            static_cast<float>(result->getDouble("volume"))
        };
    }
};

struct SimpleMovingAverages {
    float sma2;
    float sma4;
    float sma8;
    float sma16;
    float sma32;

    static SimpleMovingAverages fromResult(const sql::ResultSet* result) {
        return SimpleMovingAverages{
            static_cast<float>(result->getDouble("sma2")),
            static_cast<float>(result->getDouble("sma4")),
            static_cast<float>(result->getDouble("sma8")),
            static_cast<float>(result->getDouble("sma16")),
            static_cast<float>(result->getDouble("sma32"))
        };
    }
};

struct ExpMovingAverages {
    float ema2;
    float ema4;
    float ema8;
    float ema16;
    float ema32;

    static ExpMovingAverages fromResult(const sql::ResultSet* result) {
        return ExpMovingAverages{
            static_cast<float>(result->getDouble("ema2")),
            static_cast<float>(result->getDouble("ema4")),
            static_cast<float>(result->getDouble("ema8")),
            static_cast<float>(result->getDouble("ema16")),
            static_cast<float>(result->getDouble("ema32"))
        };
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
        return StandardDeviation{
        static_cast<float>(result->getDouble("std5")),
            static_cast<float>(result->getDouble("std10")),
            static_cast<float>(result->getDouble("std15")),
            static_cast<float>(result->getDouble("std20")),
            static_cast<float>(result->getDouble("std25")),
            static_cast<float>(result->getDouble("std30"))
        };
    }
};

struct Return {
    float ret5;
    float ret10;
    float ret15;
    float ret20;
    float ret25;
    float ret30;

    static Return fromResult(const sql::ResultSet* result) {
        return Return{
            static_cast<float>(result->getDouble("ret5")),
            static_cast<float>(result->getDouble("ret10")),
            static_cast<float>(result->getDouble("ret15")),
            static_cast<float>(result->getDouble("ret20")),
            static_cast<float>(result->getDouble("ret25")),
            static_cast<float>(result->getDouble("ret30"))
        };
    }
};

struct StraddleDerived {
    std::chrono::year_month_day date;
    SimpleMovingAverages sma;
    ExpMovingAverages ema;
    StandardDeviation std;
    Return ret;

    static StraddleDerived fromResult(const sql::ResultSet* result) {
        return StraddleDerived{
            parseDate(result->getString("date")),
            SimpleMovingAverages::fromResult(result),
            ExpMovingAverages::fromResult(result),
            StandardDeviation::fromResult(result),
            Return::fromResult(result)
        };
    }
};