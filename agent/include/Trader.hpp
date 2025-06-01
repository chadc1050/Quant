#pragma once
#include <chrono>
#include <set>

#include "Data/DataStore.hpp"
#include "Data/DateUtils.hpp"

using Straddle = std::pair<OptionValues, OptionValues>;
using Straddles = std::unordered_map<OptionId, Straddle>;

enum PositionType {
    SHORT,
    LONG
};

struct Position {
    PositionType type = LONG;
    OptionId id;
    uint32_t callContracts = 0;
    uint32_t putContracts = 0;
    float costBasis = 0.0f;
};

struct Portfolio {
    std::vector<Position> positions;
    float liquidity = 0.0f;
    float unrealized = 0.0f;
};

struct State {
    std::chrono::year_month_day date;
    std::chrono::year_month_day exp;
    std::unordered_map<std::string, Stock> stocks;
    Straddles straddles;
    Portfolio portfolio;

    std::optional<OptionId> getOptionBySymbol(const std::string &symbol) {
        for (auto straddle = this->straddles.begin(); straddle != this->straddles.end(); ++straddle) {
            if (straddle->first.symbol == symbol) {
                return straddle->first;
            }
        }

        return std::nullopt;
    }
};

class Trader {
    public:
        explicit Trader(const float liquidity);
        void start(const std::chrono::year_month_day start);
    private:
        State state;
        std::shared_ptr<DataStore> data;
        void advance(const std::chrono::year_month_day date, const bool init);
        void end();
        void openPositions();
        void closePositions();
        void createState(const std::chrono::year_month_day date);
        void updateState(const std::chrono::year_month_day date);
        std::set<std::string> getAvailableStocks();
        std::vector<std::string> getAllowedStocks() const;
        float getIndicator(std::string symbol);
        bool shouldClosePosition(const Position& _) const;
};