#pragma once
#include <chrono>
#include <set>

#include "Data/DataStore.hpp"
#include "Data/DateUtils.hpp"
#include "Logger/Logger.hpp"
#include "Model/Signal.hpp"

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
    std::unordered_map<OptionId, Straddle> straddles;
    Portfolio portfolio;
};

struct Stats {
    std::unordered_map<std::chrono::year_month_day, float> unrealized = {};
    std::unordered_map<std::chrono::year_month_day, std::unordered_map<OptionId, float>> positions = {};
    std::unordered_map<std::chrono::year_month_day, float> liquidity = {};

    void update(const State& state) {
        unrealized[state.date] = state.portfolio.unrealized;
        liquidity[state.date] = state.portfolio.liquidity;
        for (auto const& position : state.portfolio.positions) {
            const Straddle straddle = state.straddles.at(position.id);
            positions[state.date][position.id] = static_cast<float>(position.putContracts) * straddle.call.midpoint() + static_cast<float>(position.callContracts) * straddle.put.midpoint();
        }
    }
};

struct Config {
    int min_diversification;
    float initial_liquidity;
};

class Trader {
    public:
        explicit Trader(const std::shared_ptr<Signal> &signal, const std::shared_ptr<DataStore> &data, float liquidity);
        void start(std::chrono::year_month_day start);
    private:
        State state;
        Stats stats;
        std::shared_ptr<DataStore> data;
        std::shared_ptr<Signal> signal;
        void advance(std::chrono::year_month_day date, bool init);
        void end() const;
        void openPositions();
        void closePositions(bool force);
        void updatePositions();
        void createState(std::chrono::year_month_day date);
        void updateState(std::chrono::year_month_day date);
        std::vector<std::string> getAllowedStocks() const;
        bool shouldClosePosition(const Position& _) const;
};