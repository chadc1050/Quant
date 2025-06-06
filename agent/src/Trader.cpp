#include "Trader.hpp"

#include <ranges>

Trader::Trader(const std::shared_ptr<Signal>& signal, const std::shared_ptr<DataStore>& data, const float liquidity) {
    this->stats = {};
    this->state.portfolio.liquidity = liquidity;
    this->data = data;
    this->signal = signal;
};

void Trader::start(const std::chrono::year_month_day start) {
    this->state.date = start;
    if (!isThirdFriday(this->state.date)) {
        createState(start);
    }
    advance(start, true);
}

void Trader::advance(const std::chrono::year_month_day date, const bool init) {
    info("============================");
    info(std::format("Date: {}-{}-{}",  this->state.date.month(), this->state.date.day(), this->state.date.year()));
    info(std::format("Next Expiration Date: {}-{}-{}", this->state.exp.month(), this->state.exp.day(), this->state.exp.year()));
    info(std::format("Starting Liquidity: {}", this->state.portfolio.liquidity));
    info(std::format("Starting Unrealized: {}", this->state.portfolio.unrealized));
    info(std::format("Starting Open Positions: {}", this->state.portfolio.positions.size()));

    const auto next = data->getNextDate(date);
    if (!next.has_value()) {
        end();
        return;
    }

    if (!init) {
        updateState(date);
        updatePositions();
    }

    closePositions(false);

    // On expiration open new positions for next month at market open
    if (next.value() > nextThirdFriday(this->state.date)) {
        warn("Missing expiration date, early closing positions!");

        closePositions(true);
        createState(next.value());
        openPositions();

    } else if (isThirdFriday(this->state.date)) {
        createState(date);
        openPositions();
    }

    stats.update(state);

    info("End of Date.");
    info(std::format("Ending Liquidity: {}", this->state.portfolio.liquidity));
    info(std::format("Ending Unrealized: {}", this->state.portfolio.unrealized));
    info(std::format("Ending Open Positions: {}", this->state.portfolio.positions.size()));

    advance(next.value(), false);
}

void Trader::end() const {
    info("============================");
    info("Simulation Complete");
    info(std::format("Ending Liquidity: {}", this->state.portfolio.liquidity));
    info(std::format("Ending Unrealized: {}", this->state.portfolio.unrealized));
    info(std::format("Ending Open Positions: {}", this->state.portfolio.positions.size()));
}

void Trader::openPositions() {

    debug("Opening positions...");

    // Scan to consist of only S&P 100 Companies (Due to diversification and high liquidity).
    // Will be for ATM straddled contracts such that call and put delta are the same weight.
    const std::set<std::string, std::less<>> available = getAvailableStocks();

    std::vector<std::string> allowed = getAllowedStocks();

    std::vector<std::string> symbols = {};

    for (const auto & symbol : available) {
        if (std::ranges::find(allowed, symbol) != allowed.end()) {
            symbols.push_back(symbol);
        }
    }

    const float allocation = this->state.portfolio.liquidity / symbols.size();

    if (symbols.size() < 15) {
        warn(std::format("Low diversification in positions available ({})! Skipping!", symbols.size()));
        return;
    }

    for (std::string& symbol : symbols) {

        // TODO: This should pass all the information needed to feed the model
        const float indicator = signal->signal(symbol, this->state.date);

        debug(std::format("Signal strength: {}", indicator));

        if (indicator == 0.0f) {
            continue;
        }

        const float positioning = indicator * allocation;

        for (auto &[id, options] : state.straddles) {
            if (id.symbol == symbol) {
                // Construct delta neutral allocation
                auto&[call, put] = options;
                const float totalDelta = call.delta - put.delta;

                const float callPrice = call.midpoint() * 100;

                const float callAllocation = call.delta / totalDelta * positioning;
                const int callContracts = static_cast<int>(callAllocation / callPrice);

                const float putPrice = put.midpoint() * 100;

                const float putAllocation = -1 * (put.delta / totalDelta) * positioning;
                const int putContracts = static_cast<int>(putAllocation / putPrice);

                if (const float price = callContracts * callPrice + putContracts * putPrice; price > 0) {

                    Position position;
                    position.id = id;
                    position.callContracts = callContracts;
                    position.putContracts = putContracts;
                    position.costBasis = price;

                    if (indicator > 0) {
                        position.type = LONG;
                        this->state.portfolio.liquidity -= price;
                        this->state.portfolio.unrealized += price;
                    } else {
                        position.type = SHORT;
                        this->state.portfolio.liquidity += price;
                        this->state.portfolio.unrealized -= price;
                    }

                    this->state.portfolio.positions.push_back(position);
                }
            }
        }
    }

    debug(std::format("Opened {} positions", this->state.portfolio.positions.size()));
}

void Trader::closePositions(const bool force) {
    if (force) {
        warn("Force closing positions!");
    } else {
        debug("Checking for positions to close...");
    }
    int nClosed = 0;
    for (auto position = state.portfolio.positions.rbegin(); position != state.portfolio.positions.rend();) {
        if (force || shouldClosePosition(*position)) {

            if (!state.straddles.contains(position->id)) {
                throw std::runtime_error(std::format("No straddle found for position {}", position->id.symbol));
            }

            // Close out the position
            const auto&[call, put] = state.straddles[position->id];

            const float price = position->callContracts * call.midpoint() * 100 + position->putContracts * put.midpoint() * 100;

            if (position->type == LONG) {
                this->state.portfolio.liquidity += price;
                this->state.portfolio.unrealized -= price;
            } else if (position->type == SHORT) {
                this->state.portfolio.liquidity -= price;
                this->state.portfolio.unrealized += price;
            }

            const auto current = position;
            ++position;
            this->state.portfolio.positions.erase(std::next(current).base());
            nClosed++;
        } else {
            ++position;
        }
    }
    if (nClosed > 0) {
        debug(std::format("Closed {} positions", nClosed));
    }
}

void Trader::updatePositions() {
    float reprice = 0;
    for (auto & position : state.portfolio.positions) {

        if (!state.straddles.contains(position.id)) {
            throw std::runtime_error(std::format("No straddle found for position {}", position.id.symbol));
        }

        auto const&[call, put] = this->state.straddles[position.id];
        reprice += position.callContracts * call.midpoint() * 100;
        reprice += position.putContracts * put.midpoint() * 100;
    }

    state.portfolio.unrealized = reprice;
}


void Trader::createState(const std::chrono::year_month_day date) {

    debug("Creating market simulation state...");

    this->state.date = date;
    this->state.exp = nextThirdFriday(date);

    const std::vector<Stock> stocks = data->getStocks(this->state.date);

    this->state.stocks.clear();
    for (const Stock& stock: stocks) {
        this->state.stocks[stock.symbol] = stock;
    }

    // Only want options that expire on the third Friday of the month
    std::vector<OptionChain> optionChains = data->getOptionChain(this->state.date, this->state.exp);

    this->state.straddles.clear();

    Straddles straddles = data->getStraddles(this->state.date, this->state.exp);

    std::unordered_map<std::string, float> strikes = {};

    // NOTE: The data I am using for does not contain the open interest but ensuring positive open interest would be ideal
    for (auto straddle = straddles.begin(); straddle != straddles.end();) {

        const auto&[call, put] = straddle->second;
        auto id = straddle->first;
        // Swap existing if there is an option more at the money
        if (strikes.contains(id.symbol)) {
            if (const auto existing = strikes[id.symbol]; existing != id.strike) {
                const float open = this->state.stocks[id.symbol].open;
                const float chainAtmRatio = open / id.strike;
                const float existingAtmRatio = open / existing;

                if (std::abs(1 - chainAtmRatio) < std::abs(1 - existingAtmRatio)) {
                    straddles.erase(straddle);
                    ++straddle;
                    continue;
                }

                // Exclude non-American option price bounds.
                if (call.bid == 0 || put.bid == 0 || call.ask <= call.bid || put.ask <= put.bid) {
                    straddles.erase(straddle);
                    ++straddle;
                    continue;
                }

                straddles.erase(OptionId{id.symbol, this->state.exp, existing});
                continue;
            }
        }

        // Exclude non-American option price bounds.
        if (call.bid == 0 || put.bid == 0 || call.ask <= call.bid || put.ask <= put.bid) {
            ++straddle;
            straddles.erase(straddle);
            continue;
        }

        // Exclude options that are not ATM
        if (const float open = this->state.stocks[id.symbol].open; id.strike < 0.95 * open || id.strike > 1.05 * open) {
            ++straddle;
            straddles.erase(straddle);
            continue;
        }

        strikes[id.symbol] = id.strike;
    }
}

void Trader::updateState(const std::chrono::year_month_day date) {
    debug("Updating market simulation state...");

    this->state.date = date;

    std::vector<Stock> stocks = data->getStocks(this->state.date);

    this->state.stocks.clear();
    for (auto stock = stocks.begin(); stock != stocks.end(); ++stock) {
        this->state.stocks[stock->symbol] = *stock;
    }

    std::vector<OptionChain> optionChains = data->getOptionChain(this->state.date, this->state.exp);

    for (auto chain = optionChains.begin(); chain != optionChains.end(); ++chain) {
        if (chain->option.type == OptionType::CALL) {
            this->state.straddles[chain->option.id].first = chain->value;
        } else if (chain->option.type == OptionType::PUT) {
            this->state.straddles[chain->option.id].second = chain->value;
        }
    }
}

std::set<std::string, std::less<>> Trader::getAvailableStocks() {
    std::set<std::string, std::less<>> availableTickers = {};
    for (const auto &id: state.straddles | std::views::keys) {
        availableTickers.insert(id.symbol);
    }

    return availableTickers;
}

std::vector<std::string> Trader::getAllowedStocks() const {
    return data->getSAP100Stocks(this->state.date);
}

bool Trader::shouldClosePosition(const Position& _) const {
    // TODO: This should be improved to consider the literature for closing out an option.
    return this->state.exp == this->state.date;
}