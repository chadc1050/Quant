#include "Trader.hpp"

Trader::Trader(const float liquidity) {
    this->state.portfolio.liquidity = liquidity;
    this->data = std::make_shared<DataStore>();
};

void Trader::start(const std::chrono::year_month_day start) {
    this->state.date = start;
    createState(start);
    advance(start, true);
}

void Trader::advance(const std::chrono::year_month_day date, const bool init) {

    std::cout << "Starting new day: " << this->state.date.year() << "-" << this->state.date.month() << "-" << this->state.date.day() << std::endl;
    std::cout << "Expiration: " << this->state.exp.year() << "-" << this->state.exp.month() << "-" << this->state.exp.day() << std::endl;
    std::cout << "Liquidity: " << this->state.portfolio.liquidity << std::endl;
    std::cout << "Unrealized: " << this->state.portfolio.unrealized << std::endl;

    const auto next = data->getNextDate(date);

    if (!next.has_value()) {
        end();
        return;
    }

    if (!init) {
        updateState(date);
        updatePositions();
    }

    closePositions();

    // On expiration open new positions for next month at market open
    if (isThirdFriday(this->state.date)) {
        createState(date);
        openPositions();
    }

    advance(next.value(), false);
}

void Trader::end() {
    std::cout << "Simulation ending" << std::endl;
}

void Trader::openPositions() {

    std::cout << "Opening positions" << std::endl;

    // Scan to consist of only S&P 100 Companies (Due to diversification and high liquidity).
    // Will be for ATM straddled contracts such that call and put delta are the same weight.
    std::set<std::string> available = getAvailableStocks();

    std::vector<std::string> allowed = getAllowedStocks();

    std::vector<std::string> symbols = {};

    for (auto symbol = available.begin(); symbol != available.end(); ++symbol) {
        if (std::ranges::find(allowed, *symbol) != allowed.end()) {
            symbols.push_back(*symbol);
        }
    }

    const float allocation = this->state.portfolio.liquidity / symbols.size();

    for (auto symbol = symbols.begin(); symbol != symbols.end(); ++symbol) {

        // TODO: This should pass all the information needed to feed the model
        const float indicator = getIndicator(*symbol);

        const float positioning = indicator * allocation;

        for (auto straddle = state.straddles.begin(); straddle != state.straddles.end(); ++straddle) {
            if (straddle->first.symbol == *symbol) {
                if (indicator > 0) {
                    // Construct delta neutral allocation
                    auto&[call, put] = straddle->second;
                    const float totalDelta = call.delta - put.delta;

                    const int callUnitPrice = call.bid * 100;
                    const float callAllocation = call.delta / totalDelta * positioning;
                    const int callContracts = static_cast<int>(callAllocation / callUnitPrice);

                    const int putUnitPrice = put.bid * 100;
                    const float putAllocation = -1 * (put.delta / totalDelta) * positioning;
                    const int putContracts = static_cast<int>(putAllocation / putUnitPrice);

                    if (putContracts > 0 || callContracts > 0) {

                        const float price = callContracts * call.bid * 100 + putContracts * put.bid * 100;

                        Position position;
                        position.type = LONG;
                        position.id = straddle->first;
                        position.callContracts = callContracts;
                        position.putContracts = putContracts;
                        position.costBasis = callContracts * call.bid * 100 + putContracts * put.bid * 100;

                        this->state.portfolio.positions.push_back(position);

                        this->state.portfolio.liquidity -= price;
                        this->state.portfolio.unrealized += price;
                    }
                } else if (indicator < 0) {
                    // TODO: Implement short
                }
            }
        }
    }
}

void Trader::closePositions() {
    std::cout << "Checking for positions to close" << std::endl;
    for (auto position = state.portfolio.positions.rbegin(); position != state.portfolio.positions.rend();) {
        if (shouldClosePosition(*position)) {

            if (!state.straddles.contains(position->id)) {
                throw std::runtime_error("No straddle found for position");
            }

            // Close out the position
            const auto&[call, put] = state.straddles[position->id];


            const float callRealized = position->callContracts * call.ask * 100;

            this->state.portfolio.liquidity += callRealized;
            this->state.portfolio.unrealized -= callRealized;

            const float putRealized = position->putContracts * put.ask * 100;
            this->state.portfolio.liquidity += putRealized;
            this->state.portfolio.unrealized -= putRealized;

            const auto current = position;
            ++position;
            this->state.portfolio.positions.erase(std::next(current).base());
        } else {
            ++position;
        }
    }
}

void Trader::updatePositions() {
    float reprice = 0;
    for (auto position = state.portfolio.positions.begin(); position != state.portfolio.positions.end(); ++position) {

        if (!state.straddles.contains(position->id)) {
            throw std::runtime_error("No straddle found for position");
        }

        auto&[call, put] = this->state.straddles[position->id];
        reprice += position->callContracts * call.ask * 100;
        reprice += position->putContracts * put.ask * 100;
    }

    state.portfolio.unrealized = reprice;
}


void Trader::createState(const std::chrono::year_month_day date) {

    std::cout << "Creating market simulation state." << std::endl;

    this->state.date = date;
    this->state.exp = nextThirdFriday(date);

    std::vector<Stock> stocks = data->getStocks(this->state.date);

    this->state.stocks.clear();
    for (auto stock = stocks.begin(); stock != stocks.end(); ++stock) {
        this->state.stocks[stock->symbol] = *stock;
    }

    // Only want options that expire on the third Friday of the month
    std::vector<OptionChain> optionChains = data->getOptionChain(this->state.date, this->state.exp);

    this->state.straddles.clear();

    // NOTE: The data I am using for does not contain the open interest but ensuring positive open interest would be ideal
    for (auto chain = optionChains.begin(); chain != optionChains.end(); ++chain) {

        // Swap existing if there is an option more at the money
        if (std::optional<OptionId> existing = this->state.getOptionBySymbol(chain->option.id.symbol); existing.has_value()) {
            if (existing.value().strike != chain->option.id.strike) {
                const float open = this->state.stocks[chain->option.id.symbol].open;
                const float chainAtmRatio = open / chain->option.id.strike;
                const float existingAtmRatio = open / existing.value().strike;

                if (std::abs(1 - chainAtmRatio) < std::abs(1 - existingAtmRatio)) {

                    this->state.straddles.erase(existing.value());

                    if (chain->option.type == OptionType::CALL) {
                        this->state.straddles[chain->option.id].first = chain->value;
                    } else if (chain->option.type == OptionType::PUT) {
                        this->state.straddles[chain->option.id].second = chain->value;
                    }
                }

                continue;
            }
        }

        // Exclude non-American option price bounds.
        if (chain->value.bid == 0 || chain->value.ask <= chain->value.bid) {
            continue;
        }

        // Exclude options that are not ATM
        if (const float open = this->state.stocks[chain->option.id.symbol].open; chain->option.id.strike < 0.95 * open || chain->option.id.strike > 1.05 * open) {
            continue;
        }

        if (chain->option.type == OptionType::CALL) {
            this->state.straddles[chain->option.id].first = chain->value;
        } else if (chain->option.type == OptionType::PUT) {
            this->state.straddles[chain->option.id].second = chain->value;
        }
    }
}

void Trader::updateState(const std::chrono::year_month_day date) {
    std::cout << "Updating market simulation state." << std::endl;

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

std::set<std::string> Trader::getAvailableStocks() {
    std::set<std::string> availableTickers = {};
    for (auto chain = state.straddles.begin(); chain != state.straddles.end(); ++chain) {
        availableTickers.insert(chain->first.symbol);
    }

    return availableTickers;
}

std::vector<std::string> Trader::getAllowedStocks() const {
    return data->getSAP100Stocks(this->state.date);
}

float Trader::getIndicator(std::string symbol) {
    // Long only
    return 1.0f;
}

bool Trader::shouldClosePosition(const Position& _) const {
    // TODO: This should be improved to consider the literature for closing out an option.
    return this->state.exp == this->state.date;
}