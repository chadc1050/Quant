#include "Data/DataStore.hpp"

[[nodiscard]] std::chrono::year_month_day DataStore::getStartDate() const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::Statement* statement = conn->createStatement();
    sql::ResultSet* result = statement->executeQuery("SELECT MIN(date) FROM `options`");

    result->next();
    const std::string start = result->getString(1);

    delete statement;
    delete result;

    return parseDate(start);
}

[[nodiscard]] std::optional<std::chrono::year_month_day> DataStore::getNextDate(std::chrono::year_month_day const& date) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT MIN(date) FROM `options` WHERE date > ?");
    prepared_statement->setString(1, formatDate(date));
    sql::ResultSet* result = prepared_statement->executeQuery();

    result->next();
    if (result->isNull(1)) {

        delete prepared_statement;
        delete result;

        return std::nullopt;
    }

    const std::string next = result->getString(1);

    delete prepared_statement;
    delete result;

    return parseDate(next);
}

[[nodiscard]] std::vector<OptionChain> DataStore::getOptionChain(std::chrono::year_month_day const& date) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT date, symbol, expiration, strike, option_type, bid, ask, delta FROM `options` WHERE date = ?");
    prepared_statement->setString(1, formatDate(date));
    sql::ResultSet* result = prepared_statement->executeQuery();

    std::vector<OptionChain> data = {};

    while (result->next()) {
        data.push_back(OptionChain::fromResult(result));
    }

    delete prepared_statement;
    delete result;

    return data;
}

[[nodiscard]] std::vector<OptionChain> DataStore::getOptionChain(std::chrono::year_month_day const& date, std::chrono::year_month_day const& expiration) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement(
        "SELECT date, symbol, expiration, strike, option_type, bid, ask, delta FROM `options` WHERE date = ? AND expiration = ?"
    );
    prepared_statement->setString(1, formatDate(date));
    prepared_statement->setString(2, formatDate(expiration));
    sql::ResultSet* result = prepared_statement->executeQuery();

    std::vector<OptionChain> data = {};

    while (result->next()) {
        data.push_back(OptionChain::fromResult(result));
    }

    delete prepared_statement;
    delete result;

    return data;
}

[[nodiscard]] std::vector<OptionChain> DataStore::getOptionChain(std::string const& symbol, std::chrono::year_month_day const& date) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT date, symbol, expiration, strike, option_type, bid, ask, delta FROM `options` WHERE symbol = ? AND date = ?");
    prepared_statement->setString(1, symbol);
    prepared_statement->setString(2, formatDate(date));
    sql::ResultSet* result = prepared_statement->executeQuery();

    std::vector<OptionChain> data(result->rowsCount());

    while (result->next()) {
        data.push_back(OptionChain::fromResult(result));
    }

    delete prepared_statement;
    delete result;

    return data;
}

[[nodiscard]] std::vector<Straddle> DataStore::getStraddles(const std::chrono::year_month_day& date, const std::chrono::year_month_day& expiration) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement(
        "SELECT * FROM straddle_view WHERE date = ? AND expiration = ?"
    );
    prepared_statement->setString(1, formatDate(date));
    prepared_statement->setString(2, formatDate(expiration));
    sql::ResultSet* result = prepared_statement->executeQuery();

    std::vector<Straddle> straddles = {};

    while (result->next()) {
        straddles.push_back(Straddle::fromResult(result));
    }

    delete prepared_statement;
    delete result;

    return straddles;
}


[[nodiscard]] std::vector<Stock> DataStore::getStocks(std::chrono::year_month_day const& date) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT * FROM stocks WHERE date = ?");
    prepared_statement->setString(1, formatDate(date));
    sql::ResultSet* result = prepared_statement->executeQuery();

    std::vector<Stock> data = {};

    while (result->next()) {
        data.push_back(Stock::fromResult(result));
    }

    delete prepared_statement;
    delete result;

    return data;
}

[[nodiscard]] std::vector<std::string> DataStore::getSAP100Stocks(std::chrono::year_month_day const& date) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT stocks FROM sap100 where date < ? ORDER BY date DESC LIMIT 1");
    prepared_statement->setString(1, formatDate(date));

    sql::ResultSet* result = prepared_statement->executeQuery();
    result->next();
    std::string stocks = result->getString(1);

    size_t pos = 0;
    std::vector<std::string> res;

    while ((pos = stocks.find(',')) != std::string::npos) {
        std::string stock = stocks.substr(0, pos);
        res.push_back(stock);
        stocks.erase(0, pos + 1);
    }

    return res;
}

[[nodiscard]] std::vector<VixData> DataStore::getVix() const {

    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::Statement* statement = conn->createStatement();
    sql::ResultSet* result = statement->executeQuery("SELECT * FROM vix ORDER BY observation_date ASC");

    std::vector<VixData> data = {};

    while (result->next()) {
        VixData vix;
        vix.date = result->getString("observation_date");
        vix.vix = static_cast<float>(result->getDouble("index_value"));
        data.push_back(vix);
    }

    delete statement;
    delete result;

    return data;
}

SimpleMovingAverages DataStore::getSimpleMovingAverages(const std::string& symbol, const std::chrono::year_month_day date) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT sma2, sma4, sma8, sma16, sma32 FROM option_straddle_derived WHERE symbol = ? AND date = ?");
    prepared_statement->setString(1, symbol);
    prepared_statement->setString(2, formatDate(date));
    sql::ResultSet* result = prepared_statement->executeQuery();

    result->next();
    const SimpleMovingAverages sma = SimpleMovingAverages::fromResult(result);

    delete prepared_statement;
    delete result;

    return sma;
}

ExpMovingAverages DataStore::getExpMovingAverages(const std::string& symbol, const std::chrono::year_month_day date) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT ema2, ema4, ema8, ema16, ema32 FROM option_straddle_derived WHERE symbol = ? AND date = ?");
    prepared_statement->setString(1, symbol);
    prepared_statement->setString(2, formatDate(date));
    sql::ResultSet* result = prepared_statement->executeQuery();

    result->next();
    const ExpMovingAverages ema = ExpMovingAverages::fromResult(result);

    delete prepared_statement;
    delete result;

    return ema;
}

StandardDeviation DataStore::getStandardDeviation(const std::string& symbol, const std::chrono::year_month_day date) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT std5, std10, std15, std20, std25, std30 FROM option_straddle_derived WHERE symbol = ? AND date = ?");
    prepared_statement->setString(1, symbol);
    prepared_statement->setString(2, formatDate(date));
    sql::ResultSet* result = prepared_statement->executeQuery();

    result->next();
    const StandardDeviation std = StandardDeviation::fromResult(result);

    delete prepared_statement;
    delete result;

    return std;
}

StraddleDerived DataStore::getStraddleDerived(const int straddle_id, const std::chrono::year_month_day date) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT * FROM option_straddle_derived WHERE straddle_id = ? AND date = ?");
    prepared_statement->setInt(1, straddle_id);
    prepared_statement->setString(2, formatDate(date));
    sql::ResultSet* result = prepared_statement->executeQuery();

    result->next();
    const StraddleDerived derived = StraddleDerived::fromResult(result);

    delete prepared_statement;
    delete result;

    return derived;
}

std::vector<StraddleDerived> DataStore::getStraddleDerivedHistory(const int straddle_id, const std::chrono::year_month_day from, const int days) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT * FROM option_straddle_derived WHERE straddle_id = ? AND date <= ? ORDER BY date DESC LIMIT ?");
    prepared_statement->setInt(1, straddle_id);
    prepared_statement->setString(2, formatDate(from));
    prepared_statement->setInt(3, days);
    sql::ResultSet* result = prepared_statement->executeQuery();

    result->next();

    std::vector<StraddleDerived> data = {};

    while (result->next()) {
        data.push_back(StraddleDerived::fromResult(result));
    }

    delete prepared_statement;
    delete result;

    return data;
}
