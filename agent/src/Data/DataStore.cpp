#include "Data/DataStore.hpp"

[[nodiscard]] std::chrono::year_month_day DataStore::getStartDate() const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::Statement* statement = conn->createStatement();
    sql::ResultSet* result = statement->executeQuery("SELECT MIN(date) FROM option_chain");

    result->next();
    const std::string start = result->getString(1);

    delete statement;
    delete result;

    return parseDate(start);
}

[[nodiscard]] std::optional<std::chrono::year_month_day> DataStore::getNextDate(std::chrono::year_month_day const& date) const {
    const std::shared_ptr<sql::Connection> conn = pool->get();

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT MIN(date) FROM option_chain WHERE date > ?");
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

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT date, symbol, expiration, strike, option_type, bid, ask, delta FROM option_chain WHERE date = ?");
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
        "SELECT date, symbol, expiration, strike, option_type, bid, ask, delta FROM option_chain WHERE date = ? AND expiration = ?"
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

    sql::PreparedStatement* prepared_statement = conn->prepareStatement("SELECT date, symbol, expiration, strike, option_type, bid, ask, delta FROM option_chain WHERE symbol = ? AND date = ?");
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

    while ((pos = stocks.find(",")) != std::string::npos) {
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
