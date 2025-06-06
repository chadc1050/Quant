#include "Model/MACDMR.hpp"

MACDMR::MACDMR(std::shared_ptr<DataStore>& data) {
    this->macd = std::make_shared<MACD>(data);
}

float MACDMR::signal(const std::string &symbol, const std::chrono::year_month_day date) const {
    return -1 * this->macd->signal(symbol, date);
}