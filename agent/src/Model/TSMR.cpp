#include "Model/TSMR.hpp"

TSMR::TSMR(std::shared_ptr<DataStore>& data) {
    this->tsmom = std::make_shared<TSMOM>(data);
}

float TSMR::signal(const std::string &symbol, const std::chrono::year_month_day date) const {
    return -1 * this->tsmom->signal(symbol, date);
}