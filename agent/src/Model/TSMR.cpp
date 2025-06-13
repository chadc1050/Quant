#include "Model/TSMR.hpp"

TSMR::TSMR(std::shared_ptr<DataStore>& data) {
    this->tsmom = std::make_shared<TSMOM>(data);
}

float TSMR::signal(const int straddle_id, const std::chrono::year_month_day date) const {
    return -1 * this->tsmom->signal(straddle_id, date);
}