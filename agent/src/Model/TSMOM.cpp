#include "Model/TSMOM.hpp"

TSMOM::TSMOM(std::shared_ptr<DataStore>& data) {
    this->data = data;
}

[[nodiscard]] float TSMOM::signal(const std::string& symbol, const std::chrono::year_month_day date) const {
    const float ret = this->data->getStraddleDerived(symbol, date).ret.ret20;

    if (ret > 0.0f) {
        return 1.0f;
    }
    if (ret < 0.0f) {
        return -1.0f;
    }

    return 0.0f;
}
