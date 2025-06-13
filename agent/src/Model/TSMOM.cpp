#include "Model/TSMOM.hpp"

TSMOM::TSMOM(std::shared_ptr<DataStore>& data) {
    this->data = data;
}

[[nodiscard]] float TSMOM::signal(const int straddle_id, const std::chrono::year_month_day date) const {
    const StraddleDerived derived = this->data->getStraddleDerived(straddle_id, date);
    const float ret = derived.ret.ret20;

    if (ret > 0.0f) {
        return 1.0f;
    }
    if (ret < 0.0f) {
        return -1.0f;
    }

    return 0.0f;
}
