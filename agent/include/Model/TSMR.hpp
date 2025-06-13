#pragma once

#include "Signal.hpp"
#include "TSMOM.hpp"
#include "Data/DataStore.hpp"

class TSMR final : public Signal {
public:
    explicit TSMR(std::shared_ptr<DataStore>& data);
    [[nodiscard]] float signal(const int straddle_id, std::chrono::year_month_day date) const override;

private:
    std::shared_ptr<TSMOM> tsmom;
};
