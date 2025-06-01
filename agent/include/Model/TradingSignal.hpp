#pragma once

class TradingSignal {
    public:
        virtual float Signal() = 0;
        virtual ~TradingSignal() = default;
};
