#pragma once

struct Account {
    float balance = 0.0f;
    float liquidity = 0.0f;
};

struct Trader {


    Trader() = default;

    void start() {
        while (true) {
            refresh();
        }
    }

    void refresh() {
        // Pull account balances
    }

    void scan() {

    }
}