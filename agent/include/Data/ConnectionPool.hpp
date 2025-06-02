#pragma once

#include "Logger/Logger.hpp"

#include <cppconn/driver.h>

template<std::size_t N>
struct ConnectionPool {

    sql::Driver *driver;
    std::array<std::shared_ptr<sql::Connection>, N> connections;

    ConnectionPool() {
        try {
            driver = get_driver_instance();

            for (uint i = 0; i < N; i++) {
                connections[i] = std::move(init());
            }
        } catch (sql::SQLException &e) {
            error(std::format("Error loading driver: {}", e.what()));
            throw;
        }
    }

    ~ConnectionPool() {
        for (auto conn = connections.begin(); conn != connections.end(); ++conn) {
            conn->get()->close();
        }
    }

    [[nodiscard]] std::shared_ptr<sql::Connection> init() const {
        auto conn = std::shared_ptr<sql::Connection>(driver->connect("tcp://192.168.1.189:3306", "root", "password"));
        conn->setSchema("financial_data");
        return conn;
    }

    std::shared_ptr<sql::Connection> get() {
        for (auto conn = connections.begin(); conn != connections.end(); ++conn) {
            if (!conn->get()->isClosed()) {
                return *conn;
            }
        }
        return nullptr;
    }
};
