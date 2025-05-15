#pragma once

#include "mysql_connection.h"
#include <cppconn/driver.h>

#include "cppconn/resultset.h"
#include "cppconn/statement.h"

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
            std::cout << "Error loading driver: " << e.what() << std::endl;
            throw;
        }
    }

    [[nodiscard]] std::shared_ptr<sql::Connection> init() const {
        std::shared_ptr<sql::Connection> conn(driver->connect("tcp://192.168.1.189:3306", "root", "password"));
        conn->setSchema("financial_data");
        return conn;
    }

    std::shared_ptr<sql::Connection> get() {
        for (auto it = connections.begin(); it != connections.end(); ++it) {
            if (!(*it)->isClosed()) {
                return std::move(*it);
            }
        }
        return nullptr;
    }
};

struct VixData {
    std::string date;
    double vix = 0.0f;
};

struct Data {
    std::shared_ptr<ConnectionPool<10>> pool = std::make_shared<ConnectionPool<10>>();

    [[nodiscard]] std::vector<VixData> getVixData() const {

        const std::shared_ptr<sql::Connection> conn = pool->get();

        std::shared_ptr<sql::Statement> statement(conn->createStatement());

        sql::ResultSet *result = statement->executeQuery("SELECT * FROM vix ORDER BY observation_date ASC");

        std::vector<VixData> data = {};

        while (result->next()) {
            VixData vix;
            vix.date = result->getString("observation_date");
            vix.vix = result->getDouble("index_value");
            data.push_back(vix);
        }

        return data;
    }

};
