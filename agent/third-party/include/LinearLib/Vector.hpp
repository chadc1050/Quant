#pragma once

#include <array>
#include <cassert>
#include <initializer_list>
#include <type_traits>

#include "Matrix.hpp"

namespace LinearLib {
    template<std::size_t N, typename T>
    requires std::is_arithmetic_v<T>
    struct Vector {

        std::array<T, N> data;

        Vector(std::initializer_list<T> init) {
            assert(init.size() == N && "Vector size must match vector dimension");
            std::copy(init.begin(), init.end(), data.begin());
        }

        // Default constructor is still needed
        Vector() = default;

        T magnitude() const {
            T res = 0;

            for (std::size_t i = 0; i < N; i++) {
                res += std::pow(data[i], 2);
            }

            return std::sqrt(res);
        }

        Matrix<1, N, T> asMatrix() {
            Matrix<1, N, T> res;

            for (std::size_t i = 0; i < N; i++) {
                res[0][i] = data[i];
            }

            return res;
        }

        std::array<T, N> getData() const {
            return data;
        }

        bool operator==(const Vector& other) const {
            for (std::size_t i = 0; i < N; i++) {
                if (data[i] != other.data[i]) {
                    return false;
                }
            }

            return true;
        }

        T& operator[](std::size_t index) {
            assert(index < N && "Index out of bounds");
            return data[index];
        }

        const T& operator[](std::size_t index) const {
            assert(index < N && "Index out of bounds");
            return data[index];
        }

        Vector operator+(const Vector& other) const {
            Vector res;

            for (std::size_t i = 0; i < N; i++) {
                res.data[i] = data[i] + other.data[i];
            }

            return res;
        }

        Vector operator-(const Vector& other) const {
            Vector res;

            for (std::size_t i = 0; i < N; i++) {
                res.data[i] = data[i] - other.data[i];
            }

            return res;
        }

        /**
         * Dot Product
         */
        T operator*(const Vector& other) const {
            T res = 0;

            for (std::size_t i = 0; i < N; i++) {
                res+= data[i] * other.data[i];
            }

            return res;
        }

        /**
         * Scalar Multiplication
         */
        Vector operator*(const T& other) const {
            Vector res;

            for (std::size_t i = 0; i < N; i++) {
                res[i] = data[i] * other;
            }

            return res;
        }

        Vector<3,T> operator&(const Vector<3,T>& other) const {

            Vector<3,T> res;

            res.data[0] = data[1] * other.data[2] - data[2] * other.data[1];
            res.data[1] = data[2] * other.data[0] - data[0] * other.data[2];
            res.data[2] = data[0] * other.data[1] - data[1] * other.data[0];

            return res;
        }
    };
}
