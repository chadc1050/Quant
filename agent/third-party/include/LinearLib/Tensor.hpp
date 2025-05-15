#pragma once

#include <array>

#include "Matrix.hpp"

// TODO: This needs to be the general template for Vector and Matrix and all other dims
namespace LinearLib {
    template<std::size_t R, std::size_t C, std::size_t Z, typename T>
    requires std::is_arithmetic_v<T>
    struct Tensor {
        std::array<Matrix<R, C, T>, Z> data;

        Tensor() = default;

        // Iterator methods
        typename std::array<Matrix<R, C, T>, Z>::iterator begin() noexcept { return data.begin(); }
        typename std::array<Matrix<R, C, T>, Z>::const_iterator begin() const noexcept { return data.begin(); }
        typename std::array<Matrix<R, C, T>, Z>::const_iterator cbegin() const noexcept { return data.cbegin(); }

        typename std::array<Matrix<R, C, T>, Z>::iterator end() noexcept { return data.end(); }
        typename std::array<Matrix<R, C, T>, Z>::const_iterator end() const noexcept { return data.end(); }
        typename std::array<Matrix<R, C, T>, Z>::const_iterator cend() const noexcept { return data.cend(); }

        typename std::array<Matrix<R, C, T>, Z>::reverse_iterator rbegin() noexcept { return data.rbegin(); }
        typename std::array<Matrix<R, C, T>, Z>::const_reverse_iterator rbegin() const noexcept { return data.rbegin(); }
        typename std::array<Matrix<R, C, T>, Z>::const_reverse_iterator crbegin() const noexcept { return data.crbegin(); }

        typename std::array<Matrix<R, C, T>, Z>::reverse_iterator rend() noexcept { return data.rend(); }
        typename std::array<Matrix<R, C, T>, Z>::const_reverse_iterator rend() const noexcept { return data.rend(); }
        typename std::array<Matrix<R, C, T>, Z>::const_reverse_iterator crend() const noexcept { return data.crend(); }

        static Tensor identity() {
            Tensor res;

            for (std::size_t i = 0; i < Z; i++) {
                res[i] = Matrix<R, C, T>::identity();
            }

            return res;
        }

        static Tensor zeros() {
            return uniform(0);
        }

        static Tensor ones() {
            return uniform(1);
        }

        static Tensor uniform(const T& val) {
            Tensor res;

            for (std::size_t i = 0; i < Z; i++) {
                res[i] = Matrix<R, C, T>::uniform(val);
            }

            return res;
        }

        static Tensor random(T const min, T const max, std::size_t const seed) {
            return random(min, max, std::mt19937_64(seed));
        }

        static Tensor random(T const min, T const max) {
            return random(min, max, std::random_device{}());
        }

        static Tensor random(T const min, T const max, std::mt19937_64 rng) {
            Tensor res;

            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<T> dist(min, max);
                for (std::size_t i = 0; i < Z; i++) {
                    for (std::size_t j = 0; j < R; j++) {
                        for (std::size_t k = 0; k < C; k++) {
                            res.data[i][j][k] = dist(rng);
                        }
                    }
                }
            } else if constexpr (std::is_floating_point_v<T>) {
                std::uniform_real_distribution<T> dist(min, max);
                for (std::size_t i = 0; i < Z; i++) {
                    for (std::size_t j = 0; j < R; j++) {
                        for (std::size_t k = 0; k < C; k++) {
                            res.data[i][j][k] = dist(rng);
                        }
                    }
                }
            }

            return res;
        }

        static bool isCube() {
            return R == C && C == Z;
        }

        bool operator==(const Tensor& other) const {
            for (std::size_t i = 0; i < Z; i++) {
                if (data[i] != other[i]) {
                    return false;
                }
            }

            return true;
        }

        Matrix<R, C, T>& operator[](std::size_t index) {
            assert(index < Z && "Index out of bounds");
            return data[index];
        }

        const Matrix<R, C, T>& operator[](std::size_t index) const {
            assert(index < Z && "Index out of bounds");
            return data[index];
        }

        static Tensor add(const Tensor& lhs, const Tensor& rhs) {
            Tensor res;

            for (std::size_t i = 0; i < Z; i++) {
                res[i] = lhs[i] + rhs[i];
            }

            return res;
        }

        friend Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
            return add(lhs, rhs);
        }

        Tensor operator+=(const Tensor& other) {
            *this = *this + other;
            return *this;
        }

        static Tensor subtract(const Tensor& minuend, const Tensor& subtrahend) {
            Tensor res;

            for (std::size_t i = 0; i < Z; i++) {
                res[i] = minuend[i] - subtrahend[i];
            }

            return res;
        }

        friend Tensor operator-(const Tensor& minuend, const Tensor& subtrahend) {
            return subtract(minuend, subtrahend);
        }

        Tensor operator-=(const Tensor& other) {
            *this = *this + other;
            return *this;
        }

        static Tensor multiply(const Tensor& multiplicand, const Tensor& multiplier) {
            Tensor res;

            for (std::size_t i = 0; i < Z; i++) {
                res[i] = multiplicand[i] * multiplier[i];
            }

            return res;
        }

        friend Tensor operator*(const Tensor& multiplicand, const Tensor& multiplier) {
            return multiply(multiplicand, multiplier);
        }

        Tensor operator*=(const Tensor& other) {
            *this = *this * other;
            return *this;
        }

        static Tensor divide(const Tensor& dividend, const Tensor& divisor) {
            Tensor res;

            for (std::size_t i = 0; i < Z; i++) {
                res[i] = dividend[i] / divisor[i];
            }

            return res;
        }

        friend Tensor operator/(const Tensor& dividend, const Tensor& divisor) {
            return divide(dividend, divisor);
        }

        Tensor operator/=(const Tensor& other) {
            *this = *this / other;
            return *this;
        }

        static Tensor modulus(const Tensor& tensor, const Tensor& mod) {
            Tensor res;

            for (std::size_t i = 0; i < Z; i++) {
                res[i] = tensor[i] % mod[i];
            }

            return res;
        }

        friend Tensor operator%(const Tensor& tensor, const Tensor& mod) {
            return modulus(tensor, mod);
        }

        Tensor operator%=(const Tensor& other) {
            *this = *this % other;
            return *this;
        }
    };
}
