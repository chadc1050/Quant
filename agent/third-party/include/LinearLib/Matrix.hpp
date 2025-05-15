#pragma once

#include <array>
#include <cassert>
#include <functional>
#include <initializer_list>
#include <random>
#include <type_traits>
#include <ranges>

#include "Vector.hpp"

namespace LinearLib {
    template<std::size_t R, std::size_t C, typename T>
    requires std::is_arithmetic_v<T>
    struct Matrix {
        std::array<Vector<C, T>, R> data;

        Matrix(std::initializer_list<std::initializer_list<T>> init) {
            assert(init.size() == R && "Initializer list size must match vector dimension");
            auto row = init.begin();
            for (std::size_t i = 0; i < R && row != init.end(); i++, ++row) {
                assert(row->size() == C && "Number of columns must match matrix dimensions");
                std::copy(row->begin(), row->end(), data[i].begin());
            }
        }

        // Default constructor is still needed
        Matrix() = default;

        // Iterator methods
        typename std::array<Vector<C, T>, R>::iterator begin() noexcept { return data.begin(); }
        typename std::array<Vector<C, T>, R>::const_iterator begin() const noexcept { return data.begin(); }
        typename std::array<Vector<C, T>, R>::const_iterator cbegin() const noexcept { return data.cbegin(); }

        typename std::array<Vector<C, T>, R>::iterator end() noexcept { return data.end(); }
        typename std::array<Vector<C, T>, R>::const_iterator end() const noexcept { return data.end(); }
        typename std::array<Vector<C, T>, R>::const_iterator cend() const noexcept { return data.cend(); }

        typename std::array<Vector<C, T>, R>::reverse_iterator rbegin() noexcept { return data.rbegin(); }
        typename std::array<Vector<C, T>, R>::const_reverse_iterator rbegin() const noexcept { return data.rbegin(); }
        typename std::array<Vector<C, T>, R>::const_reverse_iterator crbegin() const noexcept { return data.crbegin(); }

        typename std::array<Vector<C, T>, R>::reverse_iterator rend() noexcept { return data.rend(); }
        typename std::array<Vector<C, T>, R>::const_reverse_iterator rend() const noexcept { return data.rend(); }
        typename std::array<Vector<C, T>, R>::const_reverse_iterator crend() const noexcept { return data.crend(); }

        static Matrix identity() {
            return eye(0);
        }

        static Matrix eye(int const offset) {

            assert(offset < static_cast<int>(C) && "Offset absolute value must be less than matrix dimension");
            assert(offset > -static_cast<int>(C) && "Offset absolute value must be less than matrix dimension");

            Matrix res;

            for (std::size_t i = 0; i < R; i++) {
                for (std::size_t j = 0; j < C; j++) {
                    if (i + offset == j) {
                        res.data[i][j] = T{1};
                    } else {
                        res.data[i][j] = T{0};
                    }
                }
            }

            return res;
        }

        static Matrix zeros() {
            return uniform(T{0});
        }

        static Matrix ones() {
            return uniform(T{1});
        }

        static Matrix uniform(T const val) {
            Matrix res;

            for (std::size_t i = 0; i < R; i++) {
                for (std::size_t j = 0; j < C; j++) {
                    res.data[i][j] = val;
                }
            }

            return res;
        }

        static Matrix random(T const min, T const max, std::size_t const seed) {
            return random(min, max, std::mt19937_64(seed));
        }

        static Matrix random(T const min, T const max) {
            return random(min, max, std::random_device{}());
        }

        static Matrix random(T const min, T const max, std::mt19937_64 rng) {
            Matrix res;

            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<T> dist(min, max);
                for (std::size_t i = 0; i < R; i++) {
                    for (std::size_t j = 0; j < C; j++) {
                        res.data[i][j] = dist(rng);
                    }
                }
            } else if constexpr (std::is_floating_point_v<T>) {
                std::uniform_real_distribution<T> dist(min, max);
                for (std::size_t i = 0; i < R; i++) {
                    for (std::size_t j = 0; j < C; j++) {
                        res.data[i][j] = dist(rng);
                    }
                }
            }

            return res;
        }

        template<std::size_t SubR, std::size_t SubC>
        Matrix<SubR, SubC, T> splice(std::ranges::range auto& rows, std::ranges::range auto& cols) const {
            assert(std::ranges::distance(rows) == SubR && std::format("Row range must dimensionally cover %i rows", SubR).c_str());
            assert(std::ranges::distance(cols) == SubC && std::format("Row range must dimensionally cover %i columns", SubC).c_str());

            Matrix<SubR, SubC, T> res;

            std::size_t row = 0;
            for (auto&& r : rows) {
                if (r >= R) {
                    throw std::out_of_range("Row index out of bounds");
                }


                std::size_t col = 0;
                for (auto&& c : cols) {
                    if (c >= C) {
                        throw std::out_of_range("Column index out of bounds");
                    }

                    res.data[row][col] = data[r][c];
                    col++;
                }
                row++;
            }

            return res;
        }

        Matrix<C, R, T> transpose() const {
            Matrix<C, R, T> res;

            for (std::size_t i = 0; i < C; i++) {
                for (std::size_t j = 0; j < R; j++) {
                    res.data[i][j] = data[j][i];
                }
            }

            return res;
        }

        Matrix reshape(std::size_t const rows, std::size_t const cols) const {

            assert(rows * cols == R * C && "Reshaping matrix must have the same number of elements");

            Matrix res;

            for (std::size_t i = 0; i < rows; i++) {
                for (std::size_t j = 0; j < cols; j++) {
                    res[i][j] = data[i * cols + j];
                }
            }

            return res;
        }

        T determinant() const {

            assert(isSquare() && "Determinant is only defined for square matrices");

            // Base case 1
            if (R == 1) {
                return data[0][0];
            }

            // Base case 2
            if (R == 2) {
                return data[0][0] * data[1][1] - data[0][1] * data[1][0];
            }

            T res = T{};
            int sign = 1;

            // Create a range of indices for rows, skipping the first row
            auto rows = std::views::iota(size_t{1}, R);

            // Apply Laplace Transformation...
            for (std::size_t i = 0; i < C; i++) {

                // Create a range of indices for columns, skipping the current column
                auto cols = std::views::iota(std::size_t{0}, C) |
                            std::views::filter([i](const std::size_t col) { return col != i; });

                // Calculate determinant of submatrix
                Matrix<(R - 1 > 1 ? R - 1 : 1), (C - 1 > 1 ? C - 1 : 1), T> subMatrix =
                    splice<(R - 1 > 1 ? R - 1 : 1), (C - 1 > 1 ? C - 1 : 1)>(rows, cols);

                T subDeterminant = subMatrix.determinant();

                res += sign * data[0][i] * subDeterminant;
                sign = -sign;
            }

            return res;
        }

        [[nodiscard]] static bool isSquare() {
            return R == C;
        }

        [[nodiscard]] bool isSymmetric() const {

            if (!isSquare()) {
                return false;
            }

            for (std::size_t i = 0; i < R; i++) {
                for (std::size_t j = 0; j < C; j++) {
                    if (data[i][j] != data[j][i]) {
                        return false;
                    }
                }
            }

            return true;
        }

        void forEach(const std::function<void()>& func) {
            for (std::size_t i = 0; i < R; i++) {
                for (std::size_t j = 0; j < C; j++) {
                    func();
                }
            }
        }

        void forEach(const std::function<void(T&)>& func) {
            for (std::size_t i = 0; i < R; i++) {
                for (std::size_t j = 0; j < C; j++) {
                    func(data[i][j]);
                }
            }
        }

        void forEach(const std::function<void(T&, std::size_t, std::size_t)> func) {
            for (std::size_t i = 0; i < R; i++) {
                for (std::size_t j = 0; j < C; j++) {
                    func(data[i][j], i, j);
                }
            }
        }

        bool operator==(const Matrix& other) const {
            for (std::size_t i = 0; i < R; i++) {
                for (std::size_t j = 0; j < C; j++) {
                    if (data[i][j] != other[i][j]) {
                        return false;
                    }
                }
            }

            return true;
        }

        Vector<C, T>& operator[](std::size_t index) {
            assert(index < R && "Index out of bounds");
            return data[index];
        }

        const Vector<C, T>& operator[](std::size_t index) const {
            assert(index < R && "Index out of bounds");
            return data[index];
        }

        static Matrix add(const Matrix& lhs, const Matrix& rhs) {
            Matrix res;

            for (std::size_t i = 0; i < R; i++) {
                res[i] = lhs[i] + rhs[i];
            }

            return res;
        }

        friend Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
            return add(lhs, rhs);
        }

        Matrix operator+=(const Matrix& other) {
            *this = *this + other;
            return *this;
        }

        static Matrix subtract(const Matrix& minuend, const Matrix& subtrahend) {
            Matrix res;

            for (std::size_t i = 0; i < R; i++) {
                res[i] = minuend[i] - subtrahend[i];
            }

            return res;
        }

        friend Matrix operator-(const Matrix& minuend, const Matrix& subtrahend) {
            return subtract(minuend, subtrahend);
        }

        Matrix operator-=(const Matrix& other) {
            *this = *this - other;
            return *this;
        }


        static Matrix multiply(const Matrix& multiplicand, const Matrix& multiplier) {
            Matrix res;

            for (std::size_t i = 0; i < R; i++) {
                res[i] = multiplicand[i] * multiplier[i];
            }

            return res;
        }

        friend Matrix operator*(const Matrix& multiplicand, const Matrix& multiplier) {
            return multiply(multiplicand, multiplier);
        }

        Matrix operator*=(const Matrix& other) {
            *this = *this * other;
            return *this;
        }

        static Matrix divide(Matrix const& dividend, Matrix const& divisor) {
            Matrix res;

            for (std::size_t i = 0; i < R; i++) {
                res[i] = dividend[i] / divisor[i];
            }

            return res;
        }

        friend Matrix operator/(const Matrix& dividend, const Matrix& divisor) {
            return divide(dividend, divisor);
        }

        Matrix operator/=(const Matrix& other) {
            *this = *this / other;
            return *this;
        }

        static Matrix modulus(const Matrix& mat, const Matrix& mod) {
            Matrix res;

            for (std::size_t i = 0; i < R; i++) {
                res[i] = mat[i] % mod[i];
            }

            return res;
        }

        friend Matrix operator%(const Matrix& mat, const Matrix& mod) {
            return modulus(mat, mod);
        }

        Matrix operator%=(const Matrix& other) {
            *this = *this % other;
            return *this;
        }

        static Matrix multiply(const Matrix& mat, const T& scalar) {
            Matrix res;

            for (std::size_t i = 0; i < R; i++) {
                res[i] = mat[i] * scalar;
            }

            return res;
        }

        friend Matrix operator*(const Matrix& mat, const T& scalar) {
            return multiply(mat, scalar);
        }

        Matrix operator*=(const T& scalar) {
            *this = *this * scalar;
            return *this;
        }

        /**
         * Matrix Multiplication
         */
        template<std::size_t I>
        friend Matrix<R, I, T> operator&(const Matrix& lhs, const Matrix<C, I, T>& rhs) {

            Matrix<R, I, T> res;

            for (std::size_t i = 0; i < R; i++) {
                for (std::size_t j = 0; j < I; j++) {
                    T sum = T{};
                    for (std::size_t k = 0; k < C; k++) {
                        sum += lhs[i][k] * rhs[k][j];
                    }
                    res[i][j] = sum;
                }
            }

            return res;
        }
    };
}
