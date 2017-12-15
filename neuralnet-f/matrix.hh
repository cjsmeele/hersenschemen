/**
 * \file
 * \brief  Matrix class.
 * \author Chris Smeele
 *
 * Copyright (c) 2016, Chris Smeele
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#pragma once

#include <initializer_list>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <type_traits>

#ifdef NDEBUG
#define MATRIX_NDEBUG 1
#endif

template<typename T1, uint rows, uint cols>
class Matrix {
    static_assert(std::is_arithmetic<T1>::value,
                  "Matrix type must be arithmetic");
    static_assert(rows >= 1 && cols >= 1,
                  "A Matrix must have a positive non-zero amount of rows and columns");
protected:
    T1 elems[rows][cols] { };

public:
    constexpr const T1 &operator()(uint row, uint col) const {
        #ifndef MATRIX_NDEBUG
        if (!row || !col || row > rows || col > cols)
            throw std::logic_error("Matrix index out of bounds");
        #endif
        return elems[row-1][col-1];
    }
    constexpr T1 &operator()(uint row, uint col) {
        #ifndef MATRIX_NDEBUG
        if (!row || !col || row > rows || col > cols)
            throw std::logic_error("Matrix index out of bounds");
        #endif
        return elems[row-1][col-1];
    }

    /**
     * \brief Negate matrices.
     */
    constexpr Matrix<T1, rows, cols> operator-() const {
        Matrix<T1, rows, cols> c;

        for (uint row = 1; row <= rows; row++) {
            for (uint col = 1; col <= cols; col++)
                c(row, col) = -this->operator()(row, col);
        }

        return c;
    }

    /**
     * \brief Transpose matrices.
     */
    constexpr Matrix<T1, cols, rows> T() const {
        Matrix<T1, cols, rows> c;

        for (uint row = 1; row <= rows; row++) {
            for (uint col = 1; col <= cols; col++)
                c(col, row) = this->operator()(row, col);
        }

        return c;
    }

    /**
     * \brief Multiply a matrix with a scalar.
     */
    constexpr Matrix<T1, rows, cols> operator*(T1 n) const {
        Matrix<T1, rows, cols> c;

        for (uint row = 1; row <= rows; row++) {
            for (uint col = 1; col <= cols; col++) {
                c(row, col) = this->operator()(row, col) * n;
            }
        }

        return c;
    }
    /**
     * \brief Multiply a matrix with a scalar. Assign the result.
     */
    Matrix<T1, rows, cols> &operator*=(T1 n) {
        for (uint row = 1; row <= rows; row++) {
            for (uint col = 1; col <= cols; col++)
                this->operator()(row, col) *= n;
        }

        return *this;
    }

    /**
     * \brief Add matrices.
     */
    constexpr Matrix<T1, rows, cols> operator+(Matrix<T1, rows, cols> b) const {
        Matrix<T1, rows, cols> c;

        for (uint row = 1; row <= rows; row++) {
            for (uint col = 1; col <= cols; col++)
                c(row, col) = this->operator()(row, col) + b(row, col);
        }

        return c;
    }

    /**
     * \brief Add matrices. Assign the result
     */
    Matrix<T1, rows, cols> &operator+=(Matrix<T1, rows, cols> b) {
        for (uint row = 1; row <= rows; row++) {
            for (uint col = 1; col <= cols; col++)
                this->operator()(row, col) += b(row, col);
        }

        return *this;
    }

    /**
     * \brief Subtract matrices.
     */
    constexpr Matrix<T1, rows, cols> operator-(Matrix<T1, rows, cols> b) const {
        // Perhaps I'm being a bit too lazy.
        return this->operator+(-b);
    }

    /**
     * \brief Subtract matrices. Assign the result.
     */
    Matrix<T1, rows, cols> &operator-=(Matrix<T1, rows, cols> b) {
        return this->operator+=(-b);
    }

    /**
     * \brief Dot product.
     */
    template<uint bCols>
    constexpr Matrix<T1, rows, bCols> dot(const Matrix<T1, cols, bCols> &b) const {

        Matrix<T1, rows, bCols> c;

        for (uint cRow = 1; cRow <= rows; cRow++) {
            for (uint cCol = 1; cCol <= bCols; cCol++) {
                T1 sum = 0;
                for (uint i = 1; i <= cols; i++)
                    sum += this->operator()(cRow, i) * b(i, cCol);
                c(cRow, cCol) = sum;
            }
        }

        return c;
    }

    /**
     * \brief Multiply matrices with each other.
     */
    constexpr Matrix<T1, rows, cols> operator*(const Matrix<T1, rows, cols> &b) const {
        Matrix<T1, rows, cols> m;
        for (uint r = 1; r <= rows; ++r)
            for (uint c = 1; c <= cols; ++c)
                m(r,c) = (*this)(r,c) * b(r,c);
        return m;
    }

    /**
     * \brief Multiply matrices with each other. Assign the result.
     *
     * This is only valid for square matrices.
     */
    //typename std::enable_if<(rows==cols),Matrix<T1,rows,rows>>::type *
    Matrix<T1,rows,cols> &operator*=(const Matrix<T1,cols,rows> &b) {
        static_assert(rows == cols, "Cannot assign non-square matrix multiplication result");
        *this = this->operator*(b);
        return *this;
    }

    constexpr double det() {
        // TODO.
        return 0;
    }

    constexpr Matrix<T1, rows, cols> invert() {
        Matrix<T1, rows, cols> b;
        // TODO.
        return b;
    }

    template<typename F>
    constexpr auto map(const F &f) const {

        Matrix<T1, rows, cols> b;

        for (uint r = 1; r <= rows; ++r)
            for (uint c = 1; c <= cols; ++c)
                b(r,c) = f(this->operator()(r, c));

        return b;
    }

    template<typename F>
    constexpr auto &mip(const F &f) {
        // Map In Place.

        for (uint r = 1; r <= rows; ++r) {
            for (uint c = 1; c <= cols; ++c) {
                auto &x = this->operator()(r, c);
                x = f(x);
            }
        }

        return *this;
    }

    /**
     * \brief Get an identity matrix.
     */
    constexpr static Matrix<T1, rows, cols> identity() {
        static_assert(rows == cols, "Cannot create identity for non-square matrix");
        Matrix<T1, rows, cols> id = { };
        for (uint i = 1; i <= rows; i++)
            id(i, i) = 1;

        return id;
    }

    constexpr Matrix() = default;

    constexpr Matrix(std::initializer_list<T1> il) {
        #ifndef MATRIX_NDEBUG
        if (il.size() != rows * cols)
            throw std::logic_error("Matrix initializer Dim mismatch");
        #endif

        uint row = 1;
        uint col = 1;

        for (T1 x : il) {
            this->operator()(row, col++) = x;
            if (col > cols) {
                row++;
                col = 1;
            }
        }
    }

    Matrix<T1, rows, cols> &operator=(const Matrix<T1, rows, cols> &b) {
        for (uint row = 1; row <= rows; row++) {
            for (uint col = 1; col <= cols; col++)
                this->operator()(row, col) = b(row, col);
        }
        return *this;
    }

    ~Matrix() = default;
};

template<uint rows, uint cols>
using Matrixf = Matrix<float, rows, cols>;

template<uint rows, uint cols>
using Matrixd = Matrix<double, rows, cols>;

template<typename M1, typename M2>
constexpr auto dot(const M1 &a, const M2 &b) {
    return a.dot(b);
}

template<typename T1, uint rows, uint cols>
constexpr auto operator*(T1 a, const Matrix<T1,rows,cols> &m) {
    return m * a;
}


#ifdef MATRIX_WANT_STREAMOPS

#include <ostream>

/**
 * Dump a matrix to an output stream.
 */
template<typename T1, uint rows, uint cols>
std::ostream &operator<<(std::ostream &stream, const Matrix<T1, rows, cols> &m) {
    char buffer[32];

    for (uint row = 1; row <= rows; row++) {
        if (row > 1)
            stream << " [ ";
        else
            stream << "[[ ";

        for (uint col = 1; col <= cols; col++) {
            if (std::is_floating_point<T1>::value)
                snprintf(buffer, 32, "%4.1lf", (double)m(row, col));
            else
                snprintf(buffer, 32, "%2ld", (intmax_t)m(row, col));
            stream << buffer;

            if (col < cols)
                stream << ", ";
        }

        if (row < rows)
            stream << " ],\n";
        else
            stream << " ]]\n";
    }

    return stream;
}

#endif /* MATRIX_WANT_STREAMOPS */
