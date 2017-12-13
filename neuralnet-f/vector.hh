/**
 * \file
 * \brief  Vector class (specialization of Matrix).
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

#include "matrix.hh"

template<typename T, uint rows>
class Vector : public Matrix<T, rows, 1> {

public:
    const T &operator()(uint row) const { return Matrix<T, rows, 1>::operator()(row, 1); }
          T &operator()(uint row)       { return Matrix<T, rows, 1>::operator()(row, 1); }

    Vector()                            : Matrix<T,rows,1>()   { }
    Vector(std::initializer_list<T> il) : Matrix<T,rows,1>(il) { }

    virtual ~Vector() = default;
};