// Header: Class/function declaration for the sparse matrix operation
//
// Author: Yung-Tien Lin (UCLA)
// Created: June, 1, 2021
// Updated: June, 8, 2021

#pragma once
////////////
// Define //
////////////
#define LINKEDLIST_OMIT_VALUE 0
#define SP_PRINT_FORMAT "%3.3e "

typedef double value_type;
typedef long int int_type;


///////////////
// Libraries //
///////////////
#include <tuple>
#include <stdio.h>
#include <stdlib.h> 

///////////
// Vec1D //
///////////
class Vec1D
{
public:
    // Attribute
    value_type* value;
    int_type size;

    // Constructors
    Vec1D():size(0), value(NULL){};
    Vec1D(int_type size);      
    // Destructor
    ~Vec1D(){
        if(value != NULL){
            free(value);
        }
    };

    void size_check(int_type size);    
    void print_value();

    // Operations // 
    // Elementwise operations 
    void add(const Vec1D& A, const Vec1D& B); // C = A + B
    void sub(const Vec1D& A, const Vec1D& B); // C = A - B
    void mul(const Vec1D& A, const Vec1D& B); // C = A * B
    void div(const Vec1D& A, const Vec1D& B); // C = A / B

    // Directed operations
    void addby(const Vec1D& A); // C += A;
    void subby(const Vec1D& A); // C -= A;

    // Constant operations
    void mul(const Vec1D& a, const value_type& b); // C = A * b
    void div(const Vec1D& a, const value_type& b); // C = A / b

    // Operators
    void operator=(const Vec1D& B); //c = b
    // Elementwise operators
    Vec1D operator+(const Vec1D& B); // c = a + b
    Vec1D operator-(const Vec1D& B); // c = a - b
    Vec1D operator*(const Vec1D& B); // c = a * b
    Vec1D operator/(const Vec1D& b); // c = a / b

    // Constant operators
    Vec1D operator*(const value_type& b); // c = a * b
    Vec1D operator/(const value_type& b); // c = a * b

};

// Vec1D Related Operations
value_type dot(const Vec1D& A, const Vec1D& B);

// Vec1D Related Operators
// Constant operations
Vec1D operator*(const value_type& b, const Vec1D& A);
