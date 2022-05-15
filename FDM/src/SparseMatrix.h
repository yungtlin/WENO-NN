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
class SP_Matrix;

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

    // Sparse matrix operations
    void mul(const SP_Matrix& A, const Vec1D& b); // c = A*b
    void mul(const Vec1D& b, const SP_Matrix& A); // c = b.T*A


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
    
    // Sparse matrix operators
    Vec1D operator*(const SP_Matrix& A); // c = b.T * A
};

// Vec1D Related Operations
value_type dot(const Vec1D& A, const Vec1D& B);

// Vec1D Related Operators
// Constant operations
Vec1D operator*(const value_type& b, const Vec1D& A);

//////////
// Node //
//////////
class LinkedList;
class ListNode{
private:
    int_type __key;
    value_type __value;
    ListNode* __right;
    ListNode* __left;

public:
    ListNode():__key(0), __value(0), __right(NULL), __left(NULL){};
    ListNode(int_type key, value_type value, 
        ListNode* left, ListNode* right ){
        __key = key;
        __value = value;
        __left = left;
        __right = right;
    };

    friend class LinkedList;
    friend class SP_Matrix;
    friend class Vec1D;
};


////////////////
// LinkedList //
////////////////
class LinkedList{
private:
    int_type __size;
    ListNode* __first;
    ListNode* __last;
    ListNode* __current;


public:
    LinkedList():__size(0), __first(NULL), __last(NULL), __current(NULL){};
    virtual ~LinkedList(){clear();};

    void assign(int_type key, value_type value); 
    void remove(int_type key);
    void clear();

    void reset_iter();
    std::tuple<int_type, value_type> get_next();

    value_type get_value(int_type key);

    void print();
    int_type get_size();

    // Operators // 
    void operator=(const LinkedList& B);
    friend class SP_Matrix;
    friend class Vec1D;
};


////////////////
// CSR_Matrix //
////////////////
class CSR_Matrix // Base type of the CS family
{
private:
    int_type __ny, __nx;
    LinkedList* __sparseTable;

public:
    CSR_Matrix(int_type ny, int_type nx); // Constructor
    CSR_Matrix(): __ny(0), __nx(0), __sparseTable(NULL){};
    ~CSR_Matrix(); // Destructor

    void input_check(int_type y, int_type x);
    void size_check(int_type ny, int_type nx);
    std::tuple<int_type, int_type> shape();
    value_type get_value(int_type y, int_type x);
    void assign(int_type y, int_type x, value_type u);

    void clear();
    void set_zeros();
    void set_eye();
    void print_value();
    
    // Operation
    void add(int_type y, int_type x, value_type u);

    // Operator
    void operator=(const CSR_Matrix& B);


    friend class Vec1D;
    friend class SP_Matrix;
};


///////////////
// Sp_Matrix //
///////////////
class SP_Matrix
{
private:
    int_type __ny, __nx;
    CSR_Matrix __csr_matrix;
    CSR_Matrix __csc_matrix;


public:
    // constructors
    SP_Matrix(int_type ny, int_type nx);
    SP_Matrix();
    // destructor
    ~SP_Matrix();
    
    
    // Functions
    std::tuple<int_type, int_type> shape();
    value_type get_value(int_type y, int_type x);
    void assign(int_type y, int_type x, value_type u);
    void add(int_type y, int_type x, value_type u);
    void set_row_zero(int_type y);

    void sparse_check();
    void size_check(int_type ny, int_type nx);

    void update_csc();
    // void update_csr();

    void clear();
    void set_zeros();
    void set_eye();
    void print_value();

    // Operation
    void addby(const SP_Matrix& A); // C += A
    void subby(const SP_Matrix& A); // C -= A
    
    void mulby(const value_type a); // C *= a
    void mul(const SP_Matrix& A, const SP_Matrix& B); // C = A * B


    // Operator
    void operator=(const SP_Matrix& B);
    
    SP_Matrix operator+(const SP_Matrix& B);
    SP_Matrix operator-(const SP_Matrix& B);
    SP_Matrix operator*(const SP_Matrix& B);
    Vec1D operator*(const Vec1D& b); // c = b.T * A
    

    SP_Matrix operator*(const value_type& a);
    friend class Vec1D;
};




