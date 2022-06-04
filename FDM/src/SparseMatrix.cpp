// Cpp: Detailed implementaion of the sparse matrix operations
//
// Author: Yung-Tien Lin (UCLA)
// Created: June, 1, 2021
// Updated: June, 8, 2021

////////////
// Header //
////////////
#include "SparseMatrix.h"

///////////
// Vec1D //
///////////
// Constructor
Vec1D::Vec1D(int_type size_in){
    size = size_in;
    value = (value_type*) malloc(sizeof(value_type)*size);
    
    for(int i = 0; i < size; i++){
        value[i] = 0;
    }
}


// Allocate 2D array
void Vec1D::set(int_type size_in){
    size = size_in;
    value = (value_type*) malloc(sizeof(value_type)*size);
    
    for(int i = 0; i < size; i++){
        value[i] = 0;
    }
}


void Vec1D::size_check(int_type size_in){
    if (size != size_in){
        perror("Vec1D size not match\n");
        exit (EXIT_FAILURE);
    }
}

void Vec1D::print_value(){
    printf("Vec1D size: %li\n", size);
    for(int i = 0; i < size; i++){
        printf(SP_PRINT_FORMAT, value[i]);
        printf("\n");
    }
    printf("\n");
}

// Operations //
// Elementwise operations 
// Addition (C = A + B)
void Vec1D::add(const Vec1D& A, const Vec1D& B){
    // Check Dimension
    if ((size != A.size) || (size != B.size)){
        perror("Vec1D addition size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] = A.value[i] + B.value[i];
    }
}

// Subtraction (C = A - B)
void Vec1D::sub(const Vec1D& A, const Vec1D& B){
    // Check Dimension
    if ((size != A.size) || (size != B.size)){
        perror("Vec1D subraction size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] = A.value[i] - B.value[i];
    }
}

// Multiplication (C = A * B)
void Vec1D::mul(const Vec1D& A, const Vec1D& B){
    if ((size != A.size) || (size != B.size)){
        perror("Vec1D multiplication size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] = A.value[i] * B.value[i];
    }
}

// Division (C = A / B)
void Vec1D::div(const Vec1D& A, const Vec1D& B){ 
    // Check Dimension
    if ((size != A.size) || (size != B.size)){
        perror("Vec1D division size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] = A.value[i] / B.value[i];
    }
}

// Directed operations
// Added by (C += A)
void Vec1D::addby(const Vec1D& A){
    // Check Dimension
    if (size != A.size){
        perror("Vec1D addition size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] += A.value[i];
    }
}

// Subtacted by (C -= A)
void Vec1D::subby(const Vec1D& A){
    // Check Dimension
    if (size != A.size){
        perror("Vec1D subtraction size not match");
        exit (EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++){
        value[i] -= A.value[i];
    }
}

// Constant operations
// Multiplication (C = A * b)
void Vec1D::mul(const Vec1D& A, const value_type& b){
    size_check(A.size);

    for(int j = 0; j < size; j++){
        this->value[j] = A.value[j] * b;
    }
}

// Division (C = A / b)
void Vec1D::div(const Vec1D& A, const value_type& b){
    size_check(A.size);

    for(int j = 0; j < size; j++){
        this->value[j] = A.value[j] / b;
    }
}


// Operators //
// Assignment (c = b)
void Vec1D::operator=(const Vec1D& B){
    if (size == 0){
        size = B.size;
        value = (value_type*) malloc(sizeof(value_type)*size);
    } else if (size != B.size){
        perror("Vec1D assignment size not match");
    }

    for (int i = 0; i < size; i++){
        value[i] = B.value[i];
    }
}

// Elementwise operators
// c = a + b
Vec1D Vec1D::operator+(const Vec1D& B){
    int_type nx = size;
    Vec1D C(nx);
    C.add(*this, B);
    return C;
}

// c = a - b
Vec1D Vec1D::operator-(const Vec1D& B){
    int_type nx = size;
    Vec1D C(nx);
    C.sub(*this, B);
    return C;
}


// c = a * b
Vec1D Vec1D::operator*(const Vec1D& B){
    int_type nx = size;
    Vec1D C(nx);
    C.mul(*this, B);
    return C;
}

// c = a / b
Vec1D Vec1D::operator/(const Vec1D& b){
    Vec1D c(size);

    c.div(*this, b);

    return c;
}

// Constant operators
// C = A * b
Vec1D Vec1D::operator*(const value_type& b){
    Vec1D C(size);

    C.mul(*this, b);

    return C;
}

// C = A / b
Vec1D Vec1D::operator/(const value_type& b){
    Vec1D C(size);

    C.div(*this, b);

    return C;
}


void Vec1D::matmul(Vec1D& a, Vec2D& B){
    value_type sum;

    for(int i = 0; i < B.nx; i++){
        sum = 0;
        for(int j = 0; j < B.ny; j++){
            sum += a.value[j]*B.v[j][i];
        }
        value[i] = sum;
    }
}

// Vec1D Related Operations
value_type dot(const Vec1D& A, const Vec1D& B){
    if (A.size != B.size){
        perror("Inner product size not match");
    }
    value_type sum = 0;
    for (int i = 0; i < A.size; i++){
        sum += A.value[i] * B.value[i];
    }
    return sum;
}

// Vec1D Related Operators
// (C = b * A)
Vec1D operator*(const value_type& b, const Vec1D& A){
    Vec1D C(A.size);

    C.mul(A, b);

    return C;
}


///////////
// Vec2D //
///////////
// Allocate 2D array
void Vec2D::set(int ny_in, int nx_in){
    ny = ny_in;
    nx = nx_in;

    v = (float**) malloc(ny*sizeof(float*));

    for (int i=0; i < ny; i++){
        v[i] = (float*) malloc(nx*sizeof(float));
    }
}


void Vec2D::print_value(){
    printf("Vec2D size: %i %i\n", ny, nx);
    for (int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            printf(SP_PRINT_FORMAT, v[j][i]);
        }
        printf("\n");        
    }
    printf("\n");
}


