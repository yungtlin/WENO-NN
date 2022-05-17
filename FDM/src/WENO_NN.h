// Header: Class/function declaration for mesh data managing
//
// Author: Yung-Tien Lin (UCLA)
// Created: June, 1, 2021
// Updated: June, 8, 2021

#pragma once

// Version
#define CURRENT_VERSION 1


///////////////
// Libraries //
///////////////
#include <math.h>
#include <stdio.h>

#include "SparseMatrix.h"

//////////////
// PCSolver //
//////////////
class WENO_NN
{
private:
    int __nn_count;
    Vec2D *__weights;
    Vec1D *__outputs;
    Vec1D __intput;

public:
    WENO_NN();
    ~WENO_NN(){
        if (__nn_count > 0){
            delete[] __weights;
            delete[] __outputs;
        }
    };

    // Loading Bin
    void load_bin(const char* path);
    void load_header(FILE* fp);
    void load_weights(FILE* fp);

    // activation functions
    void activ_sigmoid(Vec1D& a);
    void activ_linear(Vec1D& a);

    // predict 
    void predict(value_type* C);
    // loads input C to input Vec1D
    void load_input(value_type* C);
    void affine_transform(Vec1D& output, value_type* C);
};

