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

public:
    WENO_NN();
    ~WENO_NN(){
        if (__nn_count > 0){
            printf("remove...\n");
        }
    };

    // Loading Bin
    void load_bin(const char* path);
    void load_header(FILE* fp);
    //value_type predict();

};

