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

#include "SparseMatrix.h"

/////////////////
// Subfunction //
/////////////////
value_type** create_2D(int_type ny, int_type nx);
int_type** create_2D_int(int_type ny, int_type nx);
void free_2D(value_type** array, int_type ny);
void free_2D(int_type** array, int_type ny);
void read_2D_value(value_type** array, FILE* fp,
    int_type ny, int_type nx);

void write_2D_value(value_type** array, FILE* fp,
    int_type ny, int_type nx);

//////////////
// Solution //
//////////////
class Solution{
private:
    int_type __nu, __nx;
    value_type* __xlim;
    value_type __gamma;

public:
    Vec1D X;
    Vec1D *U;
    

    Solution(): __nu(0), __nx(0), __gamma(1.4){};
    Solution(int_type nu, value_type* xlim, int_type nx);
    ~Solution(){
        // prevent clean empty solution
        if (__nx != 0){
            delete[] __xlim;
            delete[] U;    
        }
    }

    // solution initialization
    void mesh_uniform();
    void set_zeros(int_type nu, value_type* xlim, int_type nx);
    void set_Shu_Osher();

    // Get States
    value_type get_gamma(){return __gamma;};
    value_type get_h();

    // solution loading

    // solution saving 
    void save_solution(const char* path);
    void write_header(FILE* fp);
    void write_data(FILE* fp);

};



