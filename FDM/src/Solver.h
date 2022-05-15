// Header: Class/function declaration for 1D Euler equations
//
// Author: Yung-Tien Lin (UCLA)
// Created: April, 19, 2022
// Updated: April, 19, 2022

#pragma once

///////////////
// Libraries //
///////////////
#define ERR_CONVERGE_DEFAULT 1e-5
#define ERR_DIVERGE 1e10

#include <stdio.h>
#include <stdlib.h> 

#include "SparseMatrix.h"
#include "Mesh.h"


//////////////
// PCSolver //
//////////////
class Solver
{
private:
    int_type __nu, __nx;
    value_type __h; // mesh size (uniform)
    value_type __gamma;
    value_type __CFL;

    Vec1D __p, __c; // pressure and acoustic speed
    Vec1D *__U_tmp, *__f;
    Vec1D *__Ep, *__Em;

public:
    Solution mesh;

    Solver(int_type nu, value_type* xlim, int_type nx);
    Solver(char* path);
    ~Solver(){
        if (__nu != 0){
            delete[] __U_tmp; 
            delete[] __f;
            delete[] __Ep;
            delete[] __Em;
        }
    };

    // Initialize //
    void set_matrices();

    // Run //
    void run(value_type CFL, value_type T);

    // Time Integration //
    void time_FE(value_type dt);
    void time_RK3(value_type dt);

    // Get State //
    value_type get_dt(); // dt = CFL * h / u_ref
    value_type get_u_ref(); // u_ref = max(|u| + c)


    // State computation //
    void update_f(Vec1D* U);
    void boundary_fixed();

    void update_Epm(Vec1D* U);
    void update_c(Vec1D* U);
    void update_p(Vec1D* U);

    // WENO //
    // 5pt WENO spatial discretization
    void WENO_flux();
    void WENO_p(); // plus direction
    void WENO_m(); // minus direction

    // Saving
    void save_case(const char* path);
};


