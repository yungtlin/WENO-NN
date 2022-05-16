// Cpp: 1D Euler equations solver with WENO method
//
// Author: Yung-Tien Lin (UCLA)
// Created: April, 19, 2022
// Updated: April, 19, 2022

////////////
// Header //
////////////
#include "Solver.h"

////////////
// Solver //
////////////
// Using Shu-Osher problem to initialize values on uniform grids
Solver::Solver(int_type nu, value_type* xlim, int_type nx){
    mesh.set_zeros(nu, xlim, nx);
    mesh.set_Shu_Osher();

    // set inner state 
    __nu = nu; 
    __nx = nx; 
    __gamma = mesh.get_gamma();
    __h = mesh.get_h();
    set_matrices();

    // WENO-NN related
    __is_WENO_NN = false;
}

// Starting solver from reading an existed simulation
/*
Solver::Solver(char* path){
}
*/

// Initialize //
// Allocate memories for the matrices
void Solver::set_matrices(){
    __c = Vec1D(__nx);
    __p = Vec1D(__nx);

    __U_tmp = new Vec1D[__nu];
    __f = new Vec1D[__nu];
    __Ep = new Vec1D[__nu];
    __Em = new Vec1D[__nu];

    for(int i = 0; i < __nu; i++){
        __U_tmp[i] = Vec1D(__nx);
        __f[i] = Vec1D(__nx);
        __Ep[i] = Vec1D(__nx);
        __Em[i] = Vec1D(__nx);
    }
}

// Run //
void Solver::run(value_type CFL, value_type T){
    __CFL = CFL;

    value_type t = 0; // start time
    int iteration = 0;
    value_type dt; // time step size

    while(t < T){
        dt = get_dt();
        // close to the target time
        if (T - t < dt){
            dt = T - t;
        }

        //time_FE(dt);
        time_RK3(dt);

        t += dt;
        iteration++;
        printf("Iteration: %i, t: %f \n", iteration, t);
    }
}


// Time Integration //
// Forward Euler
void Solver::time_FE(value_type dt){
    // stage 1
    update_f(mesh.U);
    for (int j = 0; j < __nu; j++){
        mesh.U[j] = mesh.U[j] + __f[j]*dt;
    }
}

// SSP Runge-Kutta
void Solver::time_RK3(value_type dt){
    // stage 1
    update_f(mesh.U);
    for (int j = 0; j < __nu; j++){
        __U_tmp[j] = mesh.U[j] + __f[j]*dt;
    }

    // stage 2
    update_f(__U_tmp);
    for (int j = 0; j < __nu; j++){
        __U_tmp[j] = 0.75*mesh.U[j] + 0.25*__U_tmp[j] + 0.25*__f[j]*dt;
    }

    // stage 3
    update_f(__U_tmp);
    for (int j = 0; j < __nu; j++){
        mesh.U[j] = 1.0/3*mesh.U[j] + 2.0/3*__U_tmp[j] + 2.0/3*__f[j]*dt;
    }

}


// Get State //
// dt = CFL * h / u_ref
value_type Solver::get_dt(){
    value_type u_ref = get_u_ref();

    value_type dt = __CFL * __h / u_ref; 
    return dt;
} 

// uref = max(|u| + c)
value_type Solver::get_u_ref(){
    update_c(mesh.U);
    value_type u_ref = 0, u, u_i; 

    // linear search
    for (int i = 0; i < __nx; i++){
        u = mesh.U[1].value[i]/mesh.U[0].value[i];
        u_i = abs(u) + __c.value[i];
        u_ref = std::max(u_ref, u_i);
    }

    return u_ref;
}


// State computation //
void Solver::update_f(Vec1D* U){
    update_Epm(U);
    WENO_flux();
    boundary_fixed();
}

// set the nearest 3 points around the boundary flux = 0   
void Solver::boundary_fixed(){
    int i_max = 3;
    for(int j = 0; j < __nu; j ++){
        for(int i = 0; i < i_max; i++){
            __f[j].value[i] = 0;
            __f[j].value[__nx-1-i] = 0;
        }
    }
}

// plus and minus direction flux
void Solver::update_Epm(Vec1D* U){
    update_c(U);

    value_type E[3];
    value_type u[3];
    value_type p, c;
    value_type alpha;
    for(int i = 0; i < __nx; i++){
        u[0] = U[0].value[i];
        u[1] = U[1].value[i];
        u[2] = U[2].value[i];
        p = __p.value[i];
        c = __c.value[i];

        E[0] = u[1];
        E[1] = u[1]*u[1]/u[0] + p;
        E[2] = (u[2] + p)*u[1]/u[0];

        // local Lax-Friedrich
        alpha = abs(u[1]/u[0]) + c;

        for(int j = 0; j < __nu; j++){
            __Ep[j].value[i] = (E[j] + alpha * u[j])/2;
            __Em[j].value[i] = (E[j] - alpha * u[j])/2;
        }
    }
 }


// Compute the acoustic speed from the current state
void Solver::update_c(Vec1D* U){
    update_p(U);

    value_type u;
    for(int i = 0; i < __nx; i++){
        u = U[1].value[i] / U[0].value[i];
        __c.value[i] = sqrt(__gamma * __p.value[i] / U[0].value[i]);
    }
}


// Compute the pressure from the current state
void Solver::update_p(Vec1D* U){
    Vec1D Ek = 0.5 * (U[1]*U[1]/U[0]);
    __p = (U[2] - Ek)*(__gamma - 1);
}

// WENO //
void Solver::WENO_flux(){
    WENO_p();
    WENO_m();
}

void Solver::WENO_p(){
    Vec1D f = Vec1D(__nx - 1);

    value_type fh_1, fh_2, fh_3;
    value_type fm2, fm1, f0, fp1, fp2;
    value_type c1[3], c2[3], c3[3];
    value_type gamma1, gamma2, gamma3;
    value_type beta1, beta2, beta3;
    value_type sigma1, sigma2, sigma3;
    value_type omega1, omega2, omega3;
    value_type epsilon = 1e-6;
    value_type a, b;

    // WENO reconstructed coefficient
    c1[0] = 1.0/3;
    c1[1] = -7.0/6;
    c1[2] = 11.0/6;
    gamma1 = 1.0/10; 

    c2[0] = -1.0/6; 
    c2[1] = 5.0/6; 
    c2[2] = 1.0/3;
    gamma2 = 3.0/5;

    c3[0] = 1.0/3; 
    c3[1] = 5.0/6;
    c3[2] = -1.0/6;
    gamma3 = 3.0/10;


    for(int j = 0; j < __nu; j++){
        // get the WENO flux
        for(int i = 2; i < __nx - 2; i++){
            fm2 = __Ep[j].value[i-2];
            fm1 = __Ep[j].value[i-1];
            f0  = __Ep[j].value[i];
            fp1 = __Ep[j].value[i+1];
            fp2 = __Ep[j].value[i+2];

            // flux reconstruction
            fh_1 = c1[0]*fm2 + c1[1]*fm1 + c1[2]*f0;
            fh_2 = c2[0]*fm1 + c2[1]*f0  + c2[2]*fp1;
            fh_3 = c3[0]*f0  + c3[1]*fp1 + c3[2]*fp2;

            // smoothness indicator
            a = (fm2 - 2*fm1 + f0);
            b = (fm2 - 4*fm1 + 3*f0);
            beta1 = 13.0/12*a*a + 1.0/4*b*b;

            a = (fm1 - 2*f0 + fp1);
            b = (fm1 - fp1);
            beta2 = 13.0/12*a*a + 1.0/4*b*b;

            a = (f0 - 2*fp1 + fp2);
            b = (3*f0 - 4*fp1 + fp2);
            beta3 = 13.0/12*a*a + 1.0/4*b*b;
        
            // nonlinear weights
            a = epsilon + beta1;
            sigma1 = gamma1/(a*a);

            a = epsilon + beta2;
            sigma2 = gamma2/(a*a);

            a = epsilon + beta3;
            sigma3 = gamma3/(a*a);

            // normalize
            a = sigma1 + sigma2 + sigma3;

            omega1 = sigma1 / a; 
            omega2 = sigma2 / a; 
            omega3 = sigma3 / a;

            f.value[i] = omega1*fh_1 + omega2*fh_2 + omega3*fh_3;
        }
        // compute the flux difference
        for(int i = 3; i < __nx - 2; i++){
            __f[j].value[i] = -(f.value[i] - f.value[i-1])/__h;
        }
    }    
}

void Solver::WENO_m(){
    Vec1D f = Vec1D(__nx - 1);

    value_type fh_1, fh_2, fh_3;
    value_type fm1, f0, fp1, fp2, fp3;
    value_type c1[3], c2[3], c3[3], c[5];
    value_type gamma1, gamma2, gamma3;
    value_type beta1, beta2, beta3;
    value_type sigma1, sigma2, sigma3;
    value_type omega1, omega2, omega3;
    value_type epsilon = 1e-6;
    value_type a, b;

    // WENO reconstructed coefficient
    c1[0] = -1.0/6;
    c1[1] = 5.0/6;
    c1[2] = 1.0/3;
    gamma1 = 3.0/10; 

    c2[0] = 1.0/3; 
    c2[1] = 5.0/6; 
    c2[2] = -1.0/6;
    gamma2 = 3.0/5;

    c3[0] = 11.0/6; 
    c3[1] = -7.0/6;
    c3[2] = 1.0/3;
    gamma3 = 1.0/10;

    
    for(int j = 0; j < __nu; j++){
        // get the WENO flux
        for(int i = 1; i < __nx - 3; i++){
            fm1 = __Em[j].value[i-1];
            f0  = __Em[j].value[i];
            fp1 = __Em[j].value[i+1];
            fp2 = __Em[j].value[i+2];
            fp3 = __Em[j].value[i+3];

            // flux reconstruction
            fh_1 = c1[0]*fm1 + c1[1]*f0  + c1[2]*fp1;
            fh_2 = c2[0]*f0  + c2[1]*fp1 + c2[2]*fp2;
            fh_3 = c3[0]*fp1 + c3[1]*fp2 + c3[2]*fp3;

            // smoothness indicator
            a = (fm1 - 2*f0 + fp1);
            b = (fm1 - 4*f0 + 3*fp1);
            beta1 = 13.0/12*a*a + 1.0/4*b*b;

            a = (f0 - 2*fp1 + fp2);
            b = (f0 - fp2);
            beta2 = 13.0/12*a*a + 1.0/4*b*b;

            a = (fp1 - 2*fp2 + fp3);
            b = (3*fp1 - 4*fp2 + fp3);
            beta3 = 13.0/12*a*a + 1.0/4*b*b;
        
            // nonlinear weights
            a = epsilon + beta1;
            sigma1 = gamma1/(a*a);

            a = epsilon + beta2;
            sigma2 = gamma2/(a*a);

            a = epsilon + beta3;
            sigma3 = gamma3/(a*a);

            // normalize
            a = sigma1 + sigma2 + sigma3;

            omega1 = sigma1 / a; 
            omega2 = sigma2 / a; 
            omega3 = sigma3 / a;

            f.value[i] = omega1*fh_1 + omega2*fh_2 + omega3*fh_3;
        }
        // compute the flux difference
        
        for(int i = 2; i < __nx - 3; i++){
            __f[j].value[i] += -(f.value[i] - f.value[i-1])/__h;
        }
        
    } 
}

// WENO-NN //
void Solver::init_NN(const char* path){
    __is_WENO_NN = true;
    __weno_nn.load_bin(path);
}

// File I/O
void Solver::save_case(const char* path){
    mesh.save_solution(path);
}



