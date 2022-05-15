// Cpp: Detailed implementaion of mesh data managing
//
// Author: Yung-Tien Lin (UCLA)
// Created: June, 1, 2021
// Updated: June, 8, 2021

////////////
// Header //
////////////
#include "Mesh.h"

/////////////////
// Subfunction //
/////////////////
value_type** create_2D(int_type ny, int_type nx){
    value_type** array;
    array = (value_type**) malloc(sizeof(value_type*) * ny);

    for(int i = 0; i < ny; i++){
        array[i] = (value_type*) malloc(sizeof(value_type)*nx);
    }

    return array;
}

int_type** create_2D_int(int_type ny, int_type nx){
    int_type** array;
    array = (int_type**) malloc(sizeof(int_type*) * ny);

    for(int i = 0; i < ny; i++){
        array[i] = (int_type*) malloc(sizeof(int_type)*nx);
        for(int j = 0; j < nx; j++){
            array[i][j] = -1;
        }
    }

    return array;
}

void free_2D(value_type** array, int_type ny){
    if(array != NULL){
        for(int j = 0; j < ny; j++){
            if(array[j] != NULL){
                free(array[j]);    
            }
        }
        free(array);
    }
}

void free_2D(int_type** array, int_type ny){
    if(array != NULL){
        for(int j = 0; j < ny; j++){
            if(array[j] != NULL){
                free(array[j]);    
            }
        }
        free(array);
    }
}

void read_2D_value(value_type** array, FILE* fp,
    int_type ny, int_type nx){

    for(int j = 0; j < ny; j++){
        fread(array[j], sizeof(value_type), nx, fp);
    }
}

void write_2D_value(value_type** array, FILE* fp,
    int_type ny, int_type nx){

    for(int j = 0; j < ny; j++){
        fwrite(array[j], sizeof(value_type), nx, fp);
    }
}

//////////////
// Solution //
//////////////
Solution::Solution(int_type nu, value_type* xlim, int_type nx){
    set_zeros(nu, xlim, nx);
}    

void Solution::set_zeros(int_type nu, value_type* xlim, int_type nx){
    // allocate state vectors
    __nu = nu;
    U = new Vec1D[nu];
    for(int i = 0; i < __nu; i++){
        U[i] = Vec1D(nx);
    }

    // allocate mesh
    __xlim = new value_type[2];
    __xlim[0] = xlim[0];
    __xlim[1] = xlim[1];
    __nx = nx;
    X = Vec1D(nx);

    // set uniform mesh
    mesh_uniform();
}    

// set uniform mesh for X
void Solution::mesh_uniform(){

    value_type l = __xlim[1] - __xlim[0];
    value_type x;

    for(int i = 0; i < __nx; i++){
        x = l/(__nx - 1) * i + __xlim[0];
        X.value[i] = x;
    }
}


void Solution::set_Shu_Osher(){
    __gamma = 1.4;
    value_type x_shock = -4;
    value_type rho_L = 3.857143, u_L = 2.629369, p_L = 10.3333;
    value_type rho_R = 1.0, u_R = 0, p_R = 1.0;    
    value_type epsilon = 0.2, kappa = 5;

    value_type x, rho, u, p, e;
    for(int i = 0; i < __nx; i++){
        x = X.value[i];

        // left condition
        if (x < x_shock){
            rho = rho_L;
            u = u_L;
            p = p_L;
        }
        // right condition
        else{
            rho = rho_R + epsilon*sin(kappa*x);
            u = u_R;
            p = p_R;
        }

        // get states from primitive variables
        e = 0.5*rho*u*u + p/(__gamma - 1);
        U[0].value[i] = rho;
        U[1].value[i] = rho*u;
        U[2].value[i] = e;
    }
}

// Get States
value_type Solution::get_h(){
    value_type h; 

    value_type l = __xlim[1] - __xlim[0];
    h = l/(__nx - 1);

    return h;
}



void Solution::save_solution(const char* path){
    printf("Saving file: %s\n", path);

    FILE* fp;

    fp = fopen(path, "wb");
    if(fp != NULL){
        write_header(fp);
        write_data(fp);
    }
    else{
        printf("FILE PATH: %s CANNOT BE CREATED\n", path);
        exit(EXIT_FAILURE);
    }
}

void Solution::write_header(FILE* fp){
    int_type version = CURRENT_VERSION;
    fwrite(&version, sizeof(int_type), 1, fp);
    fwrite(&__gamma, sizeof(value_type), 1, fp);
}
void Solution::write_data(FILE* fp){
    // write dimensions
    fwrite(&__nu, sizeof(int_type), 1, fp);
    fwrite(&__nx, sizeof(value_type), 1, fp);

    // write X
    fwrite(X.value, sizeof(value_type), __nx, fp);
    for(int i = 0; i < __nu; i++){
        fwrite(U[i].value, sizeof(value_type), __nx, fp);    
    }
}

