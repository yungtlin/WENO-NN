// Cpp: Detailed implementaion of WENO_NN
//
// Author: Yung-Tien Lin (UCLA)
// Created: May, 15, 2022
// Updated: May, 15, 2022

////////////
// Header //
////////////
#include "WENO_NN.h"

/////////////
// WENO_NN //
/////////////
// WENO_NN constructor
WENO_NN::WENO_NN(){
    __nn_count = 0;
}

// loads the trained neural networks from binary file
void WENO_NN::load_bin(const char* path){

    FILE* fp;

    fp = fopen(path, "rb");
    if(fp != NULL){
        printf("Loading NN model: %s\n", path);
        load_header(fp);
        load_weights(fp);
    }
    else{
        printf("FILE PATH: %s CANNOT BE LOADED\n", path);
        exit(EXIT_FAILURE);
    }
}

// Loads the number of nn and allocates memory space
void WENO_NN::load_header(FILE* fp){
    fread(&__nn_count, sizeof(int), 1, fp);
    __weights = new Vec2D[__nn_count];
    __outputs = new Vec1D[__nn_count];
}

// Loads the weights of neural network
void WENO_NN::load_weights(FILE* fp){
    int dim[2];

    __intput.set(5);

    for(int idx_nn = 0; idx_nn < __nn_count; idx_nn++){
        fread(&dim, sizeof(int), 2, fp);
        __weights[idx_nn].set(dim[0], dim[1]);
        __outputs[idx_nn].set(dim[1]);

        for (int j = 0; j < __weights[idx_nn].ny; j++){
            fread(__weights[idx_nn].v[j], sizeof(float), __weights[idx_nn].nx, fp);
        }
    }
}

// activation functions
void WENO_NN::activ_sigmoid(Vec1D& a){
    for (int i = 0; i < a.size; i++){
        a.value[i] = 1/(1 + exp(-a.value[i]));
    }
}

void WENO_NN::activ_linear(Vec1D& a){
}

// 
void WENO_NN::predict(value_type* C){
    load_input(C);

    // Neural Network
    // hidden 1
    __outputs[0].matmul(__intput, __weights[0]);
    activ_sigmoid(__outputs[0]);

    // hidden 2
    __outputs[1].matmul(__outputs[0], __weights[1]);
    activ_sigmoid(__outputs[1]);

    // hidden 3
    __outputs[2].matmul(__outputs[1], __weights[2]);
    activ_sigmoid(__outputs[2]);

    // output 
    __outputs[3].matmul(__outputs[2], __weights[3]);
    activ_linear(__outputs[3]);

    // delta c_tilde to C
    affine_transform(__outputs[3], C);
}


void WENO_NN::load_input(value_type* C){
    for (int i = 0; i < __intput.size; i++){
        __intput.value[i] = C[i];
    }
}

void WENO_NN::affine_transform(Vec1D& c_tilde, value_type* C){
    value_type sum_c = 0;
    value_type c_hat_s;
    int_type nx = c_tilde.size;

    for(int i = 0; i < nx; i++){
        sum_c += c_tilde.value[i];
    }

    for (int i = 0; i < nx; i++){
        c_hat_s = c_tilde.value[i] - sum_c/5;
        C[i] += c_hat_s;
    }
}

