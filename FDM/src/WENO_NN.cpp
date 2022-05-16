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
}

// Loads the weights of neural network
void WENO_NN::load_weights(FILE* fp){
    int dim[2];

    for(int idx_nn = 0; idx_nn < __nn_count; idx_nn++){
        fread(&dim, sizeof(int), 2, fp);
        __weights[idx_nn].set(dim[0], dim[1]);
        for (int j = 0; j < __weights[idx_nn].ny; j++){
            fread(__weights[idx_nn].v[j], sizeof(float), __weights[idx_nn].nx, fp);
        }
    }
}
