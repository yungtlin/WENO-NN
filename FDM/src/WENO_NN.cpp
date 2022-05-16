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
    }
    else{
        printf("FILE PATH: %s CANNOT BE LOADED\n", path);
        exit(EXIT_FAILURE);
    }
}

// Read number of nn and allocates memory space
void WENO_NN::load_header(FILE* fp){
    fread(&__nn_count, sizeof(int), 1, fp);
    
}


