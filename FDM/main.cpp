// Main: the main function of Shu-Osher problem solver (Shu-Osher)

///////////////
// Libraries //
///////////////
// Built-in Libs
#include <stdio.h>

// Custom Libs
#include "src/Solver.h"


using namespace std;

int main(int argc, char *argv[])
{
    // variables
    int nu = 3, nx = 201;
    value_type xlim[2] = {-5, 5};
    value_type gamma = 1.4;
    value_type CFL = 0.8, T_end = 2.0, step = 0.1;

    Solver solver = Solver(nu, xlim, nx);

    // WENO-NN
    solver.init_NN("../ML_model/model_TV_med.bin");

    for(int i = 1; i < 21; i++){
        solver.run(CFL, step*i);
        solver.save_case("data/test");   
    }
    
    return 0;
}
