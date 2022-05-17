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
    value_type CFL = 0.8, T = 0.005;

    Solver solver = Solver(nu, xlim, nx);

    // WENO-NN
    solver.init_NN("../ML_model/test_model.bin");

    solver.run(CFL, T);

    solver.save_case("data/test_weno_nx200_t1.dat");   
    return 0;
}
