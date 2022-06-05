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
    value_type CFL = 0.8, T = 1.8;

    Solver solver = Solver(nu, xlim, nx);

    // WENO-NN
    solver.init_NN("../ML_model/test_batch/git_3_lamb_3/model_batch_82.bin");
    // test_model.bin 1e-100
    // test_model_SC1.bin 1e-1
    solver.run(CFL, T);

    solver.save_case("data/test");   
    return 0;
}
