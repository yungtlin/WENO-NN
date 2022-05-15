// Main: the main function of Shu-Osher problem solver (Shu-Osher)

///////////////
// Libraries //
///////////////
// Built-in Libs
#include <stdio.h>

// Custom Libs
#include "src/Mesh.h"
#include "src/Solver.h"


using namespace std;

int main(int argc, char *argv[])
{
    // variables
    int nu = 3, nx = 10001;
    value_type xlim[2] = {-5, 5};
    value_type gamma = 1.4;
    value_type CFL = 0.8, T = 1.8;
 
    Solver solver = Solver(nu, xlim, nx);
    solver.run(CFL, T);


    solver.save_case("data/shu-osher_nx10000_t180.dat");   
    return 0;
}