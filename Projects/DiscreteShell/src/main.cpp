#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <igl/readOBJ.h>
#include "../include/DiscreteShell.h"
#include "../include/App.h"

int main()
{
    DiscreteShell discrete_shell;

    discrete_shell.initializeFromFile("../../../Projects/DiscreteShell/data/grid.obj");
    // discrete_shell.initializeFromFile("../../../Projects/DiscreteShell/data/grid_double_refined.obj");
    // discrete_shell.initializeFromFile("../../../Projects/Dis/creteShell/data/grid_refined_mesh.obj");

    // discrete_shell.testIsotropicStretch();
    // discrete_shell.testVerticalDirectionStretch();
    // discrete_shell.testHorizontalDirectionStretch();

    // int static_solve_step = 0;
    // bool finished = discrete_shell.advanceOneStep(static_solve_step++);
    // while(!finished) finished = discrete_shell.advanceOneStep(static_solve_step++);
    
    App<DiscreteShell> app(discrete_shell);

    app.initializeScene();
    app.run();
    // std::cout << "Homogenised target tensor: \n" << discrete_shell.computeHomogenisedTargetTensorinWindow() << std::endl; 
    // auto F = discrete_shell.computeHomogenisedTargetTensorinWindow().block(0,0,2,2);
    // Eigen::Matrix2d Estrain = 0.5*(F.transpose()*F - Eigen::Matrix2d::Identity());
    // T k_s = discrete_shell.E * discrete_shell.thickness / (1.0 - discrete_shell.nu * discrete_shell.nu);
    // auto S = k_s * ((1-discrete_shell.nu)*2*Estrain + discrete_shell.nu*2*Estrain.trace()* Eigen::Matrix2d::Identity());
    // std::cout << "Homogenised stress tensor: \n" << discrete_shell.computeHomogenisedStressTensorinWindow() << std::endl;
    // auto C1 = (F*S*F.transpose())/F.determinant();
    // std::cout << "Homogenised stress tensor (det): \n" << C1 << std::endl;


    return 0;
}
