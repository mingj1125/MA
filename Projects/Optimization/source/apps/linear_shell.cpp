#include <iostream>
#include "../../include/Scene.h"
#include "../../include/LinearShell.h"
#include "../../include/Visualization.h"
#include "../../include/OptimizationProblem.h"

int main(){

    LinearShell sim;
    Scene scene(sim);
    // std::string mesh_name = "irregular_mesh_good";
    std::string mesh_name = "grid_double_refined";
    // std::string mesh_name = "sun_mesh_line_clean";
    scene.buildSceneFromMesh(mesh_name);

    Visualization vis(&scene);
    vis.initializeScene(false);
    vis.run();

    std::vector<Vector3a> sample_locs;
    // sample_locs.push_back({0.76, 0.76, 0});
    sample_locs.push_back({0.50, 0.64, 0});
    // sample_locs.push_back({0.24, 0.76, 0});
    sample_locs.push_back({0.36, 0.50, 0});
    sample_locs.push_back({0.50, 0.50, 0});
    sample_locs.push_back({0.64, 0.50, 0});
    // sample_locs.push_back({0.76, 0.24, 0});
    sample_locs.push_back({0.50, 0.36, 0});
    // sample_locs.push_back({0.24, 0.24, 0});
    std::vector<Vector6a> Cs;
    // Cs.push_back({1.33e7,  1.09e7, 0,  1.33e7, 0,  0});
    // Cs.push_back({1.33e7,  1.09e7, 0,  1.32e7, 0,  0});
    // Cs.push_back({1.33e7,  1.09e7, 0,  1.33e7, 0,  0});
    // // Cs.push_back({1.33e7,  1.09e7, 0,  1.33e7, 0,  0});
    // // Cs.push_back({1.33e7,  1.09e7, 0,  1.32e7, 0,  0});
    // // Cs.push_back({1.33e7,  1.09e7, 0,  1.33e7, 0,  0});
    // Cs.push_back({7.61e6,  6.26e6, 0,  7.61e6, 0,  0});
    // Cs.push_back({7.60e6,  6.26e6, 0,  7.59e6, 0,  0});
    // Cs.push_back({7.61e6,  6.26e6, 0,  7.61e6, 0,  0});

    Cs.push_back({8.05543e+06, 6.69588e+06,     12568.6, 8.01659e+06,     35480.3,      595273}); 

    Cs.push_back({8.02812e+06, 6.63201e+06,    -12555.6,  8.0592e+06,    -50146.3,      662460}); 
    Cs.push_back({7.32168e+06, 6.08005e+06,     3003.15, 7.35626e+06,   -19952.8,      598525}); 
    Cs.push_back({8.04244e+06, 6.66733e+06,    -10503.8, 8.13097e+06,    -37756.2,      667873});

    Cs.push_back({8.05773e+06, 6.69831e+06,     14570.3, 8.01796e+06,     35772.4,      594949}); 

    OptimizationProblem p(&scene, "../../../Projects/Optimization/optimization_output/shell_" + mesh_name);//, "../../../Projects/Optimization/optimization_output/" + mesh_name + "_radii_debug.dat");
    std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStiffnessTensor>(sample_locs, Cs);
    // std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateStiffnessTensorRelationship>(sample_locs);
    p.objective_energies.push_back(e); 
    // p.TestOptimizationGradient();
    // p.TestOptimizationSensitivity();
    // if(!p.Optimize()) std::cout << "\n Gradient not converged to the set criterion \n";
    // if(!p.OptimizeGD()) std::cout << "\n Gradient not converged to the set criterion \n";
    // if(!p.OptimizeGDFD()) std::cout << "\n Gradient not converged to the set criterion \n";
    // if(!p.OptimizeFD()) std::cout << "\n Gradient not converged to the set criterion \n";
    // p.ShowEnergyLandscape();

    return 0;
}