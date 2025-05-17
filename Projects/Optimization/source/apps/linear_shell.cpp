#include <iostream>
#include "../../include/Scene.h"
#include "../../include/LinearShell.h"
#include "../../include/Visualization.h"
#include "../../include/OptimizationProblem.h"

int main(){

    LinearShell sim;
    Scene scene(sim);
    std::string mesh_name = "grid_double_refined";
    scene.buildSceneFromMesh(mesh_name);

    // Visualization vis(&scene);
    // vis.initializeScene(false);
    // vis.run();

    std::vector<Vector3a> sample_locs;
    sample_locs.push_back({0.76, 0.76, 0});
    sample_locs.push_back({0.50, 0.76, 0});
    sample_locs.push_back({0.24, 0.76, 0});
    // // sample_locs.push_back({0.76, 0.50, 0});
    // sample_locs.push_back({0.50, 0.52, 0});
    // // sample_locs.push_back({0.24, 0.50, 0});
    sample_locs.push_back({0.76, 0.24, 0});
    sample_locs.push_back({0.50, 0.24, 0});
    sample_locs.push_back({0.24, 0.24, 0});
    std::vector<Vector6a> Cs;
    Cs.push_back({1.33e7,  1.09e7, 0,  1.33e7, 0,  0});
    Cs.push_back({1.33e7,  1.09e7, 0,  1.32e7, 0,  0});
    Cs.push_back({1.33e7,  1.09e7, 0,  1.33e7, 0,  0});
    // Cs.push_back({1.33e7,  1.09e7, 0,  1.33e7, 0,  0});
    // Cs.push_back({1.33e7,  1.09e7, 0,  1.32e7, 0,  0});
    // Cs.push_back({1.33e7,  1.09e7, 0,  1.33e7, 0,  0});
    Cs.push_back({7.61e6,  6.26e6, 0,  7.61e6, 0,  0});
    Cs.push_back({7.60e6,  6.26e6, 0,  7.59e6, 0,  0});
    Cs.push_back({7.61e6,  6.26e6, 0,  7.61e6, 0,  0});

    OptimizationProblem p(&scene, "../../../Projects/Optimization/optimization_output/shell_" + mesh_name);//, "../../../Projects/Optimization/optimization_output/" + mesh_name + "_radii_debug.dat");
    std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStiffnessTensor>(sample_locs, Cs);
    // std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateStiffnessTensorRelationship>(sample_locs);
    p.objective_energies.push_back(e);
    // p.TestOptimizationGradient();
    // p.TestOptimizationSensitivity();
    // if(!p.Optimize()) std::cout << "\n Gradient not converged to the set criterion \n";
    if(!p.OptimizeGD()) std::cout << "\n Gradient not converged to the set criterion \n";
    // p.ShowEnergyLandscape();

    return 0;
}