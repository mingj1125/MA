#include <iostream>
#include "../../include/Scene.h"
#include "../../include/MassSpring.h"
#include "../../include/Visualization.h"
#include "../../include/OptimizationProblem.h"

int main(){

    MassSpring sim;
    Scene scene(sim);
    // std::string mesh_name = "grid_double_refined";
    std::string mesh_name = "irregular_mesh_good";
    scene.buildSceneFromMesh(mesh_name);

    Visualization vis(&scene);
    vis.initializeScene(true);
    vis.run();

    std::vector<Vector3a> sample_locs;
    sample_locs.push_back({0.76, 0.76, 0});
    sample_locs.push_back({0.50, 0.76, 0});
    sample_locs.push_back({0.24, 0.76, 0});
    sample_locs.push_back({0.76, 0.50, 0});
    sample_locs.push_back({0.50, 0.50, 0});
    sample_locs.push_back({0.24, 0.50, 0});
    sample_locs.push_back({0.76, 0.24, 0});
    sample_locs.push_back({0.50, 0.24, 0});
    sample_locs.push_back({0.24, 0.24, 0});
    std::vector<Vector6a> Cs;
    Cs.push_back({919546,  380372, 310.317,  919546, 310.317,  757525});

    OptimizationProblem p(&scene, "../../../Projects/Optimization/optimization_output/" + mesh_name);//, "../../../Projects/Optimization/optimization_output/" + mesh_name + "_radii_debug.dat");
    // std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStiffnessTensor>(sample_locs, Cs);
    std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateStiffnessTensorRelationship>(sample_locs);
    p.objective_energies.push_back(e);
    // p.TestOptimizationGradient();
    // p.TestOptimizationSensitivity();
    // if(!p.Optimize()) std::cout << "\n Gradient not converged to the set criterion \n";
    // if(!p.OptimizeGD()) std::cout << "\n Gradient not converged to the set criterion \n";
    // if(!p.OptimizeGDFD()) std::cout << "\n Gradient not converged to the set criterion \n";
    // if(!p.OptimizeFD()) std::cout << "\n Gradient not converged to the set criterion \n";

    return 0;
}