#include <iostream>
#include "../../include/Scene.h"
#include "../../include/MassSpring.h"
#include "../../include/Visualization.h"
#include "../../include/OptimizationProblem.h"

int main(){

    MassSpring sim;
    Scene scene(sim);
    std::string mesh_name = "grid_double_refined";
    std::string filename = "../../../Projects/Optimization/data/"+mesh_name+".obj";
    scene.buildSceneFromMesh(filename);
    
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

    // std::vector<Vector3a> directions;
    // for(int i = 0; i < scene.num_directions; ++i) {
    //     AScalar angle = i*2*M_PI/scene.num_directions; 
    //     directions.push_back(Vector3a{std::cos(angle), std::sin(angle), 0});
    // }
    // scene.findBestCTensorviaProbing(sample_locs, directions);
    // scene.parameters = VectorXa(sim.springs.size()); 
    // scene.parameters.setConstant(5e-3);
    // scene.findBestCTensorviaProbing(sample_locs, directions, true);
    // sim.stretchDiagonal(1.1);
    // sim.Simulate();
    // Visualization vis(&scene);
    // vis.initializeScene(true);
    // vis.run();

    OptimizationProblem p(&scene, "../../../Projects/Optimization/optimization_output/" + mesh_name);//, "../../../Projects/EoLRods/optimization_output/" + mesh_name + "_radii_debug.dat");
    // std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStiffnessTensor>(sample_locs, Cs);
    std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateStiffnessTensorRelationship>(sample_locs);
    p.objective_energies.push_back(e);
    if(!p.Optimize()) std::cout << "\n Gradient very large \n";
    return 0;
}