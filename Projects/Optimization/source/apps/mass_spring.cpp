#include <iostream>
#include "../../include/Scene.h"
#include "../../include/MassSpring.h"
#include "../../include/Visualization.h"

int main(){

    MassSpring sim;
    Scene scene(sim);
    std::string mesh_name = "grid_double_refined";
    std::string filename = "../../../Projects/Optimization/data/"+mesh_name+".obj";
    scene.buildSceneFromMesh(filename);
    std::vector<Vector3a> sample_locs;
    sample_locs.push_back({0.5, 0.5, 0});
    std::vector<Vector3a> directions;
    for(int i = 0; i < scene.num_directions; ++i) {
        AScalar angle = i*2*M_PI/scene.num_directions; 
        directions.push_back(Vector3a{std::cos(angle), std::sin(angle), 0});
    }
    scene.findBestCTensorviaProbing(sample_locs, directions);
    sim.stretchY(1.1);
    sim.Simulate();
    Visualization vis(&scene);
    vis.initializeScene(true);
    vis.run();
    return 0;
}