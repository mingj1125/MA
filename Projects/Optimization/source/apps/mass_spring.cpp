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
    sim.stretchX(1.1);
    sim.Simulate();
    Visualization vis(&scene);
    vis.initializeScene(true);
    vis.run();
    return 0;
}