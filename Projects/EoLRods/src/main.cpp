#include "../include/EoLRodSim.h"
#include "../include/Scene.h"
#include "../include/App.h"
int main()
{
    EoLRodSim sim;
    Scene scene(sim);
    // scene.buildInterlockingSquareScene(8);
    // scene.buildFullScaleSquareScene(8);
    // scene.buildOneCrossScene(32);
    // scene.buildGridScene(0);
    // scene.buildStraightRodScene(1);
    // scene.buildFEMRodScene("../../../Projects/DiscreteShell/data/grid.obj", 0);
    // scene.buildFEMRodScene("../../../Projects/EoLRods/data/random_triangle_mesh.obj", 0);
    // scene.buildFEMRodScene("../../../Projects/EoLRods/data/irregular_mesh.obj", 0);
    scene.buildFEMRodScene("../../../Projects/EoLRods/data/irregular_mesh_good.obj", 0);
    
    App app(sim);
    // app.initializeScene("../../../Projects/EoLRods/data/irregular_mesh.obj");
    app.initializeScene("../../../Projects/EoLRods/data/irregular_mesh_good.obj");
    // app.initializeScene("../../../Projects/EoLRods/data/random_triangle_mesh.obj");
    // app.initializeScene("../../../Projects/DiscreteShell/data/grid.obj");
    // app.initializeScene();
    app.run();

    int num_directions = 4;
    std::vector<Eigen::Vector3d> directions;
    for(int i = 0; i < num_directions; ++i) {
        double angle = i*2*M_PI/num_directions; 
        directions.push_back(Eigen::Vector3d{std::cos(angle), std::sin(angle), 0});
    }
    std::cout << "Found C: \n" << scene.findBestCTensorviaProbing({0.0,0.0,0},directions) << std::endl;
    return 0;
}