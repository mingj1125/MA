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
    // app.initializeScene("irregular_mesh");
    app.initializeScene("irregular_mesh_good");
    // app.initializeScene("random_triangle_mesh");
    // app.initializeScene("grid");
    // app.initializeScene();
    app.run();

    // int num_directions = 8;
    // std::vector<Eigen::Vector3d> directions;
    // for(int i = 0; i < num_directions; ++i) {
    //     double angle = i*2*M_PI/num_directions; 
    //     directions.push_back(Eigen::Vector3d{std::cos(angle), std::sin(angle), 0});
    // }
    // std::cout << "Found C: \n" << scene.findBestCTensorviaProbing({0.4, 0.53, 0},directions) << std::endl;

    // scene.optimizeForThickness({0.4, 0.53, 0}, {2.3e7, 5e6, 7e5, 2e7, 7e5, 1e7}, "../../../Projects/EoLRods/optimization_output/irregular_mesh_good");

    // std::vector<Vector<T, 3>> locations;
    // locations.push_back({0.4, 0.53, 0});
    // locations.push_back({0.64, 0.36, 0});
    // std::vector<Vector<T, 6>> Cs;
    // Cs.push_back({2.3e7, 5e6, 7e5, 2e7, 7e5, 1e7});
    // Cs.push_back({2.3e7, 5e6, 7e5, 2e7, 7e5, 1e7});
    // scene.optimizeForThicknessDistribution(locations, Cs, "../../../Projects/EoLRods/optimization_output/irregular_mesh_good");
    return 0;
}