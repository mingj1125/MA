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
    std::string mesh_name = "grid_double_refined";
    scene.buildFEMRodScene("../../../Projects/EoLRods/data/"+mesh_name+".obj", 0);
    
    App app(sim);
    app.initializeScene(mesh_name);
    // app.initializeScene();
    app.run();

    // int num_directions = 20;
    // std::vector<Eigen::Vector3d> directions;
    // for(int i = 0; i < num_directions; ++i) {
    //     double angle = i*2*M_PI/num_directions; 
    //     directions.push_back(Eigen::Vector3d{std::cos(angle), std::sin(angle), 0});
    // }
    // std::vector<double> rods_radius;
    // std::string filename = "../../../Projects/EoLRods/optimization_output/"+mesh_name+"_radii.dat";
    // std::ifstream in_file(filename);
    // if (!in_file) {
    //     std::cerr << "Error opening file for reading: " << filename << std::endl;
    // }

    // T a;
    // while (in_file >> a) {
    //     rods_radius.push_back(a);
    // }
    // scene.rods_radii.resize(rods_radius.size());
    // scene.rods_radii.setConstant(6.5e-3);
    // // for(int i = 0; i < rods_radius.size(); ++i){
    // //     scene.rods_radii(i) = rods_radius[i];
    // // }
    std::vector<Vector<T, 3>> locations;
    
    // irregular
    // locations.push_back({0.4, 0.53, 0});
    // locations.push_back({0.64, 0.36, 0});

    // nine points
    locations.push_back({-0.76, 0.76, 0});
    locations.push_back({-0.50, 0.76, 0});
    locations.push_back({-0.24, 0.76, 0});
    locations.push_back({-0.76, 0.50, 0});
    locations.push_back({-0.50, 0.50, 0});
    locations.push_back({-0.24, 0.50, 0});
    locations.push_back({-0.76, 0.24, 0});
    locations.push_back({-0.50, 0.24, 0});
    locations.push_back({-0.24, 0.24, 0});

    // 2 points double refined 
    // locations.push_back({-0.6, 0.56, 0});
    // locations.push_back({-0.32, 0.4, 0});

    // scene.findBestCTensorviaProbing(locations, directions, true); 

    // scene.optimizeForThickness({0.4, 0.53, 0}, {3.1e5, 8e4, 1.5e4, 4e5, 4e4, 2e5}, "../../../Projects/EoLRods/optimization_output/" + mesh_name);

    std::vector<Vector<T, 6>> Cs;

    // // // Cs.push_back({3e5, 8e4, -5e4, 3e5, -1e4, 1.5e5});
    // // // Cs.push_back({3.1e5, 8e4, 1.5e4, 4e5, 4e4, 2e5});

    // nine points
    // Cs.push_back({548000,  225800, -150,  544000,  -150,  453000});
    // Cs.push_back({548000,  225800, -150,  544000,  -150,  453000});
    // Cs.push_back({548000,  225800, -150,  544000,  -150,  453000});
    // Cs.push_back({590000,  244500, -150,  594000,  -150,  490000});
    // Cs.push_back({590000,  244500, -150,  594000,  -150,  490000});
    // Cs.push_back({590000,  244500, -150,  594000,  -150,  490000});
    // Cs.push_back({635000,  263000, -150,  640000,  -150,  529000});
    // Cs.push_back({635000,  263000, -150,  640000,  -150,  529000});
    // Cs.push_back({635000,  263000, -150,  640000,  -150,  529000});

    Cs.push_back({590000,  244500, -150,  594000,  -150,  400000});
    Cs.push_back({590000,  244500, -150,  594000,  -150,  400000});
    Cs.push_back({590000,  244500, -150,  594000,  -150,  400000});
    Cs.push_back({590000,  244500, -150,  594000,  -150,  400000});
    Cs.push_back({590000,  244500, -150,  594000,  -150,  400000});
    Cs.push_back({590000,  244500, -150,  594000,  -150,  400000});
    Cs.push_back({590000,  244500, -150,  594000,  -150,  400000});
    Cs.push_back({590000,  244500, -150,  594000,  -150,  400000});
    Cs.push_back({590000,  244500, -150,  594000,  -150,  400000});

    // scene.optimizeForThicknessDistribution(locations, Cs, "../../../Projects/EoLRods/optimization_output/" + mesh_name);

    // scene.finiteDifferenceEstimation({0.4,  0.53, 0}, {634576, 181359, 4336.13, 726504, 40214.4, 380732});
    return 0;
}