#include "../include/EoLRodSim.h"
#include "../include/Scene.h"
#include "../include/App.h"
#include "../include/OptimizationProblem.h"
#include "../include/ObjectiveEnergy.h"
#include <memory>

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
    
    // App app(sim);
    // app.initializeScene(mesh_name);
    // // app.initializeScene();
    // app.run();

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
    // // scene.rods_radii.setConstant(1e-2);
    // for(int i = 0; i < rods_radius.size(); ++i){
    //     scene.rods_radii(i) = rods_radius[i];
    // }
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

    // scene.findBestCTensorviaProbing(locations, directions); 

    // scene.optimizeForThickness({0.4, 0.53, 0}, {3.1e5, 8e4, 1.5e4, 4e5, 4e4, 2e5}, "../../../Projects/EoLRods/optimization_output/" + mesh_name);

    std::vector<Vector<T, 6>> Cs;
    
    Cs.push_back({919546,  380372, 310.317,  919546, 310.317,  757525});
    Cs.push_back({936408,  390733, 5111.18,  910424, 2643.15,  781467});
    Cs.push_back({919549,   380374, -285.445,   919549, -285.443,   757544});
    // Cs.push_back({919546,  380372, 310.317,  919546, 310.317,  757525});
    // Cs.push_back({936408,  390733, 5111.18,  910424, 2643.15,  781467});
    // Cs.push_back({919549,   380374, -285.445,   919549, -285.443,   757544});
    // Cs.push_back({919546,  380372, 310.317,  919546, 310.317,  757525});
    // Cs.push_back({936408,  390733, 5111.18,  910424, 2643.15,  781467});
    // Cs.push_back({919549,   380374, -285.445,   919549, -285.443,   757544});

    // 0 poisson ratio
    // Cs.push_back({550000,  0, 0,  550000,  0,  200000});
    // Cs.push_back({550000,  0, 0,  550000,  0,  200000});
    // Cs.push_back({550000,  0, 0,  550000,  0,  200000});
    // Cs.push_back({550000,  0, 0,  550000,  0,  200000});
    // Cs.push_back({550000,  0, 0,  550000,  0,  200000});
    // Cs.push_back({550000,  0, 0,  550000,  0,  200000});
    // Cs.push_back({550000,  0, 0,  550000,  0,  200000});
    // Cs.push_back({550000,  0, 0,  550000,  0,  200000});
    // Cs.push_back({550000,  0, 0,  550000,  0,  200000});

    // scene.optimizeForThicknessDistribution(locations, Cs, "../../../Projects/EoLRods/optimization_output/" + mesh_name, "../../../Projects/EoLRods/optimization_output/" + mesh_name + "_radii_graded_h_penalize.dat");

    // scene.finiteDifferenceEstimation({-0.5,  0.56, 0}, {634576, 181359, 4336.13, 726504, 40214.4, 380732});

    // OptimizationProblem p(&scene, "../../../Projects/EoLRods/optimization_output/" + mesh_name);//, "../../../Projects/EoLRods/optimization_output/" + mesh_name + "_radii_debug.dat");
    // // std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStiffnessTensor>(locations, Cs);
    // std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateStiffnessTensorRelationship>(locations);
    // p.objective_energies.push_back(e);
    // if(!p.Optimize()) std::cout << "\n Gradient very large \n";

    return 0;
}