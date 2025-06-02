#include <iostream>
#include "../../include/Scene.h"
#include "../../include/LinearShell.h"
#include "../../include/Visualization.h"
#include "../../include/OptimizationProblem.h"

int main(){

    LinearShell sim;
    Scene scene(sim);
    // std::string mesh_name = "irregular_mesh_good";
    // std::string mesh_name = "grid_double_refined";
    std::string mesh_name = "fused_alternating_rectangles_mesh";
    // std::string mesh_name = "sun_mesh_line_clean";
    scene.buildSceneFromMesh(mesh_name);

    Visualization vis(&scene);
    vis.initializeScene(false);
    vis.run();

    std::vector<Vector3a> sample_locs;
    std::vector<Vector4a> window_corners;

    // sample_locs.push_back({0.5, 0.5, 0.0});
    // AScalar length = 0.1;
    // Vector2a max_corner = sample_locs[0].segment(0,2) + length*Vector2a({1,1});
    // Vector2a min_corner = sample_locs[0].segment(0,2) - length*Vector2a({1,1});
    // window_corners.push_back({Vector4a({max_corner(0), max_corner(1), min_corner(0), min_corner(1)})});

    sample_locs = vis.getSampleLocations();
    std::vector<Vector6a> Cs = vis.getCs();
    window_corners = vis.getWindowCorners();
    std::vector<Vector6a> window_Cs = vis.getWindowCs();
    // std::cout << "Cs: " << Cs.size() << std::endl;
    // std::cout << "Window Cs: " << window_Cs.size() << std::endl;

    // sun mesh 
    // std::vector<Vector6a> Cs;
    // Cs.push_back({7.48708e+07, 7.18969e+07,      1587.9, 7.47151e+07,     17276.6, 1.46273e+06});

    // Cs.push_back({2.62875e+06,      654152,    -8568.08, 2.62763e+06,    -5255.23,      949660});
    // Cs.push_back({2.66592e+06,      683764,     414.024, 2.68369e+06,     1560.64,      982606});
    // Cs.push_back({2.61467e+06,      651208,     11004.5,  2.6155e+06,     11476.8,      949017});

    // Cs.push_back({2.6853e+06,      677171,    -605.442, 2.66351e+06,    -2324.94,      984281}); 
    // Cs.push_back({3.0532e+06,      760293,     -6183.9, 3.04781e+06,     7722.51, 1.11462e+06}); 
    // Cs.push_back({2.68503e+06,      677076,    -622.029, 2.66426e+06,    -2274.02,      984261});

    // Cs.push_back({2.61461e+06,      651205,     11066.8, 2.61552e+06,     11488.3,      949030}); 
    // Cs.push_back({2.66522e+06,      683722,     246.049, 2.68385e+06,      1667.5,      982353}); 
    // Cs.push_back({2.60616e+06,      659583,    -17425.7, 2.60269e+06,    -15401.3,      945104}); 

    // OptimizationProblem p(&scene, "../../../Projects/Optimization/optimization_output/shell_" + mesh_name);
    // std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStiffnessTensor>(sample_locs, Cs);

    OptimizationProblem p(&scene, "../../../Projects/Optimization/optimization_output/shell_window_" + mesh_name);
    std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStiffnessTensorWindow>(window_corners, window_Cs);
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