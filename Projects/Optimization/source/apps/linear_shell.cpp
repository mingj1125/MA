#include <iostream>
#include "../../include/Scene.h"
#include "../../include/LinearShell.h"
#include "../../include/Visualization.h"
#include "../../include/OptimizationProblem.h"

int main(){

    LinearShell sim;
    Scene scene(sim);
    std::string mesh_name = "irregular_mesh_good";
    // std::string mesh_name = "grid_double_refined";
    // std::string mesh_name = "fused_alternating_rectangles_mesh";
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
    std::vector<Vector3a> Es = vis.getEs();
    std::vector<Vector3a> window_Es = vis.getWindowEs();
    // std::cout << "Cs: " << Cs.size() << std::endl;
    // std::cout << "Window Cs: " << window_Cs.size() << std::endl;

    // OptimizationProblem p(&scene, "../../../Projects/Optimization/optimization_output/shell_" + mesh_name);
    // std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStiffnessTensor>(sample_locs, Cs);

    // OptimizationProblem p(&scene, "../../../Projects/Optimization/optimization_output/shell_window_" + mesh_name);
    // std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStiffnessTensorWindow>(window_corners, window_Cs);

    // OptimizationProblem p(&scene, "../../../Projects/Optimization/optimization_output/shell_strain_" + mesh_name);
    // std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStrainTensor>(sample_locs, Es);

    OptimizationProblem p(&scene, "../../../Projects/Optimization/optimization_output/shell_window_strain_" + mesh_name);
    std::shared_ptr<ObjectiveEnergy> e = std::make_shared<ApproximateTargetStrainTensorWindow>(window_corners, window_Es);

    p.objective_energies.push_back(e); 
    // p.TestOptimizationGradient();
    // p.TestOptimizationSensitivity();
    // if(!p.Optimize()) std::cout << "\n Gradient not converged to the set criterion \n";
    // if(!p.OptimizeGD()) std::cout << "\n Gradient not converged to the set criterion \n";
    // if(!p.OptimizeGDFD()) std::cout << "\n Gradient not converged to the set criterion \n";
    if(!p.OptimizeFD()) std::cout  << "\n Gradient not converged to the set criterion \n";
    // p.ShowEnergyLandscape();

    return 0;
}