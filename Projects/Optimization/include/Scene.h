#ifndef SCENE_H
#define SCENE_H

#include "vector_manipulation.h"
#include "Simulation.h"
#include <iostream>

class Scene
{

private:
    Simulation& sim;

    std::string output_dir;
    
public:
    Scene(Simulation& mesh_sim) : sim(mesh_sim) {}
    ~Scene() {}
    
    // ------------------------------- Scene Setup -------------------------------
    void buildSceneFromMesh(const std::string& mesh_name_s);

    // ------------------------------- Scene Property -------------------------------
    std::string mesh_file = "";
    std::string mesh_name;
    VectorXa parameters; 
    struct C_info{

        std::vector<Vector6a> C_diff_p;
        std::vector<Vector6a> C_diff_x;
        VectorXa C_entry;

        C_info(int dof_x, int dof_p){
            Vector6a empty; empty.setZero();
            C_diff_p = std::vector<Vector6a>(dof_p, empty);
            C_diff_x = std::vector<Vector6a>(dof_x, empty);
        }
    
    };
    std::vector<C_info> sample_Cs_info; 
    int num_directions = 15;

    int parameter_dof();
    int x_dof();
    VectorXa get_undeformed_nodes(){return sim.get_undeformed_nodes();};
    VectorXa get_deformed_nodes(){return sim.get_deformed_nodes();};
    std::vector<int> get_constraints(){return sim.get_constraint_map();}
    std::vector<std::array<size_t, 2>> get_edges(){return sim.get_edges();};
    void findBestCTensorviaProbing(std::vector<Vector3a> sample_locs, 
            const std::vector<Vector3a> line_directions, bool opt = false);
    void buildSimulationHessian(Eigen::SparseMatrix<AScalar>& K){sim.build_d2Edx2(K);}
    void buildSimulationdEdxp(Eigen::SparseMatrix<AScalar>& K){sim.build_d2Edxp(K);}
    void simulateWithParameter(const VectorXa parameters, int stretch_type){sim.setOptimizationParameter(parameters); sim.applyBoundaryStretch(stretch_type); sim.Simulate();}
    Matrix3a returnApproxStressInCurrentSimulation(const Vector3a sample_loc, const std::vector<Vector3a> line_directions);
    Matrix3a returnApproxStrainInCurrentSimulation(const Vector3a sample_loc, const std::vector<Vector3a> line_directions);

};

#endif