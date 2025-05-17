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
    int num_test = 3;

    struct C_info{

        std::vector<Vector6a> C_diff_p;
        std::vector<Vector6a> C_diff_x;
        std::vector<Eigen::Matrix<AScalar, 6, 3>> C_diff_x_sim;
        VectorXa C_entry;

        C_info(int dof_x, int dof_p){
            Vector6a empty; empty.setZero();
            Eigen::Matrix<AScalar, 6, 3> empty_m; empty_m.setZero();
            C_diff_p = std::vector<Vector6a>(dof_p, empty);
            C_diff_x = std::vector<Vector6a>(dof_x, empty);
            C_diff_x_sim = std::vector<Eigen::Matrix<AScalar, 6, 3>>(dof_x, empty_m);
        }
    
    };
    std::vector<C_info> sample_Cs_info; 
    std::vector<std::vector<int>> constraint_sims;
    std::vector<Eigen::SparseMatrix<AScalar>> hessian_sims;
    std::vector<Eigen::SparseMatrix<AScalar>> hessian_p_sims;
    int num_directions = 4;

    int parameter_dof();
    int x_dof();
    VectorXa get_initial_params(){return sim.get_initial_parameter();}
    VectorXa get_curent_sim_params(){return sim.get_current_parameter();}
    VectorXa get_undeformed_nodes(){return sim.get_undeformed_nodes();};
    VectorXa get_deformed_nodes(){return sim.get_deformed_nodes();};
    std::vector<int> get_constraints(){return sim.get_constraint_map();}
    std::vector<int> get_constraint_sim(int i){return constraint_sims[i];}
    std::vector<std::array<size_t, 2>> get_edges(){return sim.get_edges();};
    void findBestCTensorviaProbing(std::vector<Vector3a> sample_locs, 
            const std::vector<Vector3a> line_directions, bool opt = false);
    void CTensorPerturbx(std::vector<Vector3a> sample_locs, 
        const std::vector<Vector3a> line_directions, MatrixXa deformed_x_offset);       
    void buildSimulationHessian(Eigen::SparseMatrix<AScalar>& K){sim.build_d2Edx2(K);}
    void buildSimulationdEdxp(Eigen::SparseMatrix<AScalar>& K){sim.build_d2Edxp(K);}
    void simulateWithParameter(const VectorXa parameters, int stretch_type){sim.setOptimizationParameter(parameters); sim.applyBoundaryStretch(stretch_type, 1.1); sim.Simulate();}
    Matrix3a returnApproxStressInCurrentSimulation(const Vector3a sample_loc, const std::vector<Vector3a> line_directions);
    Matrix3a returnApproxStrainInCurrentSimulation(const Vector3a sample_loc, const std::vector<Vector3a> line_directions);

};

#endif