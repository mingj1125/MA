#ifndef SCENE_H
#define SCENE_H

#include "EoLRodSim.h"

// class EoLRodSim;

class Scene
{
public:
    using TV = Vector<T, 3>;
    using TV2 = Vector<T, 2>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;

    using Offset = Vector<int, 3 + 1>;
    using Range = Vector<T, 2>;
    using Mask = Vector<bool, 3>;
    using Mask2 = Vector<bool, 2>;

private:
    EoLRodSim& sim;
    std::string mesh_file = "";
    VectorXT& deformed_states = sim.deformed_states;
    double ROD_A = 1e-2;
    double ROD_B = 1e-2;

    std::vector<VectorXT> C_diff_thickness;
    std::vector<VectorXT> C_diff_x;
    VectorXT C_entry;
    
public:
    Scene(EoLRodSim& eol_sim) : sim(eol_sim) {}
    ~Scene() {}
    
    // ------------------------------- Scene Setup -------------------------------
    void buildInterlockingSquareScene(int sub_div);
    void buildStraightRodScene(int sub_div);
    void buildGridScene(int sub_div, bool bc_data = false);
    void buildFullScaleSquareScene(int sub_div);
    void buildFullCircleScene(int sub_div);
    void buildOneCrossScene(int sub_div);
    void buildOneCrossSceneCurved(int sub_div);
    void buildFEMRodScene(const std::string& filename, int sub_div, bool bc_data = false);

    // ------------------------------- Scene Property -------------------------------
    // EoLRodSim& sim;
    VectorXT rods_radii; // assume circular cross-section
    struct C_info{

        std::vector<VectorXT> C_diff_thickness;
        std::vector<VectorXT> C_diff_x;
        VectorXT C_entry;
    
    };
    std::vector<C_info> sample_Cs_info; 
    StiffnessMatrix equilibrium_K;

    Matrix<T, 3, 3> findBestCTensorviaProbing(TV sample_loc, const std::vector<TV> line_directions, bool opt = false);
    void findBestCTensorviaProbing(std::vector<TV> sample_locs, const std::vector<TV> line_directions, bool opt = false);
    void optimizeForThickness(TV target_location, Vector<T, 6> stiffness_tensor, std::string filename);
    void optimizeForThicknessDistribution(const std::vector<TV> target_locations, const std::vector<Vector<T, 6>> stiffness_tensors, const std::string filename, const std::string start_from_file = "");
    void finiteDifferenceEstimation(TV target_location, Vector<T, 6> stiffness_tensor);
    int num_rods(){return sim.Rods.size();}
    int num_nodes(){return sim.deformed_states.rows();}
    void buildSimulationHessian(StiffnessMatrix& K){sim.buildSimulationHessian(K);}
    void buildSimulationdEdxp(StiffnessMatrix& K){sim.buildForceGradientWrtThicknessMatrix(K);}
    StiffnessMatrix simulationW(){return sim.W;};
    VectorXT solveForAdjoint(StiffnessMatrix& K, VectorXT rhs);

private:

    // ------------------------------- Common Function -------------------------------
    void clearSimData();

    void appendThetaAndJointDoF(std::vector<Entry>& w_entry, 
        int& full_dof_cnt, int& dof_cnt);   
    
    void addStraightYarnCrossNPoints(const TV& from, const TV& to,
        const std::vector<TV>& passing_points, 
        const std::vector<int>& passing_points_id, 
        int sub_div,
        std::vector<TV>& sub_points, std::vector<int>& node_idx,
        std::vector<int>& key_points_location,
        int start, bool pbc = false);

    void markCrossingDoF(
        std::vector<Eigen::Triplet<T>>& w_entry,
        int& dof_cnt);

    void addAStraightRod(const TV& from, const TV& to, 
        const std::vector<TV>& passing_points, 
        const std::vector<int>& passing_points_id, 
        int sub_div,
        int& full_dof_cnt, int& node_cnt, int& rod_cnt);
    
    void addCurvedRod(const std::vector<TV2>& data_points,
        const std::vector<TV>& passing_points, 
        const std::vector<int>& passing_points_id, 
        int sub_div, int& full_dof_cnt, int& node_cnt, int& rod_cnt, bool closed);
    
    void addPoint(const TV& point, int& full_dof_cnt, int& node_cnt);

    void addCrossingPoint(std::vector<TV>& existing_nodes, 
        const TV& point, int& full_dof_cnt, int& node_cnt);
};

#endif