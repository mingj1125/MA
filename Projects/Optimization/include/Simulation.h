#ifndef SIMULATION_H
#define SIMULATION_H

#include "vector_manipulation.h"
#include "damped_newton.h"

class Simulation
{

public:

    virtual void initializeScene(const std::string& filename) = 0;
    virtual VectorXa get_initial_parameter() = 0;
    virtual VectorXa get_undeformed_nodes() = 0;
    virtual VectorXa get_deformed_nodes() = 0;
    virtual VectorXa get_current_parameter(){}
    virtual std::vector<int> get_constraint_map() = 0;
    virtual std::vector<std::array<size_t, 2>> get_edges(){}
    virtual void set_kernel_std(AScalar std){}

    virtual Matrix3a findBestStressTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){}
    virtual Matrix3a findBestStrainTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){}
    virtual void setOptimizationParameter(VectorXa parameters){}
    virtual void setDeformedState(VectorXa parameters){}
    virtual MatrixXa getStressGradientWrtParameter(){}
    virtual MatrixXa getStressGradientWrtx(){}
    virtual MatrixXa getStrainGradientWrtx(){}
    virtual void build_sim_hessian(Eigen::SparseMatrix<AScalar>& K){}
    virtual void build_d2Edxp(Eigen::SparseMatrix<AScalar>& K){}

    virtual void applyBoundaryStretch(int i, AScalar strain = 0){}
    virtual damped_newton_result Simulate(bool use_log = true){}

    virtual Matrix3a findStrainTensorinWindow(const Vector2a max_corner, const Vector2a min_corner){}
    virtual Matrix3a findStressTensorinWindow(const Vector2a max_corner, const Vector2a min_corner){}

};


#endif