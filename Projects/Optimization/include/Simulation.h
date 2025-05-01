#ifndef SIMULATION_H
#define SIMULATION_H

#include "vector_manipulation.h"
#include "damped_newton.h"

class Simulation
{

public:

    virtual void initializeScene(const std::string& filename) = 0;
    virtual VectorXa get_undeformed_nodes() = 0;
    virtual VectorXa get_deformed_nodes() = 0;
    virtual std::vector<std::array<size_t, 2>> get_edges(){}
    virtual Matrix3a findBestStressTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){}
    virtual Matrix3a findBestStrainTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){}
    virtual void setOptimizationParameter(VectorXa parameters){}
    virtual MatrixXa getStressGradientWrtParameter(){}
    virtual MatrixXa getStressGradientWrtx(){}
    virtual MatrixXa getStrainGradientWrtx(){}
    virtual void build_d2Edx2(Eigen::SparseMatrix<AScalar>& K){}
    virtual void build_d2Edxp(Eigen::SparseMatrix<AScalar>& K){}

    virtual void ApplyBoundaryStretch(int i){}
    virtual damped_newton_result Simulate(){}

};


#endif