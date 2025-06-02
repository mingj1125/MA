#ifndef OBJECTIVE_ENERGY_PROBLEM_H
#define OBJECTIVE_ENERGY_PROBLEM_H

#include "vector_manipulation.h"
#include "Scene.h"

class ObjectiveEnergy
{
private:
    /* data */
public:

    virtual AScalar ComputeEnergy(Scene* scene){}
    virtual VectorXa Compute_dfdp(Scene* scene){}
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdx2(Scene* scene){}
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdxp(Scene* scene){}
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdp2(Scene* scene){}

    virtual VectorXa Compute_dfdx_sim(Scene* scene){}

    virtual void SimulateAndCollect(Scene* scene){}
    virtual void OnlyCollect(Scene* scene, MatrixXa offsets){}
};


class ApproximateTargetStiffnessTensor: public ObjectiveEnergy
{    
public:

    Vector6a consider_entry; // 1 if considered
    std::vector<Vector3a> target_locations;
    std::vector<Vector6a> target_stiffness_tensors;

    ApproximateTargetStiffnessTensor(std::vector<Vector3a> target_locations_m, std::vector<Vector6a> target_stiffness_tensors_m){
        target_locations = target_locations_m;
        target_stiffness_tensors = target_stiffness_tensors_m;
        // consider_entry.setConstant(1);
        setConsideringEntry({1, 1, 0, 1, 0, 0.001});
    }

    void setConsideringEntry(Vector6a ce){
        consider_entry = ce;
    }

    virtual AScalar ComputeEnergy(Scene* scene);
    virtual VectorXa Compute_dfdp(Scene* scene);
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdx2(Scene* scene);
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdxp(Scene* scene);
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdp2(Scene* scene);

    virtual VectorXa Compute_dfdx_sim(Scene* scene);

    virtual void SimulateAndCollect(Scene* scene);
    virtual void OnlyCollect(Scene* scene, MatrixXa offsets);

};

class ApproximateStiffnessTensorRelationship: public ObjectiveEnergy
{    
public:

    // Vector<int, 6> consider_entry; // 1 if considered
    std::vector<Vector3a> target_locations;
    // std::vector<Eigen::Vector3d> directions;
    AScalar ratio = 3;


    ApproximateStiffnessTensorRelationship(std::vector<Vector3a> target_locations_m){
        target_locations = target_locations_m;
    }

    virtual AScalar ComputeEnergy(Scene* scene);
    virtual VectorXa Compute_dfdp(Scene* scene);
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdx2(Scene* scene);
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdxp(Scene* scene);
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdp2(Scene* scene);

    virtual VectorXa Compute_dfdx_sim(Scene* scene);

    virtual void SimulateAndCollect(Scene* scene);
    virtual void OnlyCollect(Scene* scene, MatrixXa offsets);

};

class ApproximateTargetStiffnessTensorWindow: public ObjectiveEnergy
{    
public:

    Vector6a consider_entry; // 1 if considered
    std::vector<Vector4a> target_corners;
    std::vector<Vector6a> target_stiffness_tensors;

    ApproximateTargetStiffnessTensorWindow(std::vector<Vector4a> target_corners_m, std::vector<Vector6a> target_stiffness_tensors_m){
        target_corners = target_corners_m;
        target_stiffness_tensors = target_stiffness_tensors_m;
        // consider_entry.setConstant(1);
        setConsideringEntry({1, 1, 0, 1, 0, 0.001});
    }

    void setConsideringEntry(Vector6a ce){
        consider_entry = ce;
    }

    virtual AScalar ComputeEnergy(Scene* scene);
    virtual VectorXa Compute_dfdp(Scene* scene);
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdx2(Scene* scene);
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdxp(Scene* scene);
    virtual std::vector<Eigen::SparseMatrix<AScalar>> Compute_d2fdp2(Scene* scene);

    virtual VectorXa Compute_dfdx_sim(Scene* scene);

    virtual void SimulateAndCollect(Scene* scene);

};

#endif