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
    virtual Eigen::SparseMatrix<AScalar> Compute_dcdx(Scene* scene){}
    virtual Eigen::SparseMatrix<AScalar> Compute_dcdp(Scene* scene){}
    virtual VectorXa Compute_dfdx(Scene* scene){}
    virtual VectorXa Compute_dfdp(Scene* scene){}
    virtual Eigen::SparseMatrix<AScalar> Compute_d2fdx2(Scene* scene){}
    virtual Eigen::SparseMatrix<AScalar> Compute_d2fdxp(Scene* scene){}
    virtual Eigen::SparseMatrix<AScalar> Compute_d2fdp2(Scene* scene){}

    virtual void SimulateAndCollect(Scene* scene){}
};


class ApproximateTargetStiffnessTensor: public ObjectiveEnergy
{    
public:

    Eigen::Vector<int, 6> consider_entry; // 1 if considered
    std::vector<Vector3a> target_locations;
    std::vector<Vector6a> target_stiffness_tensors;
    std::vector<Eigen::Vector3d> directions;


    ApproximateTargetStiffnessTensor(std::vector<Vector3a> target_locations_m, std::vector<Vector6a> target_stiffness_tensors_m){
        target_locations = target_locations_m;
        target_stiffness_tensors = target_stiffness_tensors_m;
        // consider_entry.setConstant(1);
        setConsideringEntry({1, 1, 0, 1, 0, 0});
        int num_directions = 20;
        for(int i = 0; i < num_directions; ++i) {
            double angle = i*2*M_PI/num_directions; 
            directions.push_back(Eigen::Vector3d{std::cos(angle), std::sin(angle), 0});
        }
    }

    void setConsideringEntry(Eigen::Vector<int, 6> ce){
        consider_entry = ce;
    }

    virtual AScalar ComputeEnergy(Scene* scene);
    virtual Eigen::SparseMatrix<AScalar> Compute_dcdx(Scene* scene);
    virtual Eigen::SparseMatrix<AScalar> Compute_dcdp(Scene* scene);
    virtual VectorXa Compute_dfdx(Scene* scene);
    virtual VectorXa Compute_dfdp(Scene* scene);
    virtual Eigen::SparseMatrix<AScalar> Compute_d2fdx2(Scene* scene);
    virtual Eigen::SparseMatrix<AScalar> Compute_d2fdxp(Scene* scene);
    virtual Eigen::SparseMatrix<AScalar> Compute_d2fdp2(Scene* scene);

    virtual void SimulateAndCollect(Scene* scene);

};

class ApproximateStiffnessTensorRelationship: public ObjectiveEnergy
{    
public:

    // Vector<int, 6> consider_entry; // 1 if considered
    std::vector<Vector3a> target_locations;
    // std::vector<Eigen::Vector3d> directions;
    AScalar ratio = 2;


    ApproximateStiffnessTensorRelationship(std::vector<Vector3a> target_locations_m){
        target_locations = target_locations_m;
        // int num_directions = 20;
        // for(int i = 0; i < num_directions; ++i) {
        //     double angle = i*2*M_PI/num_directions; 
        //     directions.push_back(Eigen::Vector3d{std::cos(angle), std::sin(angle), 0});
        // }
    }

    virtual AScalar ComputeEnergy(Scene* scene);
    virtual Eigen::SparseMatrix<AScalar> Compute_dcdx(Scene* scene);
    virtual Eigen::SparseMatrix<AScalar> Compute_dcdp(Scene* scene);
    virtual VectorXa Compute_dfdx(Scene* scene);
    virtual VectorXa Compute_dfdp(Scene* scene);
    virtual Eigen::SparseMatrix<AScalar> Compute_d2fdx2(Scene* scene);
    virtual Eigen::SparseMatrix<AScalar> Compute_d2fdxp(Scene* scene);
    virtual Eigen::SparseMatrix<AScalar> Compute_d2fdp2(Scene* scene);

    virtual void SimulateAndCollect(Scene* scene);

};

#endif