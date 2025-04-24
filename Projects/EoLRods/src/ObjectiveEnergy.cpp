#include "../include/ObjectiveEnergy.h"
#include "../include/EoLRodSim.h"

AScalar ApproximateTargetStiffnessTensor::ComputeEnergy(Scene* scene){

    AScalar energy = 0;
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        Vector6a target_C = target_stiffness_tensors[k];
        for(int j = 0; j < 6; ++j){

            if(consider_entry(j) == 1) energy += 0.5*(sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))*(sample_Cs_info[k].C_entry(j) - target_C(j));

        }
    }    
    return energy;
}

Eigen::SparseMatrix<AScalar> ApproximateTargetStiffnessTensor::Compute_dcdx(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> K;
    scene->buildSimulationHessian(K);

    return K;
}

Eigen::SparseMatrix<AScalar> ApproximateTargetStiffnessTensor::Compute_dcdp(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> fd_thickness;
    scene->buildSimulationdEdxp(fd_thickness);

    return fd_thickness;
}

VectorXa ApproximateTargetStiffnessTensor::Compute_dfdx(Scene* scene){
    
    VectorXa gradient_wrt_x(scene->num_nodes()); gradient_wrt_x.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        Vector6a target_C = target_stiffness_tensors[k];
        for(int i = 0; i < gradient_wrt_x.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                if(consider_entry(j) == 1) g += (sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_x[i](j);

            }
            gradient_wrt_x(i) += g;
        }
    }
    
    VectorXa dfdx = scene->simulationW().transpose()*gradient_wrt_x;

    return dfdx;
}

VectorXa ApproximateTargetStiffnessTensor::Compute_dfdp(Scene* scene){
    
    VectorXa gradient_wrt_thickness(scene->num_rods()); gradient_wrt_thickness.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        Vector6a target_C = target_stiffness_tensors[k];
        for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                if(consider_entry(j) == 1) g += (sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_thickness[i](j);

            }
            gradient_wrt_thickness(i) += g;
        }
    }

    return gradient_wrt_thickness;
}

Eigen::SparseMatrix<AScalar> ApproximateTargetStiffnessTensor::Compute_d2fdx2(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> drdx(scene->num_nodes(), 1); drdx.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        Vector6a target_C = target_stiffness_tensors[k];
        for(int i = 0; i < drdx.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                if(consider_entry(j) == 1) g += sample_Cs_info[k].C_diff_x[i](j)/abs(target_C(j));

            }
            if(std::abs(g) > 1e-7) drdx.coeffRef(i, 0) += g;
        }
    }
    Eigen::SparseMatrix<AScalar> d2fdx2 = (drdx * drdx.transpose()).pruned();


    return d2fdx2;
}

Eigen::SparseMatrix<AScalar> ApproximateTargetStiffnessTensor::Compute_d2fdxp(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> drdx(scene->num_nodes(), 1); drdx.setZero();
    Eigen::SparseMatrix<AScalar> drdp(scene->num_rods(), 1); drdp.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        Vector6a target_C = target_stiffness_tensors[k];
        for(int i = 0; i < drdx.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                if(consider_entry(j) == 1) g += sample_Cs_info[k].C_diff_x[i](j)/abs(target_C(j));

            }
            
            if(std::abs(g) > 1e-7) drdx.coeffRef(i, 0) += g;
        }
        for(int i = 0; i < drdp.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                if(consider_entry(j) == 1) g += sample_Cs_info[k].C_diff_thickness[i](j)/abs(target_C(j));

            }
            if(std::abs(g) > 1e-7) drdp.coeffRef(i, 0) += g;
        }
    }
    Eigen::SparseMatrix<AScalar> d2fdxp = (drdp * drdx.transpose()).pruned();

    return d2fdxp;
}

Eigen::SparseMatrix<AScalar> ApproximateTargetStiffnessTensor::Compute_d2fdp2(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> drdp(scene->num_rods(), 1); drdp.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        Vector6a target_C = target_stiffness_tensors[k];
        for(int i = 0; i < drdp.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                if(consider_entry(j) == 1) g += sample_Cs_info[k].C_diff_thickness[i](j)/abs(target_C(j));

            }
            if(std::abs(g) > 1e-7) drdp.coeffRef(i, 0) += g;
        }
    }
    Eigen::SparseMatrix<AScalar> d2fdp2 = (drdp * drdp.transpose()).pruned();

    return d2fdp2;
}

void ApproximateTargetStiffnessTensor::SimulateAndCollect(Scene* scene){

    scene->findBestCTensorviaProbing(target_locations, directions, true);
}



