#include "../include/ObjectiveEnergy.h"

AScalar WEIGHTS = 1e-11;

AScalar ApproximateTargetStiffnessTensor::ComputeEnergy(Scene* scene){

    AScalar energy = 0;
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        Vector6a target_C = target_stiffness_tensors[k];
        for(int j = 0; j < 6; ++j){

            if(consider_entry(j) == 1) energy += 0.5*(sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))/abs(target_C(j))*(sample_Cs_info[k].C_entry(j) - target_C(j));

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
    
    Eigen::SparseMatrix<AScalar> K;
    scene->buildSimulationdEdxp(K);

    return K;
}

VectorXa ApproximateTargetStiffnessTensor::Compute_dfdx(Scene* scene){
    
    VectorXa gradient_wrt_x(scene->x_dof()); gradient_wrt_x.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        Vector6a target_C = target_stiffness_tensors[k];
        for(int i = 0; i < gradient_wrt_x.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                if(consider_entry(j) == 1) g += (sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_x[i](j);

            }
            gradient_wrt_x(i) += g;
        }
    }
    
    VectorXa dfdx = gradient_wrt_x;

    return dfdx;
}

VectorXa ApproximateTargetStiffnessTensor::Compute_dfdp(Scene* scene){
    
    VectorXa gradient_wrt_thickness(scene->parameter_dof()); gradient_wrt_thickness.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        Vector6a target_C = target_stiffness_tensors[k];
        for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                if(consider_entry(j) == 1) g += (sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_p[i](j);

            }
            gradient_wrt_thickness(i) += g;
        }
    }

    return gradient_wrt_thickness;
}

Eigen::SparseMatrix<AScalar> ApproximateTargetStiffnessTensor::Compute_d2fdx2(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> drdx(scene->x_dof(), 1); drdx.setZero();
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
    Eigen::SparseMatrix<AScalar> d = drdx;
    Eigen::SparseMatrix<AScalar> d2fdx2 = (d * d.transpose()).pruned();


    return d2fdx2;
}

Eigen::SparseMatrix<AScalar> ApproximateTargetStiffnessTensor::Compute_d2fdxp(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> drdx(scene->x_dof(), 1); drdx.setZero();
    Eigen::SparseMatrix<AScalar> drdp(scene->parameter_dof(), 1); drdp.setZero();
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

                if(consider_entry(j) == 1) g += sample_Cs_info[k].C_diff_p[i](j)/abs(target_C(j));

            }
            if(std::abs(g) > 1e-7) drdp.coeffRef(i, 0) += g;
        }
    }
    Eigen::SparseMatrix<AScalar> d2fdxp = (drdp * drdx.transpose()).pruned();

    return d2fdxp;
}

Eigen::SparseMatrix<AScalar> ApproximateTargetStiffnessTensor::Compute_d2fdp2(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> drdp(scene->parameter_dof(), 1); drdp.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        Vector6a target_C = target_stiffness_tensors[k];
        for(int i = 0; i < drdp.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                if(consider_entry(j) == 1) g += sample_Cs_info[k].C_diff_p[i](j)/abs(target_C(j));

            }
            if(std::abs(g) > 1e-7) drdp.coeffRef(i, 0) += g;
        }
    }
    Eigen::SparseMatrix<AScalar> d2fdp2 = (drdp * drdp.transpose()).pruned();

    return d2fdp2;
}

void ApproximateTargetStiffnessTensor::SimulateAndCollect(Scene* scene){

    std::vector<Vector3a> directions;
    for(int i = 0; i < scene->num_directions; ++i) {
        AScalar angle = i*2*M_PI/scene->num_directions; 
        directions.push_back(Vector3a{std::cos(angle), std::sin(angle), 0});
    }
    scene->findBestCTensorviaProbing(target_locations, directions, true);
}


AScalar ApproximateStiffnessTensorRelationship::ComputeEnergy(Scene* scene){

    AScalar energy = 0;
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
      
        AScalar r = sample_Cs_info[k].C_entry(0) - 2*sample_Cs_info[k].C_entry(1);
        energy += 0.5*r*r;
        r = sample_Cs_info[k].C_entry(3) - 2*sample_Cs_info[k].C_entry(1);
        energy += 0.5*r*r;
    }
    energy *= WEIGHTS / target_locations.size();    
    return energy;
}

Eigen::SparseMatrix<AScalar> ApproximateStiffnessTensorRelationship::Compute_dcdx(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> K;
    scene->buildSimulationHessian(K);

    return K;
}

Eigen::SparseMatrix<AScalar> ApproximateStiffnessTensorRelationship::Compute_dcdp(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> K;
    scene->buildSimulationdEdxp(K);

    return K;
}

VectorXa ApproximateStiffnessTensorRelationship::Compute_dfdx(Scene* scene){
    
    auto sample_Cs_info = scene->sample_Cs_info;
    VectorXa gradient_wrt_x(scene->x_dof()); gradient_wrt_x.setZero();

    std::vector<bool> constrained(scene->x_dof());
    std::vector<int> constraints = scene->get_constraints();
    for(int j = 0; j < constraints.size(); ++j){
        constrained[constraints[j]] = true;
    }

    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        for(int i = 0; i < gradient_wrt_x.size(); ++i){
            if(constrained[i]) continue;
            AScalar g = 0;

            g += (sample_Cs_info[k].C_entry(0) -  2*sample_Cs_info[k].C_entry(1))*sample_Cs_info[k].C_diff_x[i](0);
            g += (sample_Cs_info[k].C_entry(0) -  2*sample_Cs_info[k].C_entry(1))*(-2)*sample_Cs_info[k].C_diff_x[i](1);
            g += (sample_Cs_info[k].C_entry(3) -  2*sample_Cs_info[k].C_entry(1))*sample_Cs_info[k].C_diff_x[i](3);
            g += (sample_Cs_info[k].C_entry(3) -  2*sample_Cs_info[k].C_entry(1))*(-2)*sample_Cs_info[k].C_diff_x[i](1);

            gradient_wrt_x(i) += g;
        }
    }
    
    VectorXa dfdx = gradient_wrt_x*WEIGHTS/target_locations.size();

    

    return dfdx;
}

VectorXa ApproximateStiffnessTensorRelationship::Compute_dfdp(Scene* scene){
    
    VectorXa gradient_wrt_thickness(scene->parameter_dof()); gradient_wrt_thickness.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
            AScalar g = 0;

            g += (sample_Cs_info[k].C_entry(0) -  2*sample_Cs_info[k].C_entry(1))*sample_Cs_info[k].C_diff_p[i](0);
            g += (sample_Cs_info[k].C_entry(0) -  2*sample_Cs_info[k].C_entry(1))*(-2)*sample_Cs_info[k].C_diff_p[i](1);
            g += (sample_Cs_info[k].C_entry(3) -  2*sample_Cs_info[k].C_entry(1))*sample_Cs_info[k].C_diff_p[i](3);
            g += (sample_Cs_info[k].C_entry(3) -  2*sample_Cs_info[k].C_entry(1))*(-2)*sample_Cs_info[k].C_diff_p[i](1);
  
            gradient_wrt_thickness(i) += g*WEIGHTS/target_locations.size();
        }
    }

    return gradient_wrt_thickness;
}

Eigen::SparseMatrix<AScalar> ApproximateStiffnessTensorRelationship::Compute_d2fdx2(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> drdx(scene->x_dof(), 1); drdx.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;

    std::vector<bool> constrained(scene->x_dof());
    std::vector<int> constraints = scene->get_constraints();
    for(int j = 0; j < constraints.size(); ++j){
        constrained[constraints[j]] = true;
    }

    for(int k = 0; k < target_locations.size(); ++k){
        for(int i = 0; i < drdx.size(); ++i){
            if(constrained[i]) continue;
            AScalar g = 0;

            g += sample_Cs_info[k].C_diff_x[i](0);
            g += (-2.0)*sample_Cs_info[k].C_diff_x[i](1);
            g += sample_Cs_info[k].C_diff_x[i](3);
            g += (-2.0)*sample_Cs_info[k].C_diff_x[i](1);

            drdx.coeffRef(i, 0) += g;
        }
    }
    Eigen::SparseMatrix<AScalar> d = drdx;
    Eigen::SparseMatrix<AScalar> d2fdx2 = WEIGHTS/target_locations.size()*(d * d.transpose()).pruned();

    d2fdx2.makeCompressed();
    return d2fdx2;
}

Eigen::SparseMatrix<AScalar> ApproximateStiffnessTensorRelationship::Compute_d2fdxp(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> drdx(scene->x_dof(), 1); drdx.setZero();
    Eigen::SparseMatrix<AScalar> drdp(scene->parameter_dof(), 1); drdp.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;

    std::vector<bool> constrained(scene->x_dof());
    std::vector<int> constraints = scene->get_constraints();
    for(int j = 0; j < constraints.size(); ++j){
        constrained[constraints[j]] = true;
    }
    
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        for(int i = 0; i < drdx.size(); ++i){
            if(constrained[i]) continue;
            AScalar g = 0;
            g += sample_Cs_info[k].C_diff_x[i](0);
            g += (-2)*sample_Cs_info[k].C_diff_x[i](1);
            g += sample_Cs_info[k].C_diff_x[i](3);
            g += (-2)*sample_Cs_info[k].C_diff_x[i](1);
            
            if(std::abs(g) > 1e-7) drdx.coeffRef(i, 0) += g;
        }
        for(int i = 0; i < drdp.size(); ++i){
            AScalar g = 0;
            g += sample_Cs_info[k].C_diff_p[i](0);
            g += (-2)*sample_Cs_info[k].C_diff_p[i](1);
            g += sample_Cs_info[k].C_diff_p[i](3);
            g += (-2)*sample_Cs_info[k].C_diff_p[i](1);
   
            drdp.coeffRef(i, 0) += g;
        }
    }
    Eigen::SparseMatrix<AScalar> d2fdxp = WEIGHTS/target_locations.size()*(drdp *drdx.transpose()).pruned();
    d2fdxp.makeCompressed();
    return d2fdxp;
}

Eigen::SparseMatrix<AScalar> ApproximateStiffnessTensorRelationship::Compute_d2fdp2(Scene* scene){
    
    Eigen::SparseMatrix<AScalar> drdp(scene->parameter_dof(), 1); drdp.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector3a target_location = target_locations[k];
        for(int i = 0; i < drdp.size(); ++i){
            AScalar g = 0;
            g += sample_Cs_info[k].C_diff_p[i](0);
            g += (-2)*sample_Cs_info[k].C_diff_p[i](1);
            g += sample_Cs_info[k].C_diff_p[i](3);
            g += (-2)*sample_Cs_info[k].C_diff_p[i](1);
            drdp.coeffRef(i, 0) += g;
        }
    }
    Eigen::SparseMatrix<AScalar> d2fdp2 = WEIGHTS/target_locations.size()*(drdp * drdp.transpose()).pruned();

    d2fdp2.makeCompressed();
    return d2fdp2;
}

void ApproximateStiffnessTensorRelationship::SimulateAndCollect(Scene* scene){

    std::vector<Vector3a> directions;
    for(int i = 0; i < scene->num_directions; ++i) {
        AScalar angle = i*2*M_PI/scene->num_directions; 
        directions.push_back(Vector3a{std::cos(angle), std::sin(angle), 0});
    }
    scene->findBestCTensorviaProbing(target_locations, directions, true);
}
