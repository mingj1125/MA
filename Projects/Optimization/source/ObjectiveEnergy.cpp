#include "../include/ObjectiveEnergy.h"
#include "../include/enforce_matrix_constraints.h"

AScalar WEIGHTS = 1e-8;

AScalar ApproximateTargetStiffnessTensor::ComputeEnergy(Scene* scene){

    AScalar energy = 0;
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector6a target_C = target_stiffness_tensors[k];
        for(int j = 0; j < 6; ++j){

             energy += consider_entry(j)*0.5*(sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))/abs(target_C(j))*(sample_Cs_info[k].C_entry(j) - target_C(j));

        }
       
    }    
    return energy;
}

VectorXa ApproximateTargetStiffnessTensor::Compute_dfdx_sim(Scene* scene){
    
    VectorXa gradient_wrt_x(scene->x_dof()*scene->num_test); gradient_wrt_x.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;

    for(int test = 0; test < scene->num_test; ++test){
        std::vector<bool> constrained(scene->x_dof());
        std::vector<int> constraints = scene->get_constraint_sim(test);
        for(int j = 0; j < constraints.size(); ++j){
            constrained[constraints[j]] = true;
        }
        for(int k = 0; k < target_locations.size(); ++k){
            Vector6a target_C = target_stiffness_tensors[k];
            for(int i = 0; i < scene->x_dof(); ++i){
                if(constrained[i]) continue;
                // if(sample_Cs_info[k].C_diff_x_sim[i].norm() > 0) std::cout << i << " dCdx : \n"<< sample_Cs_info[k].C_diff_x_sim[i] << std::endl;
    
                AScalar g = 0;
                for(int j = 0; j < 6; ++j){
    
                    g += consider_entry(j)*(sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_x_sim[i](j, test);
    
                }
                gradient_wrt_x(scene->x_dof()*test+ i) += g;
            }
        }
    }
    
    return gradient_wrt_x;
}

VectorXa ApproximateTargetStiffnessTensor::Compute_dfdp(Scene* scene){
    
    VectorXa gradient_wrt_thickness(scene->parameter_dof()); gradient_wrt_thickness.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        Vector6a target_C = target_stiffness_tensors[k];
        for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                g += consider_entry(j)*(sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_p[i](j);

            }
            gradient_wrt_thickness(i) += g;
        }
    }

    return gradient_wrt_thickness;
}

std::vector<Eigen::SparseMatrix<AScalar>> ApproximateTargetStiffnessTensor::Compute_d2fdx2(Scene* scene){
    
    std::vector<Eigen::SparseMatrix<AScalar>> d2fdx2(scene->num_test, Eigen::SparseMatrix<AScalar>(scene->x_dof(), scene->x_dof()));
    auto sample_Cs_info = scene->sample_Cs_info;

    for(int test = 0; test < scene->num_test; ++test){
        Eigen::SparseMatrix<AScalar> drdx(scene->x_dof(), 1); drdx.setZero();
        std::vector<bool> constrained(scene->x_dof());
        std::vector<int> constraints = scene->get_constraint_sim(test);
        for(int j = 0; j < constraints.size(); ++j){
            constrained[constraints[j]] = true;
        }

        for(int k = 0; k < target_locations.size(); ++k){
            Vector6a target_C = target_stiffness_tensors[k];
            for(int i = 0; i < drdx.size(); ++i){
                if(constrained[i]) continue;
                AScalar g = 0;
                for(int j = 0; j < 6; ++j){

                    g += consider_entry(j)*sample_Cs_info[k].C_diff_x_sim[i](j, test)/abs(target_C(j));

                }
                drdx.coeffRef(i, 0) += g;
            }
        }
        d2fdx2[test] = (drdx * drdx.transpose()).pruned();
    }

    return d2fdx2;
}

std::vector<Eigen::SparseMatrix<AScalar>> ApproximateTargetStiffnessTensor::Compute_d2fdxp(Scene* scene){
    
    std::vector<Eigen::SparseMatrix<AScalar>> d2fdxp(scene->num_test, Eigen::SparseMatrix<AScalar>(scene->parameter_dof(), scene->x_dof()));
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int test = 0; test < scene->num_test; ++test){
        Eigen::SparseMatrix<AScalar> drdx(scene->x_dof(), 1); drdx.setZero();
        Eigen::SparseMatrix<AScalar> drdp(scene->parameter_dof(), 1); drdp.setZero();

        std::vector<bool> constrained(scene->x_dof());
        std::vector<int> constraints = scene->get_constraints();
        for(int j = 0; j < constraints.size(); ++j){
            constrained[constraints[j]] = true;
        }

        for(int k = 0; k < target_locations.size(); ++k){
            Vector6a target_C = target_stiffness_tensors[k];
            for(int i = 0; i < drdx.size(); ++i){
                if(constrained[i]) continue;
                AScalar g = 0;
                for(int j = 0; j < 6; ++j){

                    g += consider_entry(j)*sample_Cs_info[k].C_diff_x_sim[i](j, test)/abs(target_C(j));

                }
                
                if(std::abs(g) > 1e-7) drdx.coeffRef(i, 0) += g;
            }
            for(int i = 0; i < drdp.size(); ++i){
                AScalar g = 0;
                for(int j = 0; j < 6; ++j){

                    g += consider_entry(j)*sample_Cs_info[k].C_diff_p[i](j)/abs(target_C(j));

                }
                if(std::abs(g) > 1e-7) drdp.coeffRef(i, 0) += g;
            }
        }
        d2fdxp[test] = (drdp * drdx.transpose()).pruned();
    }    

    return d2fdxp;
}

std::vector<Eigen::SparseMatrix<AScalar>> ApproximateTargetStiffnessTensor::Compute_d2fdp2(Scene* scene){
    
    std::vector<Eigen::SparseMatrix<AScalar>> d2fdp2(scene->num_test, Eigen::SparseMatrix<AScalar>(scene->parameter_dof(), scene->parameter_dof()));
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int test = 0; test < scene->num_test; ++test){
        Eigen::SparseMatrix<AScalar> drdp(scene->parameter_dof(), 1); drdp.setZero();
        for(int k = 0; k < target_locations.size(); ++k){
            Vector6a target_C = target_stiffness_tensors[k];
            for(int i = 0; i < drdp.size(); ++i){
                AScalar g = 0;
                for(int j = 0; j < 6; ++j){

                    g += consider_entry(j)*sample_Cs_info[k].C_diff_p[i](j)/abs(target_C(j));

                }
                drdp.coeffRef(i, 0) += g;
            }
        }
        d2fdp2[test] = (drdp * drdp.transpose()).pruned();
    }

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

void ApproximateTargetStiffnessTensor::OnlyCollect(Scene* scene, MatrixXa offsets){

    std::vector<Vector3a> directions;
    for(int i = 0; i < scene->num_directions; ++i) {
        AScalar angle = i*2*M_PI/scene->num_directions; 
        directions.push_back(Vector3a{std::cos(angle), std::sin(angle), 0});
    }
    scene->CTensorPerturbx(target_locations, directions, offsets);
}


AScalar ApproximateStiffnessTensorRelationship::ComputeEnergy(Scene* scene){

    AScalar energy = 0;
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
      
        AScalar r = sample_Cs_info[k].C_entry(0) - ratio*sample_Cs_info[k].C_entry(1);
        energy += 0.5*r*r;
        r = sample_Cs_info[k].C_entry(3) - ratio*sample_Cs_info[k].C_entry(1);
        energy += 0.5*r*r;
    }
    energy *= WEIGHTS / target_locations.size();    
    return energy;
}

VectorXa ApproximateStiffnessTensorRelationship::Compute_dfdx_sim(Scene* scene){
    
    VectorXa gradient_wrt_x(scene->x_dof()*scene->num_test); gradient_wrt_x.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;

    for(int test = 0; test < scene->num_test; ++test){
        std::vector<bool> constrained(scene->x_dof());
        std::vector<int> constraints = scene->get_constraint_sim(test);
        for(int j = 0; j < constraints.size(); ++j){
            constrained[constraints[j]] = true;
        }
        for(int k = 0; k < target_locations.size(); ++k){
            for(int i = 0; i < scene->x_dof(); ++i){
                if(constrained[i]) continue;
    
                AScalar g = 0;
                g += (sample_Cs_info[k].C_entry(0) -  ratio*sample_Cs_info[k].C_entry(1))*sample_Cs_info[k].C_diff_x_sim[i](0, test);
                g += (sample_Cs_info[k].C_entry(0) -  ratio*sample_Cs_info[k].C_entry(1))*(-ratio)*sample_Cs_info[k].C_diff_x_sim[i](1, test);
                g += (sample_Cs_info[k].C_entry(3) -  ratio*sample_Cs_info[k].C_entry(1))*sample_Cs_info[k].C_diff_x_sim[i](3, test);
                g += (sample_Cs_info[k].C_entry(3) -  ratio*sample_Cs_info[k].C_entry(1))*(-ratio)*sample_Cs_info[k].C_diff_x_sim[i](1, test);
    
                gradient_wrt_x(scene->x_dof()*test+ i) += g;
            }
        }
    }
    
    return gradient_wrt_x*WEIGHTS/target_locations.size();
}

VectorXa ApproximateStiffnessTensorRelationship::Compute_dfdp(Scene* scene){
    
    VectorXa gradient_wrt_thickness(scene->parameter_dof()); gradient_wrt_thickness.setZero();
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int k = 0; k < target_locations.size(); ++k){
        for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
            AScalar g = 0;

            g += (sample_Cs_info[k].C_entry(0) -  ratio*sample_Cs_info[k].C_entry(1))*sample_Cs_info[k].C_diff_p[i](0);
            g += (sample_Cs_info[k].C_entry(0) -  ratio*sample_Cs_info[k].C_entry(1))*(-ratio)*sample_Cs_info[k].C_diff_p[i](1);
            g += (sample_Cs_info[k].C_entry(3) -  ratio*sample_Cs_info[k].C_entry(1))*sample_Cs_info[k].C_diff_p[i](3);
            g += (sample_Cs_info[k].C_entry(3) -  ratio*sample_Cs_info[k].C_entry(1))*(-ratio)*sample_Cs_info[k].C_diff_p[i](1);
  
            gradient_wrt_thickness(i) += g*WEIGHTS/target_locations.size();
        }
    }

    return gradient_wrt_thickness;
}

std::vector<Eigen::SparseMatrix<AScalar>> ApproximateStiffnessTensorRelationship::Compute_d2fdx2(Scene* scene){
    
    std::vector<Eigen::SparseMatrix<AScalar>> d2fdx2(scene->num_test, Eigen::SparseMatrix<AScalar>(scene->x_dof(), scene->x_dof()));
    auto sample_Cs_info = scene->sample_Cs_info;

    for(int test = 0; test < scene->num_test; ++test){
        Eigen::SparseMatrix<AScalar> drdx(scene->x_dof(), 1); drdx.setZero();
        std::vector<bool> constrained(scene->x_dof());
        std::vector<int> constraints = scene->get_constraint_sim(test);
        for(int j = 0; j < constraints.size(); ++j){
            constrained[constraints[j]] = true;
        }

        for(int k = 0; k < target_locations.size(); ++k){
            for(int i = 0; i < drdx.size(); ++i){
                if(constrained[i]) continue;
                AScalar g = 0;
                g += sample_Cs_info[k].C_diff_x_sim[i](0, test);
                g += (-ratio)*sample_Cs_info[k].C_diff_x_sim[i](1, test);
                g += sample_Cs_info[k].C_diff_x_sim[i](3, test);
                g += (-ratio)*sample_Cs_info[k].C_diff_x_sim[i](1, test);

                drdx.coeffRef(i, 0) += g;
            }
        }
        d2fdx2[test] = WEIGHTS/target_locations.size()*(drdx * drdx.transpose()).pruned();
    }

    return d2fdx2;
}

std::vector<Eigen::SparseMatrix<AScalar>> ApproximateStiffnessTensorRelationship::Compute_d2fdxp(Scene* scene){
    
    std::vector<Eigen::SparseMatrix<AScalar>> d2fdxp(scene->num_test, Eigen::SparseMatrix<AScalar>(scene->parameter_dof(), scene->x_dof()));
    auto sample_Cs_info = scene->sample_Cs_info;

    for(int test = 0; test < scene->num_test; ++test){
        Eigen::SparseMatrix<AScalar> drdx(scene->x_dof(), 1); drdx.setZero();
        Eigen::SparseMatrix<AScalar> drdp(scene->parameter_dof(), 1); drdp.setZero();

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
                g += sample_Cs_info[k].C_diff_x_sim[i](0, test);
                g += (-ratio)*sample_Cs_info[k].C_diff_x_sim[i](1, test);
                g += sample_Cs_info[k].C_diff_x_sim[i](3, test);
                g += (-ratio)*sample_Cs_info[k].C_diff_x_sim[i](1, test);
                
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
        d2fdxp[test] = WEIGHTS/target_locations.size()*(drdp *drdx.transpose()).pruned();
    }
    return d2fdxp;
}

std::vector<Eigen::SparseMatrix<AScalar>> ApproximateStiffnessTensorRelationship::Compute_d2fdp2(Scene* scene){
    
    std::vector<Eigen::SparseMatrix<AScalar>> d2fdp2(scene->num_test, Eigen::SparseMatrix<AScalar>(scene->parameter_dof(), scene->parameter_dof()));
    auto sample_Cs_info = scene->sample_Cs_info;
    for(int test = 0; test < scene->num_test; ++test){
        Eigen::SparseMatrix<AScalar> drdp(scene->parameter_dof(), 1); drdp.setZero();
        for(int k = 0; k < target_locations.size(); ++k){
            for(int i = 0; i < drdp.size(); ++i){
                AScalar g = 0;
                g += sample_Cs_info[k].C_diff_p[i](0);
                g += (-ratio)*sample_Cs_info[k].C_diff_p[i](1);
                g += sample_Cs_info[k].C_diff_p[i](3);
                g += (-ratio)*sample_Cs_info[k].C_diff_p[i](1);
                drdp.coeffRef(i, 0) += g;
            }
        }
        d2fdp2[test] = WEIGHTS/target_locations.size()*(drdp * drdp.transpose()).pruned();
    }
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

void ApproximateStiffnessTensorRelationship::OnlyCollect(Scene* scene, MatrixXa offsets){

    std::vector<Vector3a> directions;
    for(int i = 0; i < scene->num_directions; ++i) {
        AScalar angle = i*2*M_PI/scene->num_directions; 
        directions.push_back(Vector3a{std::cos(angle), std::sin(angle), 0});
    }
    scene->CTensorPerturbx(target_locations, directions, offsets);
}

AScalar ApproximateTargetStiffnessTensorWindow::ComputeEnergy(Scene* scene){

    AScalar energy = 0;
    auto window_Cs_info = scene->window_Cs_info;
    for(int k = 0; k < target_corners.size(); ++k){
        Vector6a target_C = target_stiffness_tensors[k];
        for(int j = 0; j < 6; ++j){

            if(consider_entry(j) > 0.0) energy += consider_entry(j)*0.5*(window_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))/abs(target_C(j))*(window_Cs_info[k].C_entry(j) - target_C(j));

        }
       
    }    
    return energy;
}

VectorXa ApproximateTargetStiffnessTensorWindow::Compute_dfdx_sim(Scene* scene){
    
    VectorXa gradient_wrt_x(scene->x_dof()*scene->num_test); gradient_wrt_x.setZero();
    auto window_Cs_info = scene->window_Cs_info;

    for(int test = 0; test < scene->num_test; ++test){
        std::vector<bool> constrained(scene->x_dof());
        std::vector<int> constraints = scene->get_constraint_sim(test);
        for(int j = 0; j < constraints.size(); ++j){
            constrained[constraints[j]] = true;
        }
        for(int k = 0; k < target_corners.size(); ++k){
            Vector6a target_C = target_stiffness_tensors[k];
            for(int i = 0; i < scene->x_dof(); ++i){
                if(constrained[i]) continue;
                // if(sample_Cs_info[k].C_diff_x_sim[i].norm() > 0) std::cout << i << " dCdx : \n"<< sample_Cs_info[k].C_diff_x_sim[i] << std::endl;
    
                AScalar g = 0;
                for(int j = 0; j < 6; ++j){
    
                    if(consider_entry(j) > 0.0) g += consider_entry(j)*(window_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))/abs(target_C(j))*window_Cs_info[k].C_diff_x_sim[i](j, test);
    
                }
                gradient_wrt_x(scene->x_dof()*test+ i) += g;
            }
        }
    }
    
    return gradient_wrt_x;
}

VectorXa ApproximateTargetStiffnessTensorWindow::Compute_dfdp(Scene* scene){
    
    VectorXa gradient_wrt_thickness(scene->parameter_dof()); gradient_wrt_thickness.setZero();
    auto sample_Cs_info = scene->window_Cs_info;
    for(int k = 0; k < target_corners.size(); ++k){
        Vector6a target_C = target_stiffness_tensors[k];
        for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
            AScalar g = 0;
            for(int j = 0; j < 6; ++j){

                if(consider_entry(j) > 0.0) g += consider_entry(j)*(sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_p[i](j);

            }
            gradient_wrt_thickness(i) += g;
        }
    }

    return gradient_wrt_thickness;
}

std::vector<Eigen::SparseMatrix<AScalar>> ApproximateTargetStiffnessTensorWindow::Compute_d2fdx2(Scene* scene){
    
    std::vector<Eigen::SparseMatrix<AScalar>> d2fdx2(scene->num_test, Eigen::SparseMatrix<AScalar>(scene->x_dof(), scene->x_dof()));
    auto sample_Cs_info = scene->window_Cs_info;

    for(int test = 0; test < scene->num_test; ++test){
        Eigen::SparseMatrix<AScalar> drdx(scene->x_dof(), 1); drdx.setZero();
        std::vector<bool> constrained(scene->x_dof());
        std::vector<int> constraints = scene->get_constraint_sim(test);
        for(int j = 0; j < constraints.size(); ++j){
            constrained[constraints[j]] = true;
        }

        for(int k = 0; k < target_corners.size(); ++k){
            Vector6a target_C = target_stiffness_tensors[k];
            for(int i = 0; i < drdx.size(); ++i){
                if(constrained[i]) continue;
                AScalar g = 0;
                for(int j = 0; j < 6; ++j){

                    if(consider_entry(j) > 0.0) g += consider_entry(j)*sample_Cs_info[k].C_diff_x_sim[i](j, test)/abs(target_C(j));

                }
                drdx.coeffRef(i, 0) += g;
            }
        }
        d2fdx2[test] = (drdx * drdx.transpose()).pruned();
    }

    return d2fdx2;
}

std::vector<Eigen::SparseMatrix<AScalar>> ApproximateTargetStiffnessTensorWindow::Compute_d2fdxp(Scene* scene){
    
    std::vector<Eigen::SparseMatrix<AScalar>> d2fdxp(scene->num_test, Eigen::SparseMatrix<AScalar>(scene->parameter_dof(), scene->x_dof()));
    auto sample_Cs_info = scene->window_Cs_info;
    for(int test = 0; test < scene->num_test; ++test){
        Eigen::SparseMatrix<AScalar> drdx(scene->x_dof(), 1); drdx.setZero();
        Eigen::SparseMatrix<AScalar> drdp(scene->parameter_dof(), 1); drdp.setZero();

        std::vector<bool> constrained(scene->x_dof());
        std::vector<int> constraints = scene->get_constraints();
        for(int j = 0; j < constraints.size(); ++j){
            constrained[constraints[j]] = true;
        }

        for(int k = 0; k < target_corners.size(); ++k){
            Vector6a target_C = target_stiffness_tensors[k];
            for(int i = 0; i < drdx.size(); ++i){
                if(constrained[i]) continue;
                AScalar g = 0;
                for(int j = 0; j < 6; ++j){

                    if(consider_entry(j) > 0.0) g += consider_entry(j)*sample_Cs_info[k].C_diff_x_sim[i](j, test)/abs(target_C(j));

                }
                
                if(std::abs(g) > 1e-7) drdx.coeffRef(i, 0) += g;
            }
            for(int i = 0; i < drdp.size(); ++i){
                AScalar g = 0;
                for(int j = 0; j < 6; ++j){

                    if(consider_entry(j) > 0.0) g += consider_entry(j)*sample_Cs_info[k].C_diff_p[i](j)/abs(target_C(j));

                }
                if(std::abs(g) > 1e-7) drdp.coeffRef(i, 0) += g;
            }
        }
        d2fdxp[test] = (drdp * drdx.transpose()).pruned();
    }    

    return d2fdxp;
}

std::vector<Eigen::SparseMatrix<AScalar>> ApproximateTargetStiffnessTensorWindow::Compute_d2fdp2(Scene* scene){
    
    std::vector<Eigen::SparseMatrix<AScalar>> d2fdp2(scene->num_test, Eigen::SparseMatrix<AScalar>(scene->parameter_dof(), scene->parameter_dof()));
    auto sample_Cs_info = scene->window_Cs_info;
    for(int test = 0; test < scene->num_test; ++test){
        Eigen::SparseMatrix<AScalar> drdp(scene->parameter_dof(), 1); drdp.setZero();
        for(int k = 0; k < target_corners.size(); ++k){
            Vector6a target_C = target_stiffness_tensors[k];
            for(int i = 0; i < drdp.size(); ++i){
                AScalar g = 0;
                for(int j = 0; j < 6; ++j){

                    if(consider_entry(j) > 0.0) g += consider_entry(j)*sample_Cs_info[k].C_diff_p[i](j)/abs(target_C(j));

                }
                drdp.coeffRef(i, 0) += g;
            }
        }
        d2fdp2[test] = (drdp * drdp.transpose()).pruned();
    }

    return d2fdp2;
}

void ApproximateTargetStiffnessTensorWindow::SimulateAndCollect(Scene* scene){

    scene->findCTensorInWindow(target_corners, true);
}