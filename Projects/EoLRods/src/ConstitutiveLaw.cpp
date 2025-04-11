#include "../include/EoLRodSim.h"
#include "../include/Scene.h"
#include <fstream>

Matrix<T, 3, 3> EoLRodSim::computeGreenLagrangianStrain(const TV sample_loc, const std::vector<TV> line_directions){
    TM F = computeWeightedDeformationGradient(sample_loc, line_directions);
    return 0.5*(F.transpose()*F-TM::Identity());
}

Matrix<T, 3, 3> Scene::findBestCTensorviaProbing(TV sample_loc, const std::vector<TV> line_directions, bool opt){

    int num_test = 2;
    int c = 2*num_test;
    Eigen::MatrixXd n(3, c);
    Eigen::MatrixXd t(3, c);
    std::vector<Eigen::MatrixXd> t_diff(sim.Rods.size(), Eigen::MatrixXd(3,c));
    std::vector<Eigen::MatrixXd> n_diff_x(sim.deformed_states.rows(), Eigen::MatrixXd(3,c));
    std::vector<Eigen::MatrixXd> t_diff_x(sim.deformed_states.rows(), Eigen::MatrixXd(3,c));
    for(int i = 1; i <= num_test; ++i){
        EoLRodSim sim1;
        sim = sim1;
        if(mesh_file != "") buildFEMRodScene(mesh_file, 0, true);
        else buildGridScene(0, true);
        TV bottom_left, top_right;
        sim.computeUndeformedBoundingBox(bottom_left, top_right);
        if(sample_loc.norm() <= 0) {
            sample_loc = (bottom_left + top_right)/2;
            sample_loc[0] += ((top_right-bottom_left)*0.25)[0];
        }
        if(opt){
            for(auto rod: sim.Rods){
                rod->a = rods_radii(rod->rod_id);
                rod->b = rods_radii(rod->rod_id);
                rod->initCoeffs();
            }
        }

        T rec_width = 0.0001 * sim.unit;
        TV shear_x_right = TV(0.001*(i%2), 0.001*(i-1), 0.0) * sim.unit;
        TV shear_x_left = TV(0.0, 0.0, 0) * sim.unit;

        auto rec1 = [bottom_left, top_right, shear_x_left, rec_width](
            const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
        {
            mask = Vector<bool, 3>(true, true, true);
            delta = shear_x_left;
            if (x[0] < bottom_left[0] + rec_width)
                return true;
            return false;
        };

        auto rec2 = [bottom_left, top_right, shear_x_right, rec_width](
            const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
        {   
            // mask = Vector<bool, 3>(true, false, true);
            mask = Vector<bool, 3>(true, true, true);
            delta = shear_x_right;

            if (x[0] > top_right[0] - rec_width)
                return true;
            return false;
        };

        sim.fixRegionalDisplacement(rec2);
        sim.fixRegionalDisplacement(rec1);

        int static_solve_step = 0;
        bool finished = false;
        do{
            finished = sim.advanceOneStep(static_solve_step++);
        } while (!finished);

        Matrix<T, 3, 3> E = sim.computeGreenLagrangianStrain(sample_loc, line_directions);
        Matrix<T, 3, 3> S = sim.findBestStressTensorviaProbing(sample_loc, line_directions);

        t.col(i-1) = TV({S(0,0), S(1,1), 2*S(1,0)});
        n.col(i-1) = TV({E(0,0), E(1,1), 2*E(1,0)});
        for(int j = 0; j < t_diff.size(); ++j){
            t_diff[j].col(i-1) = sim.stress_gradients_wrt_rod_thickness[j];
        }
        for(int j = 0; j < n_diff_x.size(); ++j){
           t_diff_x[j].col(i-1) = sim.stress_gradients_wrt_x[j];
           Matrix<T, 3, 3> green_strain = 0.5*(sim.F_gradients_wrt_x[j].transpose()*sim.F_gradients_wrt_x[j] + sim.F_gradients_wrt_x[j].transpose()*sim.F_gradients_wrt_x[j]);
           n_diff_x[j].col(i-1) = TV({green_strain(0,0), green_strain(1,1), 2*green_strain(1,0)});
        }

        EoLRodSim sim2;
        sim = sim2;
        if(mesh_file != "") buildFEMRodScene(mesh_file, 0, true);
        else buildGridScene(0, true);

        if(opt){
            for(auto rod: sim.Rods){
                rod->a = rods_radii(rod->rod_id);
                rod->b = rods_radii(rod->rod_id);
                rod->initCoeffs();
            }
        }

        TV shear_x_down = TV(0.0, 0.0, 0.0) * sim.unit;
        TV shear_x_up = TV(0.001*(i-1), 0.001*(i%2), 0) * sim.unit;
        auto rec3 = [bottom_left, top_right, shear_x_down, rec_width](
            const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
        {
            // mask = Vector<bool, 3>(true, false, true);
            mask = Vector<bool, 3>(true, true, true);
            delta = shear_x_down;
            if (x[1] < bottom_left[1] + rec_width)
                return true;
            return false;
        };

        auto rec4 = [bottom_left, top_right, shear_x_up, rec_width](
            const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
        {   
            // mask = Vector<bool, 3>(true, false, true);
            mask = Vector<bool, 3>(true, true, true);
            delta = shear_x_up;

            if (x[1] > top_right[1] - rec_width)
                return true;
            return false;
        };
        sim.fixRegionalDisplacement(rec3);
        sim.fixRegionalDisplacement(rec4);

        static_solve_step = 0;
        sim.dq = VectorXT::Zero(sim.W.cols());
        do{
            finished = sim.advanceOneStep(static_solve_step++);
        } while (!finished);

        E = sim.computeGreenLagrangianStrain(sample_loc, line_directions);
        S = sim.findBestStressTensorviaProbing(sample_loc, line_directions);

        t.col(num_test+i-1) = TV({S(0,0), S(1,1), 2*S(1,0)});
        n.col(num_test+i-1) = TV({E(0,0), E(1,1), 2*E(1,0)});
        for(int j = 0; j < t_diff.size(); ++j){
            t_diff[j].col(num_test+i-1) = sim.stress_gradients_wrt_rod_thickness[j];
        }
        for(int j = 0; j < n_diff_x.size(); ++j){
            t_diff_x[j].col(num_test+i-1) = sim.stress_gradients_wrt_x[j];
            Matrix<T, 3, 3> green_strain = 0.5*(sim.F_gradients_wrt_x[j].transpose()*sim.F_gradients_wrt_x[j] + sim.F_gradients_wrt_x[j].transpose()*sim.F_gradients_wrt_x[j]);
            n_diff_x[j].col(num_test+i-1) = TV({green_strain(0,0), green_strain(1,1), 2*green_strain(1,0)});
        }
 
    }

    // for(int i = 0; i < c; ++i){
    //     std::cout << "Stress in strain: " << n.col(i).transpose() << " is : \n" << t.col(i).transpose() << std::endl;
    // }

    Matrix<T, 3, 3> fitted_tensor; fitted_tensor.setZero();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3*c,6);
    VectorXT b(3*c);
    std::vector<VectorXT> b_diff(t_diff.size(), VectorXT(3*c));
    std::vector<VectorXT> b_diff_x(t_diff_x.size(), VectorXT(3*c));
    std::vector<Eigen::MatrixXd> A_diff_x(n_diff_x.size(), Eigen::MatrixXd::Zero(3*c,6));
    for(int i = 0; i < c; ++i){
        Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(3,6);
        TV normal = n.col(i);
        A_block << normal(0), normal(1), normal(2), 0, 0, 0,
                0, normal(0), 0, normal(1), normal(2), 0,
                0, 0, normal(0), 0, normal(1), normal(2);
        A.block(i*3, 0, 3, 6) = A_block;
        b.segment(i*3, 3) = t.col(i);
        for(int j = 0; j < t_diff.size(); ++j){
            b_diff[j].segment(i*3, 3) = t_diff[j].col(i);
        }
        for(int j = 0; j < n_diff_x.size(); ++j){
            b_diff_x[j].segment(i*3, 3) = t_diff_x[j].col(i);
            Eigen::MatrixXd A_block_diff_x = Eigen::MatrixXd::Zero(3,6);
            TV normal_x = n_diff_x[j].col(i);
            A_block_diff_x << normal_x(0), normal_x(1), normal_x(2), 0, 0, 0,
                    0, normal_x(0), 0, normal_x(1), normal_x(2), 0,
                    0, 0, normal_x(0), 0, normal_x(1), normal_x(2);
            A_diff_x[j].block(i*3, 0, 3, 6) = A_block_diff_x;
        }
    }
    VectorXT x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    C_entry = x;
    C_diff_thickness = std::vector<VectorXT>(t_diff.size());
    for(int i = 0; i < t_diff.size(); ++i){
        C_diff_thickness[i] =  (A.transpose()*A).ldlt().solve(A.transpose()*b_diff[i]);
        // std::cout << C_diff_thickness[i].transpose() << std::endl;
    }
    C_diff_x = std::vector<VectorXT>(t_diff_x.size());
    for(int i = 0; i < t_diff_x.size(); ++i){
        C_diff_x[i] =  (A.transpose()*A).ldlt().solve(A_diff_x[i].transpose()*b+A.transpose()*b_diff_x[i]-(A_diff_x[i].transpose()*A+A.transpose()*A_diff_x[i])*C_entry);
        // if(C_diff_x[i].norm()>0) std::cout << C_diff_x[i].transpose() << std::endl;
    }  
    
    fitted_tensor << x(0), x(1), x(2), 
                    x(1), x(3), x(4),
                    x(2), x(4), x(5);              

    return fitted_tensor;

}

void Scene::optimizeForThickness(TV target_location, Vector<T, 6> stiffness_tensor, std::string filename){
    int num_directions = 8;
    std::vector<Eigen::Vector3d> directions;
    for(int i = 0; i < num_directions; ++i) {
        double angle = i*2*M_PI/num_directions; 
        directions.push_back(Eigen::Vector3d{std::cos(angle), std::sin(angle), 0});
    }

    double step = 1e-4;
    double tol = 0.01;
    double minVal = 0.0001;
    rods_radii.resize(sim.Rods.size());
    rods_radii.setConstant(4e-4);
    Matrix<T, 3, 3> C_current;
    for(int iter = 0; iter < 1000; ++iter){
        C_current = findBestCTensorviaProbing(target_location, directions, true);
        VectorXT gradient_wrt_thickness(sim.Rods.size());
        T scale = 1;
        for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
            T g = 0;
            for(int j = 0; j < 6; ++j){
                g += 2*(C_entry(j) - stiffness_tensor(j))/abs(stiffness_tensor(j))*C_diff_thickness[i](j);
            }
            while(abs(g)*scale > 0.1) scale /= 10;
            gradient_wrt_thickness(i) = g;
        }
        VectorXT gradient_wrt_x(sim.deformed_states.rows());
        for(int i = 0; i < gradient_wrt_x.size(); ++i){
            T g = 0;
            for(int j = 0; j < 6; ++j){
                g += 2*(C_entry(j) - stiffness_tensor(j))/abs(stiffness_tensor(j))*C_diff_x[i](j);
            }
            gradient_wrt_x(i) = g;
        }
        VectorXT total_gradient_wrt_thickness(sim.Rods.size());
        total_gradient_wrt_thickness = gradient_wrt_thickness + sim.solveAdjointForOptimization(gradient_wrt_x);
        // std::cout << total_gradient_wrt_thickness.transpose();

        if(((C_entry - stiffness_tensor).lpNorm<Eigen::Infinity>())/abs(stiffness_tensor.minCoeff()) < tol || total_gradient_wrt_thickness.norm() < tol) {
            C_current = findBestCTensorviaProbing(target_location, directions, true);
            std::cout << "Found solution at step: " << iter << std::endl; 
            break;
        }
        rods_radii -= step * scale * total_gradient_wrt_thickness;
        rods_radii = rods_radii.cwiseMax(minVal);
    }
    
    std::cout << rods_radii.transpose() << std::endl; 
    // std::ofstream out_file(filename+"_radii.dat");
    // if (!out_file) {
    //     std::cerr << "Error opening file for writing: " << filename << std::endl;
    //     return;
    // }
    // out_file << rods_radii << "\n";
    // out_file.close();

    // std::ofstream out(filename+"_C.dat");
    // out << "Optimization location: " << target_location.transpose() << std::endl;
    // out << "Optimization target C (DoF): \n" << stiffness_tensor.transpose() << std::endl;
    // out << "Optimization result C: \n" << C_current << "\n";
    // out.close();
    std::cout << "Optimized C: \n" << C_current << std::endl;
    
    
}

void Scene::optimizeForThicknessDistribution(const std::vector<TV> target_locations, const std::vector<Vector<T, 6>> stiffness_tensors, const std::string filename){
    assert(target_locations.size() == stiffness_tensors.size());

    int num_directions = 20;
    std::vector<Eigen::Vector3d> directions;
    for(int i = 0; i < num_directions; ++i) {
        double angle = i*2*M_PI/num_directions; 
        directions.push_back(Eigen::Vector3d{std::cos(angle), std::sin(angle), 0});
    }

    double step = 7e-5;
    double tol = 0.01;
    double minVal = 0.0001;
    rods_radii.resize(sim.Rods.size());
    rods_radii.setConstant(4e-4);
    std::vector<Matrix<T, 3, 3>> C_current(target_locations.size());
    for(int iter = 0; iter < 760; ++iter){
        VectorXT gradient_wrt_thickness(sim.Rods.size()); gradient_wrt_thickness.setZero();
        VectorXT gradient_wrt_x(sim.deformed_states.rows()); gradient_wrt_x.setZero();
        T scale = 1;
        for(int k = 0; k < target_locations.size(); ++k){
            TV target_location = target_locations[k];
            Vector<T, 6> target_C = stiffness_tensors[k];
            C_current[k] = findBestCTensorviaProbing(target_location, directions, true);
            for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
                T g = 0;
                for(int j = 0; j < 6; ++j){
                    g += 2*(C_entry(j) - target_C(j))/abs(target_C(j))*C_diff_thickness[i](j);
                }
                while(abs(g)*scale > 0.1) scale /= 10;
                gradient_wrt_thickness(i) += g;
            }
            for(int i = 0; i < gradient_wrt_x.size(); ++i){
                T g = 0;
                for(int j = 0; j < 6; ++j){
                    g += 2*(C_entry(j) - target_C(j))/abs(target_C(j))*C_diff_x[i](j);
                }
                gradient_wrt_x(i) += g;
            }
        }
        VectorXT total_gradient_wrt_thickness(sim.Rods.size());
        total_gradient_wrt_thickness = gradient_wrt_thickness + sim.solveAdjointForOptimization(gradient_wrt_x);
        // std::cout << total_gradient_wrt_thickness.transpose();

        if(total_gradient_wrt_thickness.norm()*scale*10 < tol) {
            for(int i = 0; i < target_locations.size(); ++i){
                TV target_location = target_locations[i];
                C_current[i] = findBestCTensorviaProbing(target_location, directions, true);
            }
            std::cout << "Found solution at step: " << iter << std::endl; 
            break;
        }
        rods_radii -= step * scale * total_gradient_wrt_thickness;
        rods_radii = rods_radii.cwiseMax(minVal);
    }

    std::ofstream out(filename+"_C.dat");
    for(int i = 0; i < target_locations.size(); ++i){
        C_current[i] = findBestCTensorviaProbing(target_locations[i], directions, true);
        // std::cout << "Optimized C for location "<< target_locations[i].transpose() << ": \n" << C_current[i] << std::endl;
        out << "Optimization location: " << target_locations[i].transpose() << std::endl;
        out << "Optimization target C (DoF): \n" << stiffness_tensors[i].transpose() << std::endl;
        out << "Optimization result C: \n" << C_current[i] << "\n";
    }
    out.close();
    std::cout << rods_radii.transpose() << std::endl; 
    std::ofstream out_file(filename+"_radii.dat");
    if (!out_file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    out_file << rods_radii << "\n";
    out_file.close();
}