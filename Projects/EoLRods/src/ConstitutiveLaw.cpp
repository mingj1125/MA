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

struct stress_strain_relationship{

    stress_strain_relationship(int r, int c, int d){
        n.resize(3, c);
        t.resize(3, c);
        t_diff = std::vector<Eigen::MatrixXd>(r, Eigen::MatrixXd(3,c));
        n_diff_x = std::vector<Eigen::MatrixXd>(d, Eigen::MatrixXd(3,c));
        t_diff_x = std::vector<Eigen::MatrixXd> (d, Eigen::MatrixXd(3,c));
    }
    Eigen::MatrixXd n;
    Eigen::MatrixXd t;
    std::vector<Eigen::MatrixXd> t_diff;
    std::vector<Eigen::MatrixXd> n_diff_x;
    std::vector<Eigen::MatrixXd> t_diff_x;

};

void Scene::findBestCTensorviaProbing(std::vector<TV> sample_locs, const std::vector<TV> line_directions, bool opt){

    int num_test = 2;
    int c = 2*num_test;
    std::vector<stress_strain_relationship> samples(sample_locs.size(), stress_strain_relationship(sim.Rods.size(), c, sim.deformed_states.rows()));
    sample_Cs_info = std::vector<C_info>(sample_locs.size()); 
    // Eigen::MatrixXd n(3, c);
    // Eigen::MatrixXd t(3, c);
    // std::vector<Eigen::MatrixXd> t_diff(sim.Rods.size(), Eigen::MatrixXd(3,c));
    // std::vector<Eigen::MatrixXd> n_diff_x(sim.deformed_states.rows(), Eigen::MatrixXd(3,c));
    // std::vector<Eigen::MatrixXd> t_diff_x(sim.deformed_states.rows(), Eigen::MatrixXd(3,c));
    for(int i = 1; i <= num_test; ++i){
        EoLRodSim sim1;
        sim = sim1;
        if(mesh_file != "") buildFEMRodScene(mesh_file, 0, true);
        else buildGridScene(0, true);
        TV bottom_left, top_right;
        sim.computeUndeformedBoundingBox(bottom_left, top_right);
        if(opt){
            for(auto rod: sim.Rods){
                rod->a = rods_radii(rod->rod_id);
                rod->b = rods_radii(rod->rod_id);
                rod->initCoeffs();
            }
        }

        T rec_width = 0.0001 * sim.unit;
        TV shear_x_right = TV(0.001*(i-1), 0.001*(i%2), 0.0) * sim.unit;
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

        for(int l = 0; l < sample_locs.size(); ++l){
            Matrix<T, 3, 3> E = sim.computeGreenLagrangianStrain(sample_locs[l], line_directions);
            Matrix<T, 3, 3> S = sim.findBestStressTensorviaProbing(sample_locs[l], line_directions);
            samples[l].t.col(i-1) = TV({S(0,0), S(1,1), 2*S(1,0)});
            samples[l].n.col(i-1) = TV({E(0,0), E(1,1), 2*E(1,0)});
            for(int j = 0; j < (samples[l].t_diff).size(); ++j){
                samples[l].t_diff[j].col(i-1) = sim.stress_gradients_wrt_rod_thickness[j];
            }
            for(int j = 0; j < samples[l].n_diff_x.size(); ++j){
               samples[l].t_diff_x[j].col(i-1) = sim.stress_gradients_wrt_x[j];
               Matrix<T, 3, 3> green_strain = 0.5*(sim.F_gradients_wrt_x[j].transpose()*sim.F_gradients_wrt_x[j] + sim.F_gradients_wrt_x[j].transpose()*sim.F_gradients_wrt_x[j]);
               samples[l].n_diff_x[j].col(i-1) = TV({green_strain(0,0), green_strain(1,1), 2*green_strain(1,0)});
            }
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
        TV shear_x_up = TV(0.001*(i%2), 0.001*(i-1), 0) * sim.unit;
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

        for(int l = 0; l < sample_locs.size(); ++l){
            Matrix<T, 3, 3> E = sim.computeGreenLagrangianStrain(sample_locs[l], line_directions);
            Matrix<T, 3, 3> S = sim.findBestStressTensorviaProbing(sample_locs[l], line_directions);
            samples[l].t.col(num_test+i-1) = TV({S(0,0), S(1,1), 2*S(1,0)});
            samples[l].n.col(num_test+i-1) = TV({E(0,0), E(1,1), 2*E(1,0)});
            for(int j = 0; j < samples[l].t_diff.size(); ++j){
                samples[l].t_diff[j].col(num_test+i-1) = sim.stress_gradients_wrt_rod_thickness[j];
            }
            for(int j = 0; j < samples[l].n_diff_x.size(); ++j){
               samples[l].t_diff_x[j].col(num_test+i-1) = sim.stress_gradients_wrt_x[j];
               Matrix<T, 3, 3> green_strain = 0.5*(sim.F_gradients_wrt_x[j].transpose()*sim.F_gradients_wrt_x[j] + sim.F_gradients_wrt_x[j].transpose()*sim.F_gradients_wrt_x[j]);
               samples[l].n_diff_x[j].col(num_test+i-1) = TV({green_strain(0,0), green_strain(1,1), 2*green_strain(1,0)});
            }
        }
 
    }

    for(int l = 0; l < sample_locs.size(); ++l){
        Matrix<T, 3, 3> fitted_tensor; fitted_tensor.setZero();
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3*c,6);
        VectorXT b(3*c);
        std::vector<VectorXT> b_diff(samples[l].t_diff.size(), VectorXT(3*c));
        std::vector<VectorXT> b_diff_x(samples[l].t_diff_x.size(), VectorXT(3*c));
        std::vector<Eigen::MatrixXd> A_diff_x(samples[l].n_diff_x.size(), Eigen::MatrixXd::Zero(3*c,6));
        for(int i = 0; i < c; ++i){
            Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(3,6);
            TV normal = samples[l].n.col(i);
            A_block << normal(0), normal(1), normal(2), 0, 0, 0,
                    0, normal(0), 0, normal(1), normal(2), 0,
                    0, 0, normal(0), 0, normal(1), normal(2);
            A.block(i*3, 0, 3, 6) = A_block;
            b.segment(i*3, 3) = samples[l].t.col(i);
            for(int j = 0; j < samples[l].t_diff.size(); ++j){
                b_diff[j].segment(i*3, 3) = samples[l].t_diff[j].col(i);
            }
            for(int j = 0; j < samples[l].n_diff_x.size(); ++j){
                b_diff_x[j].segment(i*3, 3) = samples[l].t_diff_x[j].col(i);
                Eigen::MatrixXd A_block_diff_x = Eigen::MatrixXd::Zero(3,6);
                TV normal_x = samples[l].n_diff_x[j].col(i);
                A_block_diff_x << normal_x(0), normal_x(1), normal_x(2), 0, 0, 0,
                        0, normal_x(0), 0, normal_x(1), normal_x(2), 0,
                        0, 0, normal_x(0), 0, normal_x(1), normal_x(2);
                A_diff_x[j].block(i*3, 0, 3, 6) = A_block_diff_x;
            }
        }
        VectorXT x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
        sample_Cs_info[l].C_entry = x;
        sample_Cs_info[l].C_diff_thickness = std::vector<VectorXT>(samples[l].t_diff.size());
        for(int i = 0; i < samples[l].t_diff.size(); ++i){
            sample_Cs_info[l].C_diff_thickness[i] =  (A.transpose()*A).ldlt().solve(A.transpose()*b_diff[i]);
        }
        sample_Cs_info[l].C_diff_x = std::vector<VectorXT>(samples[l].t_diff_x.size());
        for(int i = 0; i < samples[l].t_diff_x.size(); ++i){
            sample_Cs_info[l].C_diff_x[i] =  (A.transpose()*A).ldlt().solve(A_diff_x[i].transpose()*b+A.transpose()*b_diff_x[i]-(A_diff_x[i].transpose()*A+A.transpose()*A_diff_x[i])*sample_Cs_info[l].C_entry);
        }  
    }
    for(int l = 0; l < sample_locs.size(); ++l)
        std::cout << sample_Cs_info[l].C_entry.transpose() << std::endl;

}

void Scene::optimizeForThickness(TV target_location, Vector<T, 6> stiffness_tensor, std::string filename){
    int num_directions = 20;
    std::vector<Eigen::Vector3d> directions;
    for(int i = 0; i < num_directions; ++i) {
        double angle = i*2*M_PI/num_directions; 
        directions.push_back(Eigen::Vector3d{std::cos(angle), std::sin(angle), 0});
    }

    double step = 5e-3;
    double tol = 0.01;
    double minVal = 0.0001;
    int ls_max = 5;
    rods_radii.resize(sim.Rods.size());
    rods_radii.setConstant(6e-3);
    Matrix<T, 3, 3> C_current;
    Vector<T, 6> C_current_entry;
    T err;
    for(int iter = 0; iter < 1; ++iter){
        C_current = findBestCTensorviaProbing(target_location, directions, true);
        C_current_entry = C_entry;
        VectorXT gradient_wrt_thickness(sim.Rods.size());
        T scale = 1;
        T E0 = 0;
        for(int j = 0; j < 6; ++j){
            E0 += 0.5*(C_entry(j) - stiffness_tensor(j))/std::abs(stiffness_tensor(j))*(C_entry(j) - stiffness_tensor(j));
        }
        for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
            T g = 0;
            for(int j = 0; j < 6; ++j){
                g += (C_entry(j) - stiffness_tensor(j))/std::abs(stiffness_tensor(j))*C_diff_thickness[i](j);
            }
            while(std::abs(g)*scale > 1) scale /= 10;
            gradient_wrt_thickness(i) = g;
        }
        VectorXT gradient_wrt_x(sim.deformed_states.rows());
        for(int i = 0; i < gradient_wrt_x.size(); ++i){
            T g = 0;
            for(int j = 0; j < 6; ++j){
                g += (C_entry(j) - stiffness_tensor(j))/std::abs(stiffness_tensor(j))*C_diff_x[i](j);
            }
            gradient_wrt_x(i) = g;
        }
        VectorXT total_gradient_wrt_thickness(sim.Rods.size());
        total_gradient_wrt_thickness = gradient_wrt_thickness + sim.solveAdjointForOptimization(gradient_wrt_x);
        // std::cout << total_gradient_wrt_thickness.transpose();
        
        int cnt = 0;
        VectorXT rods_radii_current = rods_radii;
        while(true)
        {   
            cnt += 1;
            if (cnt > ls_max)
                break;
            rods_radii = rods_radii_current - step * scale * total_gradient_wrt_thickness;
            rods_radii = rods_radii.cwiseMax(minVal);
            T E1 = 0;
            C_current = findBestCTensorviaProbing(target_location, directions, true);
            C_current_entry = C_entry;
            for(int j = 0; j < 6; ++j){
                E1 += 0.5*(C_entry(j) - stiffness_tensor(j))/std::abs(stiffness_tensor(j))*(C_entry(j) - stiffness_tensor(j));
            }
            if (E1 - E0 < 0) 
                break;
            scale *= T(0.2);
        }

        err = 0;
        for(int j = 0; j < 6; ++j){
            err += std::abs(C_current_entry(j) - stiffness_tensor(j))/std::abs(C_current_entry(j));
        }

        if(err < tol || total_gradient_wrt_thickness.norm() < tol) {
            C_current = findBestCTensorviaProbing(target_location, directions, true);
            std::cout << "Found solution at step: " << iter << " with error: " << err << std::endl; 
            break;
        }
    }
    
    std::cout << rods_radii.transpose() << std::endl; 
    std::cout << "Optimized C: \n" << C_current << std::endl;
    std::cout << "Optimized C array: \n" << C_entry.transpose() << std::endl;
    std::cout << "target C: \n" << stiffness_tensor.transpose() << std::endl;
    std::cout << "Error: " << err << std::endl;
    
}

void Scene::optimizeForThicknessDistribution(const std::vector<TV> target_locations, const std::vector<Vector<T, 6>> stiffness_tensors, const std::string filename, const std::string start_from_file){
    assert(target_locations.size() == stiffness_tensors.size());

    int num_directions = 20;
    std::vector<Eigen::Vector3d> directions;
    for(int i = 0; i < num_directions; ++i) {
        double angle = i*2*M_PI/num_directions; 
        directions.push_back(Eigen::Vector3d{std::cos(angle), std::sin(angle), 0});
    }

    double step = 1e-2;
    double tol = 0.01;
    const int ls_max = 10;
    double minVal = 0.00001;
    const double cut_lower_bound = 0.000001;
    double constraint_weight = 1e15;

    if(start_from_file == ""){
        rods_radii.resize(sim.Rods.size());
        rods_radii.setConstant(6.5e-3);
    } else {
        std::vector<double> rods_radius;
        std::ifstream in_file(start_from_file);
        if (!in_file) {
            std::cerr << "Error opening file for reading: " << start_from_file << std::endl;
        }

        T a;
        while (in_file >> a) {
            rods_radius.push_back(a);
        }
        rods_radii.resize(rods_radius.size());
        for(int i = 0; i < rods_radius.size(); ++i){
            rods_radii(i) = rods_radius[i];
        }
    }
    std::vector<VectorXT> C_current(target_locations.size());
    T err; 
    int num_iterations = 400; int break_iter = num_iterations-1;
    Eigen::VectorXi count(num_iterations);
    std::vector<VectorXT> total_gradients(num_iterations, VectorXT::Zero(sim.Rods.size()));
    for(int iter = 0; iter < num_iterations; ++iter){
        VectorXT gradient_wrt_thickness(sim.Rods.size()); gradient_wrt_thickness.setZero();
        VectorXT gradient_wrt_x(sim.deformed_states.rows()); gradient_wrt_x.setZero();
        T scale = 1;
        T E0 = 0;
        findBestCTensorviaProbing(target_locations, directions, true);
        for(int k = 0; k < target_locations.size(); ++k){
            TV target_location = target_locations[k];
            Vector<T, 6> target_C = stiffness_tensors[k];
            C_current[k] = sample_Cs_info[k].C_entry;
            for(int j = 0; j < 6; ++j){

                // E0 += 0.5*(sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))*(sample_Cs_info[k].C_entry(j) - target_C(j));

                if(j == 2 || j == 4) continue;
                else E0 += 0.5*(sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))*(sample_Cs_info[k].C_entry(j) - target_C(j));

                // if(j == 2 || j == 1 || j == 4) E0 += 0.5*(sample_Cs_info[k].C_entry(j) - target_C(j))*(sample_Cs_info[k].C_entry(j) - target_C(j));
            }
            for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
                T g = 0;
                for(int j = 0; j < 6; ++j){

                    // g += (sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_thickness[i](j);

                    if(j == 2 || j == 4) continue;
                    else g += (sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_thickness[i](j);

                    // if(j == 2 || j == 1 || j == 4) g += (sample_Cs_info[k].C_entry(j) - target_C(j))*sample_Cs_info[k].C_diff_thickness[i](j);
                }
                gradient_wrt_thickness(i) += g;
            }
            for(int i = 0; i < gradient_wrt_x.size(); ++i){
                T g = 0;
                for(int j = 0; j < 6; ++j){

                    // g += (sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_x[i](j);

                    if(j == 2 || j == 4) continue;
                    else g += (sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j))*sample_Cs_info[k].C_diff_x[i](j);

                    // if(j == 2 || j == 1 || j == 4) g += (sample_Cs_info[k].C_entry(j) - target_C(j))*sample_Cs_info[k].C_diff_x[i](j);
                }
                gradient_wrt_x(i) += g;
            }
        }
        VectorXT total_gradient_wrt_thickness(sim.Rods.size());
        total_gradient_wrt_thickness = gradient_wrt_thickness + sim.solveAdjointForOptimization(gradient_wrt_x);
        for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
            E0 += 0.5*constraint_weight*std::max(0.0, minVal-rods_radii(i))*std::max(0.0, minVal-rods_radii(i));
            total_gradient_wrt_thickness(i) += -constraint_weight*std::max(0.0, minVal-rods_radii(i));
            while(std::abs(total_gradient_wrt_thickness(i))*scale > 1) scale /= 10;
        }
       
        int cnt = 0;
        VectorXT rods_radii_current = rods_radii;
        while(true)
        {   
            cnt += 1;
            if (cnt > ls_max)
                break;
            rods_radii = rods_radii_current - step * scale * total_gradient_wrt_thickness;
            rods_radii = rods_radii.cwiseMax(cut_lower_bound);
            T E1 = 0;
            findBestCTensorviaProbing(target_locations, directions, true);
            for(int k = 0; k < target_locations.size(); ++k){
                C_current[k] = sample_Cs_info[k].C_entry;   
                Vector<T, 6> target_C = stiffness_tensors[k];
                for(int j = 0; j < 6; ++j){

                    // E1 += 0.5*(sample_Cs_info[k].C_entry(j) - target_C(j))*(sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j));

                    if(j == 2 || j == 4) continue;
                    else E1 += 0.5*(sample_Cs_info[k].C_entry(j) - target_C(j))*(sample_Cs_info[k].C_entry(j) - target_C(j))/abs(target_C(j));

                    // if(j == 2 || j == 1 || j == 4) E1 += 0.5*(sample_Cs_info[k].C_entry(j) - target_C(j))*(sample_Cs_info[k].C_entry(j) - target_C(j));
                }
            }
            for(int i = 0; i <sim.Rods.size(); ++i)
                E1 += 0.5*constraint_weight*std::max(0.0, minVal-rods_radii(i))*std::max(0.0, minVal-rods_radii(i));    
            if (E1 - E0 < 0.0) 
                break; 
            scale *= T(0.5);
        }
        count(iter) = cnt;
        // if(cnt > ls_max) rods_radii = rods_radii_current;

        err = 0;
        for(int k = 0; k < target_locations.size(); ++k){
            Vector<T, 6> target_C = stiffness_tensors[k];
            for(int j = 0; j < 6; ++j){
                if(j == 2 || j == 4) continue;
                err += std::abs(C_current[k](j) - target_C(j))/std::abs(C_current[k](j));

                // if(j == 2 || j == 1 || j == 4) err += std::abs(C_current[k](j) - target_C(j))/std::abs(C_current[k](j));
            }
        }
        err /= target_locations.size();    
        total_gradients[iter] = total_gradient_wrt_thickness;

        if(total_gradient_wrt_thickness.norm()*scale/step < tol || err < tol) {
            std::cout << "Found solution at step: " << iter << std::endl; 
            break_iter = iter;
            break;
        }
    }

    std::ofstream out(filename+"_C.dat");
    findBestCTensorviaProbing(target_locations, directions, true);
    for(int i = 0; i < target_locations.size(); ++i){
        out << "Optimization location: " << target_locations[i].transpose() << std::endl;
        out << "Optimization target C (DoF): \n" << stiffness_tensors[i].transpose() << std::endl;
        out << "Optimization result C: \n" << sample_Cs_info[i].C_entry.transpose() << "\n";
    }
    out.close();
    std::cout << rods_radii.transpose() << std::endl; 
    std::ofstream out_file(filename+"_radii.dat");
    if (!out_file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    out_file << rods_radii << "\n";
    std::cout << "error: " << err << std::endl;
    std::cout << "counts: " << count.head(break_iter+1).transpose() << std::endl;
    std::cout << "gradient norm: " << total_gradients[break_iter].transpose() << std::endl;
    out_file.close();
}

void Scene::finiteDifferenceEstimation(TV target_location, Vector<T, 6> stiffness_tensor){
    int num_directions = 20;
    std::vector<Eigen::Vector3d> directions;
    for(int i = 0; i < num_directions; ++i) {
        double angle = i*2*M_PI/num_directions; 
        directions.push_back(Eigen::Vector3d{std::cos(angle), std::sin(angle), 0});
    }
    const double rod_size = 8e-3;
    if(rods_radii.rows() == 0){
        rods_radii.resize(sim.Rods.size());
        rods_radii.setConstant(rod_size);
    }
    VectorXT rods_radii_save = rods_radii;
    // const double rod_size = 8e-3;
    // rods_radii.resize(sim.Rods.size());
    // rods_radii.setConstant(rod_size);
    findBestCTensorviaProbing(target_location, directions, true);
    double minVal = 0.0001;
    VectorXT gradient_wrt_thickness(sim.Rods.size());
    T step = 0.2;
    VectorXT delta_h(sim.Rods.size()); delta_h.setConstant(step);
    for(int i = 0; i < gradient_wrt_thickness.size(); ++i){
        T g = 0;
        for(int j = 0; j < 1; ++j){
            g += 2*(C_entry(j) - stiffness_tensor(j))/std::abs(stiffness_tensor(j))/std::abs(stiffness_tensor(j))*C_diff_thickness[i](j);
        }
        gradient_wrt_thickness(i) = g;
    }
    VectorXT gradient_wrt_x(sim.deformed_states.rows());
    for(int i = 0; i < gradient_wrt_x.size(); ++i){
        T g = 0;
        for(int j = 0; j < 1; ++j){
            g += 2*(C_entry(j) - stiffness_tensor(j))/std::abs(stiffness_tensor(j))/std::abs(stiffness_tensor(j))*C_diff_x[i](j);
        }
        gradient_wrt_x(i) = g;
    }
    VectorXT total_gradient_wrt_thickness(sim.Rods.size()); total_gradient_wrt_thickness.setZero();
    total_gradient_wrt_thickness = gradient_wrt_thickness + sim.solveAdjointForOptimization(gradient_wrt_x);
    T obj_init = 0;
    for(int j = 0; j < 1; ++j){
        obj_init += (C_entry(j) - stiffness_tensor(j))/std::abs(stiffness_tensor(j))/std::abs(stiffness_tensor(j))*(C_entry(j) - stiffness_tensor(j));
    }
    std::cout << obj_init << std::endl;

    int test_size = 19; 
    VectorXT errors(test_size); 
    for(int i = 0; i < test_size; ++i){

        T obj_1 = obj_init + (gradient_wrt_thickness).transpose() * (delta_h/std::pow(2, i));
        rods_radii = rods_radii_save;
        rods_radii += delta_h/std::pow(2, i);
        rods_radii = rods_radii.cwiseMax(minVal);
        findBestCTensorviaProbing(target_location, directions, true);
        T obj_2 = 0;
        for(int j = 0; j < 1; ++j){
            obj_2 += (C_entry(j) - stiffness_tensor(j))/std::abs(stiffness_tensor(j))/std::abs(stiffness_tensor(j))*(C_entry(j) - stiffness_tensor(j));
        }
        // std::cout << " T(h) + Nabla T delta h: " << obj_1 << std::endl;
        // std::cout << " T(h+delta h): " << obj_2 << std::endl;
        // std::cout << " Err: " << std::abs(obj_2-obj_1) << std::endl;
        errors(i) = std::abs(obj_2-obj_1);
    }
    for(int i = 1; i < test_size; ++i){
        std::cout <<  delta_h(0)/std::pow(2, i)  << " - " << errors(i-1)/errors(i) << std::endl;
    }
}

Matrix<T, Eigen::Dynamic, 1> Scene::solveForAdjoint(StiffnessMatrix& K, VectorXT rhs){
    VectorXT ddq(sim.W.cols());
    ddq.setZero();

    // StiffnessMatrix K;
    K.setZero();
    sim.buildSystemDoFMatrix(K);
    bool success = sim.linearSolve(K, rhs, ddq);
    if (!success){
        std::cout << "Not succeed in adjoint\n";
        return VectorXT::Zero(sim.Rods.size());
    }

    // T norm = ddq.norm();
    // std::cout << norm << std::endl;
    StiffnessMatrix fd_thickness;
    sim.buildForceGradientWrtThicknessMatrix(fd_thickness);
    VectorXT dxd_thickness = -fd_thickness.transpose() * ddq;
    
    return dxd_thickness;  
}