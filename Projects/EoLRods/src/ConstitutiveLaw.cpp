#include "../include/EoLRodSim.h"
#include "../include/Scene.h"

Matrix<T, 3, 3> EoLRodSim::computeGreenLagrangianStrain(const TV sample_loc, const std::vector<TV> line_directions){
    TM F = computeWeightedDeformationGradient(sample_loc, line_directions);
    return 0.5*(F.transpose()*F-TM::Identity());
}

Matrix<T, 3, 3> Scene::findBestCTensorviaProbing(TV sample_loc, const std::vector<TV> line_directions){

    std::string mesh_file = "../../../Projects/EoLRods/data/irregular_mesh_good.obj";
    int num_test = 4;
    int c = 2*num_test;
    Eigen::MatrixXd n(3, c);
    Eigen::MatrixXd t(3, c);
    for(int i = 1; i <= num_test; ++i){
        EoLRodSim sim1;
        sim = sim1;
        buildFEMRodScene(mesh_file, 0, true);
        // buildGridScene(0, true);
        TV bottom_left, top_right;
        sim.computeUndeformedBoundingBox(bottom_left, top_right);
        if(sample_loc.norm() <= 0) {
            sample_loc = (bottom_left + top_right)/2;
            sample_loc[0] += ((top_right-bottom_left)*0.25)[0];
        }

        T rec_width = 0.0001 * sim.unit;
        TV shear_x_right = TV(0.02*i, 0.0, 0.0) * sim.unit;
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

        EoLRodSim sim2;
        sim = sim2;
        buildFEMRodScene(mesh_file, 0, true);
        // buildGridScene(0, true);
        TV shear_x_down = TV(0.0, 0.0, 0.0) * sim.unit;
        TV shear_x_up = TV(0.0, 0.02*i, 0) * sim.unit;
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
    }

    // for(int i = 0; i < c; ++i){
    //     std::cout << "Stress in strain: " << n.col(i).transpose() << " is : \n" << t.col(i).transpose() << std::endl;
    // }

    bool fit_symmetric_constrained = true;
    Matrix<T, 3, 3> fitted_tensor; fitted_tensor.setZero();
    if(!fit_symmetric_constrained){
        Eigen::MatrixXd A = n.transpose();
        Eigen::MatrixXd b = t.transpose();
        Eigen::MatrixXd x = (A.transpose()*A).ldlt().solve(A.transpose()*b);

        fitted_tensor = x.transpose();
    } else {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3*c,6);
        VectorXT b(3*c);
        for(int i = 0; i < c; ++i){
            Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(3,6);
            TV normal = n.col(i);
            A_block << normal(0), normal(1), normal(2), 0, 0, 0,
                    0, normal(0), 0, normal(1), normal(2), 0,
                    0, 0, normal(0), 0, normal(1), normal(2);
            A.block(i*3, 0, 3, 6) = A_block;
            b.segment(i*3, 3) = t.col(i);
        }
        VectorXT x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
        fitted_tensor << x(0), x(1), x(2), 
                        x(1), x(3), x(4),
                        x(2), x(4), x(5);
    }

    return fitted_tensor;

}