#ifndef STIFFNESSTENSOR_H
#define STIFFNESSTENSOR_H

#include <igl/readOBJ.h>
#include "../include/PBCevaluation.h"

template<class Simulation>
std::vector<Matrix<T, 3, 3>> findStiffnessTensor(std::string file_name, Simulation sim){
    Matrix<T, -1, -1> V; Matrix<int, -1, -1> F;
    igl::readOBJ(file_name, V, F);
    int c = 3;
    std::vector<Eigen::MatrixXd> n(F.rows(), Eigen::MatrixXd(3, c));
    std::vector<Eigen::MatrixXd> t(F.rows(), Eigen::MatrixXd(3, c));

    for(int i = 0; i < c; ++i){
        Simulation sim_t;
        sim_t.std = sim.std;
        sim_t.initializeFromDir(sim.mesh_info, "exp"+std::to_string(i+1)+"/");
        for(int f = 0; f < F.rows(); ++f){
            Matrix<T, 3, 3> undeformed_vertices = sim_t.getVisualFaceVtxUndeformed(f);
            Vector<T,3> sample_location = sim_t.triangleCenterofMass(undeformed_vertices);

            Matrix<T,2,2> approx_strain = sim_t.findBestStrainTensorviaProbing(sample_location, sim_t.direction);
            Matrix<T,3,3> approx_stress = sim_t.findBestStressTensorviaProbing(sample_location, sim_t.direction);

            t[f].col(i) = Eigen::Vector3d({approx_stress(0,0), approx_stress(1,1), approx_stress(1,0)});
            n[f].col(i) = Eigen::Vector3d({approx_strain(0,0), approx_strain(1,1), 2*approx_strain(1,0)});
        }
    }

    std::vector<Matrix<T,3,3>> result(F.rows());
    for(int f = 0; f < F.rows(); ++f){
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3*c,6);
        Eigen::VectorXd b(3*c);
        for(int i = 0; i < c; ++i){
            Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(3,6);
            Eigen::Vector3d normal = n[f].col(i);
            A_block << normal(0), normal(1), 0, normal(2), 0, 0,
                        0, normal(0), normal(1), 0, normal(2), 0,
                        0, 0, 0, normal(0), normal(1), normal(2);
            A.block(i*3, 0, 3, 6) = A_block;
            b.segment(i*3, 3) = t[f].col(i);
        }
        Eigen::VectorXd x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
        Matrix<T,3,3> fitted_symmetric_tensor;
        fitted_symmetric_tensor << x(0), x(1), x(3), 
                                x(1), x(2), x(4),
                                x(3), x(4), x(5);
        result.at(f) = fitted_symmetric_tensor;
    }

    Eigen::MatrixXd n_homo(3, c);
    Eigen::MatrixXd t_homo(3, c);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3*c,6);
    Eigen::VectorXd b(3*c);

    for(int i = 0; i < c; ++i){
        std::string mesh_info = sim.mesh_info;
        Matrix<T,2,2> approx_stress = sim.read_matrices(mesh_info+"exp"+std::to_string(i+1)+"/"+"pbc_homo_stress.dat")[0];
        Matrix<T,2,2> approx_strain = sim.read_matrices(mesh_info+"exp"+std::to_string(i+1)+"/"+"pbc_homo_strain.dat")[0];

        t_homo.col(i) = Eigen::Vector3d({approx_stress(0,0), approx_stress(1,1), approx_stress(1,0)});
        n_homo.col(i) = Eigen::Vector3d({approx_strain(0,0), approx_strain(1,1), 2*approx_strain(1,0)});
    }
    for(int i = 0; i < c; ++i){
        Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(3,6);
        Eigen::Vector3d normal = n_homo.col(i);
        A_block << normal(0), normal(1), 0, normal(2), 0, 0,
                    0, normal(0), normal(1), 0, normal(2), 0,
                    0, 0, 0, normal(0), normal(1), normal(2);
        A.block(i*3, 0, 3, 6) = A_block;
        b.segment(i*3, 3) = t_homo.col(i);
    }
    Eigen::VectorXd x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    Matrix<T,3,3> fitted_symmetric_tensor;
    fitted_symmetric_tensor << x(0), x(1), x(3), 
                            x(1), x(2), x(4),
                            x(3), x(4), x(5);
    std::cout << "Stiffness C from homogenization: \n" << fitted_symmetric_tensor << std::endl;    
    
    for(int i = 0; i < c; ++i){
        std::string mesh_info = sim.mesh_info;
        Matrix<T,2,2> approx_stress = sim.read_matrices(mesh_info+"exp"+std::to_string(i+1)+"/"+"window_homo_stress.dat")[0];
        Matrix<T,2,2> approx_strain = sim.read_matrices(mesh_info+"exp"+std::to_string(i+1)+"/"+"window_homo_strain.dat")[0];

        t_homo.col(i) = Eigen::Vector3d({approx_stress(0,0), approx_stress(1,1), approx_stress(1,0)});
        n_homo.col(i) = Eigen::Vector3d({approx_strain(0,0), approx_strain(1,1), 2*approx_strain(1,0)});
    }
    for(int i = 0; i < c; ++i){
        Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(3,6);
        Eigen::Vector3d normal = n_homo.col(i);
        A_block << normal(0), normal(1), 0, normal(2), 0, 0,
                    0, normal(0), normal(1), 0, normal(2), 0,
                    0, 0, 0, normal(0), normal(1), normal(2);
        A.block(i*3, 0, 3, 6) = A_block;
        b.segment(i*3, 3) = t_homo.col(i);
    }
    x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    fitted_symmetric_tensor << x(0), x(1), x(3), 
                            x(1), x(2), x(4),
                            x(3), x(4), x(5);
    std::cout << "Stiffness C from window approach: \n" << fitted_symmetric_tensor << std::endl;    

    return result;
}

#endif