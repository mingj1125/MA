#ifndef STIFFNESSTENSOR_H
#define STIFFNESSTENSOR_H

#include <igl/readOBJ.h>
#include "../include/QuadraticTriangle.h"

template<class Simulation>
std::vector<Matrix<T, 3, 3>> findStiffnessTensor(std::string file_name, Simulation sim){
    Eigen::MatrixXd V; Eigen::MatrixXi F;
    igl::readOBJ(file_name, V, F);
    Vector<T,3> min_corner = V.colwise().minCoeff();
    Vector<T,3> max_corner = V.colwise().maxCoeff();

    T bb_diag = max_corner(1) - min_corner(1);

    V *= 1.0 / bb_diag;

    int c = 3;
    std::vector<Eigen::MatrixXd> n(F.rows(), Eigen::MatrixXd(3, c));
    std::vector<Eigen::MatrixXd> t(F.rows(), Eigen::MatrixXd(3, c));
    int test_face = 0;
    T beta_1 = 1./3;
    T beta_2 = 1./3;
    T shell_len = max_corner(1) - min_corner(1);
    T displacement = -0.01;

    for(int i = 0; i < 2; ++i){
        QuadraticTriangle tri(sim.nu_default, sim.graded_k, sim.std, sim.tags, sim.graded);
        tri.set_boundary_condition = false;
        tri.tag_file = sim.tag_file;
        tri.tags = true;
        tri.initializeFromFile(file_name);
    
        for (int j = 0; j < tri.undeformed.size()/3; j++)
        {
            if(tri.undeformed(j*3+i) > V.colwise().maxCoeff()(i)-1e-5){
                for (int d = 0; d < 3; d++)
                {   
                    tri.dirichlet_data[j * 3 + d] = 0.;
                }
            } else if(tri.undeformed(j*3+i) < V.colwise().minCoeff()(i)+1e-5){
                for (int d = 0; d < 3; d++)
                {   
                    tri.u[j * 3 + i] = displacement;
                    tri.dirichlet_data[j * 3 + d] = 0.;
                }
            }
        }
        int static_solve_step = 0;
        bool finished = tri.advanceOneStep(static_solve_step++);
        while(!finished) finished = tri.advanceOneStep(static_solve_step++);

        for(int f = 0; f < F.rows(); ++f){
            Matrix<T, 6, 3> vertices = tri.getFaceVtxUndeformed(f);
            Vector<T, 3> sample_location = vertices.transpose()*tri.get_shape_function(beta_1, beta_2); 

            Matrix<T,2,2> approx_strain = tri.findBestStrainTensorviaProbing(sample_location, tri.direction);
            Matrix<T,3,3> approx_stress = tri.findBestStressTensorviaProbing(sample_location, tri.direction);

            t[f].col(i) = Eigen::Vector3d({approx_stress(0,0), approx_stress(1,1), approx_stress(1,0)});
            n[f].col(i) = Eigen::Vector3d({approx_strain(0,0), approx_strain(1,1), 2*approx_strain(1,0)});
        }
        std::cout << "1036: With strain: " << n[1036].col(i).transpose() << "\n has stress: " << t[1036].col(i).transpose() << std::endl; 
        std::cout << "701: With strain: " << n[1036].col(i).transpose() << "\n has stress: " << t[1036].col(i).transpose() << std::endl; 
    }

    QuadraticTriangle tri(sim.nu_default, sim.graded_k, sim.std, sim.tags, sim.graded);
    tri.set_boundary_condition = false;
    tri.tag_file = sim.tag_file;
    tri.tags = true;
    tri.initializeFromFile(file_name);
    
    for (int j = 0; j < tri.undeformed.size()/3; j++)
    {
        if(tri.undeformed(j*3+2) <= 0 && tri.undeformed(j*3+1) >= V.colwise().maxCoeff()(1)-1e-5){
            for (int d = 0; d < 3; d++)
            {   
                tri.dirichlet_data[j * 3 + d] = 0.;
            }
        } else if(tri.undeformed(j*3+2) <= 0 && tri.undeformed(j*3+1) <= V.colwise().minCoeff()(1)+1e-5){
            for (int d = 0; d < 3; d++)
            {   
                tri.u[j * 3] = displacement;
                tri.dirichlet_data[j * 3 + d] = 0.;
            }
        }
    }
    int static_solve_step = 0;
    bool finished = tri.advanceOneStep(static_solve_step++);
    while(!finished) finished = tri.advanceOneStep(static_solve_step++);

    for(int f = 0; f < F.rows(); ++f){
        Matrix<T, 6, 3> vertices = tri.getFaceVtxUndeformed(f);
        Vector<T, 3> sample_location = vertices.transpose()*tri.get_shape_function(beta_1, beta_2); 

        Matrix<T,2,2> approx_strain = tri.findBestStrainTensorviaProbing(sample_location, tri.direction);
        Matrix<T,3,3> approx_stress = tri.findBestStressTensorviaProbing(sample_location, tri.direction);

        t[f].col(c-1) = Eigen::Vector3d({approx_stress(0,0), approx_stress(1,1), approx_stress(1,0)});
        n[f].col(c-1) = Eigen::Vector3d({approx_strain(0,0), approx_strain(1,1), 2*approx_strain(1,0)});
    }
    std::cout << "1036 With strain: " << n[1036].col(c-1).transpose() << "\n has stress: " << t[1036].col(c-1).transpose() << std::endl; 
    std::cout << "701: With strain: " << n[701].col(c-1).transpose() << "\n has stress: " << t[701].col(c-1).transpose() << std::endl; 
    
    // QuadraticTriangle tri2(sim.nu_default, sim.graded_k, sim.std);
    // tri2.set_boundary_condition = false;
    // tri2.initializeFromFile(file_name);
    
    // for (int j = 0; j < tri2.undeformed.size()/3; j++)
    // {
    //     if(tri2.undeformed(j*3+2) <= 0 && tri2.undeformed(j*3) >= V.colwise().maxCoeff()(0)-1e-5){
    //         for (int d = 0; d < 3; d++)
    //         {   
    //             tri2.dirichlet_data[j * 3 + d] = 0.;
    //         }
    //     } else if(tri2.undeformed(j*3+2) <= 0 && tri2.undeformed(j*3) <= V.colwise().minCoeff()(0)+1e-5){
    //         for (int d = 0; d < 3; d++)
    //         {   
    //             tri2.u[j * 3 + 1] = -displacement;
    //             tri2.dirichlet_data[j * 3 + d] = 0.;
    //         }
    //     }
    // }
    // static_solve_step = 0;
    // finished = tri2.advanceOneStep(static_solve_step++);
    // while(!finished) finished = tri2.advanceOneStep(static_solve_step++);

    // for(int f = 0; f < F.rows(); ++f){
    //     Matrix<T, 6, 3> vertices = tri2.getFaceVtxUndeformed(f);
    //     Vector<T, 3> sample_location = vertices.transpose()*tri2.get_shape_function(beta_1, beta_2); 

    //     Matrix<T,2,2> approx_strain = tri2.findBestStrainTensorviaProbing(sample_location, tri.direction);
    //     Matrix<T,3,3> approx_stress = tri2.findBestStressTensorviaProbing(sample_location, tri.direction);

    //     t[f].col(c-2) = Eigen::Vector3d({approx_stress(0,0), approx_stress(1,1), approx_stress(1,0)});
    //     n[f].col(c-2) = Eigen::Vector3d({approx_strain(0,0), approx_strain(1,1), 2*approx_strain(1,0)});
    // }

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

    return result;
}

#endif