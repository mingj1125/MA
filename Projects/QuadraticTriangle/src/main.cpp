#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <igl/readOBJ.h>
#include "../include/QuadraticTriangle.h"
#include "../include/App.h"

Matrix<T, 3, 3> findStiffnessTensor(std::string file_name){
    Eigen::MatrixXd V; Eigen::MatrixXi F;
    igl::readOBJ(file_name, V, F);
    Vector<T,3> min_corner = V.colwise().minCoeff();
    Vector<T,3> max_corner = V.colwise().maxCoeff();

    T bb_diag = max_corner(1) - min_corner(1);

    V *= 1.0 / bb_diag;

    V *= 0.5;

    int c = 3;
    Eigen::MatrixXd n(3, c);
    Eigen::MatrixXd t(3, c);
    int test_face = 436;
    T beta_1 = 1./4;
    T beta_2 = 1./3;
    T shell_len = max_corner(1) - min_corner(1);
    std::cout << shell_len << std::endl;
    T displacement = -0.01*shell_len;

    for(int i = 0; i < c-1; ++i){
        QuadraticTriangle tri;
        tri.set_boundary_condition = false;
        tri.initializeFromFile(file_name);
        Matrix<T, 6, 3> vertices = tri.getFaceVtxUndeformed(test_face);
        Vector<T, 3> sample_location = vertices.transpose()*tri.get_shape_function(beta_1, beta_2); 
    
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

        Matrix<T,2,2> approx_strain = tri.findBestStrainTensorviaProbing(sample_location, tri.direction);
        Matrix<T,3,3> approx_stress = tri.findBestStressTensorviaProbing(sample_location, tri.direction);

        t.col(i) = Eigen::Vector3d({approx_stress(0,0), approx_stress(1,1), approx_stress(1,0)});
        n.col(i) = Eigen::Vector3d({approx_strain(0,0), approx_strain(1,1), 2*approx_strain(1,0)});
        std::cout << "With strain: " << n.col(i).transpose() << "\n has stress: " << t.col(i).transpose() << std::endl; 
    }

    QuadraticTriangle tri;
    tri.set_boundary_condition = false;
    tri.initializeFromFile(file_name);
    Matrix<T, 6, 3> vertices = tri.getFaceVtxUndeformed(test_face);
    Vector<T, 3> sample_location = vertices.transpose()*tri.get_shape_function(beta_1, beta_2); 
    
    for (int j = 0; j < tri.undeformed.size()/3; j++)
    {
        if(tri.undeformed(j*3+2) <= 0 && tri.undeformed(j*3) >= V.colwise().maxCoeff()(0)-1e-5){
            for (int d = 0; d < 3; d++)
            {   
                tri.dirichlet_data[j * 3 + d] = 0.;
            }
        } else if(tri.undeformed(j*3+2) <= 0 && tri.undeformed(j*3) <= V.colwise().minCoeff()(0)+1e-5){
            for (int d = 0; d < 3; d++)
            {   
                tri.u[j * 3 + 1] = displacement;
                tri.dirichlet_data[j * 3 + d] = 0.;
            }
        }
    }
    int static_solve_step = 0;
    bool finished = tri.advanceOneStep(static_solve_step++);
    while(!finished) finished = tri.advanceOneStep(static_solve_step++);
    Matrix<T,2,2> approx_strain = tri.findBestStrainTensorviaProbing(sample_location, tri.direction);
    Matrix<T,3,3> approx_stress = tri.findBestStressTensorviaProbing(sample_location, tri.direction);

    t.col(c-1) = Eigen::Vector3d({approx_stress(0,0), approx_stress(1,1), approx_stress(1,0)});
    n.col(c-1) = Eigen::Vector3d({approx_strain(0,0), approx_strain(1,1), 2*approx_strain(1,0)});
    std::cout << "With strain: " << n.col(c-1).transpose() << "\n has stress: " << t.col(c-1).transpose() << std::endl; 

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3*c,6);
    Eigen::VectorXd b(3*c);
    for(int i = 0; i < c; ++i){
        Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(3,6);
        Eigen::Vector3d normal = n.col(i);
        A_block << normal(0), normal(1), 0, normal(2), 0, 0,
                    0, normal(0), normal(1), 0, normal(2), 0,
                    0, 0, 0, normal(0), normal(1), normal(2);
        A.block(i*3, 0, 3, 6) = A_block;
        b.segment(i*3, 3) = t.col(i);
    }
    Eigen::VectorXd x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    Matrix<T,3,3> fitted_symmetric_tensor;
    fitted_symmetric_tensor << x(0), x(1), x(3), 
                            x(1), x(2), x(4),
                            x(3), x(4), x(5);

    return fitted_symmetric_tensor;
}

int main()
{
    QuadraticTriangle quad_tri;

    // quad_tri.initializeFromFile("../../../Projects/QuadraticTriangle/data/grid.obj");
    quad_tri.initializeFromFile("../../../Projects/QuadraticTriangle/data/sun_mesh_line.obj");
    // quad_tri.initializeFromFile("../../../Projects/QuadraticTriangle/data/grid_double_refined.obj");
    // quad_tri.initializeFromFile("../../../Projects/QuadraticTriangle/data/grid_refined_mesh.obj");

    // quad_tri.testIsotropicStretch();
    // quad_tri.testVerticalDirectionStretch();
    // quad_tri.testHorizontalDirectionStretch();

    // int static_solve_step = 0;
    // bool finished = quad_tri.advanceOneStep(static_solve_step++);
    // while(!finished) finished = quad_tri.advanceOneStep(static_solve_step++);

    
    App<QuadraticTriangle> app(quad_tri);

    app.initializeScene();
    app.run();

    std::cout << findStiffnessTensor("../../../Projects/QuadraticTriangle/data/sun_mesh_line.obj") << std::endl;


    return 0;
}
