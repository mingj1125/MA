#include "../autodiff/Quadratic2DShell.h"
#include "../include/QuadraticTriangle.h"
#include <random>

void QuadraticTriangle::computeStrainAndStressPerElement(){

    iterateFaceSerial([&](int face_idx)
    {   
        
        Matrix<T, 6, 3> vertices = getFaceVtxDeformed(face_idx);
        Matrix<T, 6, 3> undeformed_vertices = getFaceVtxUndeformed(face_idx);

        T a = lambda, b = mu;
        T beta_1 = 1/3.; T beta_2 = 1/3.;
        // std::random_device r;
        // // Choose a random mean between 1 and 6
        // std::default_random_engine e1(r());
        // std::uniform_real_distribution<> dis(-0.2, 0.2);
        // beta_1 += dis(e1);
        // beta_2 += dis(e1);
        // beta_1 = std::min(1., beta_1);
        // beta_1 = std::max(0., beta_1);
        // beta_2 = std::min(1., beta_2);
        // beta_2 = std::max(0., beta_2);
        Matrix<T,6,1> N = get_shape_function(beta_1, beta_2);
        TV X = undeformed_vertices.transpose() * N;
        setMaterialParameter(E, nu, a, b, X, face_idx);
        nu_visualization[face_idx] = nu;
        E_visualization[face_idx] = E*thickness;

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        Matrix<T, 2, 2> F = compute2DDeformationGradient(vertices, undeformed_vertices, {beta_1, beta_2});
        TM2 GreenS = 0.5 *(F.transpose()*F - TM2::Identity());
        TM2 S = (a * GreenS.trace() *TM2::Identity() + 2 * b * GreenS)*thickness;
        T areaRatio = ((x1-x0).cross(x2-x0)).norm() / ((X1-X0).cross(X2-X0)).norm();
        cauchy_stress_tensors[face_idx].block(0,0,2,2) = F*S*F.transpose()/areaRatio;
        stress_tensors[face_idx].block(0,0,2,2) = S;
        strain_tensors[face_idx].block(0,0,2,2) = GreenS;
        defomation_gradients[face_idx].block(0,0,2,2) = F;

        // std::cout << face_idx << " " << X(1) << " " << GreenS(1,0) << std::endl;
    });
}

Vector<T, 4> QuadraticTriangle::evaluatePerTriangleStress(const Matrix<T, 6, 3> vertices, const Matrix<T, 6, 3> undeformed_vertices, 
    const TV cut_point_coordinate, const TV direction_normal, const TV sample_loc, int face_idx){
    std::vector<TV> boundary_points;
    TV x1 = undeformed_vertices.row(0); TV x2 = undeformed_vertices.row(1); TV x3 = undeformed_vertices.row(2);
    if(cut_point_coordinate(0) >= 0.-1e-6 && cut_point_coordinate(0) <= 1.+1e-8) {boundary_points.push_back(x1 + cut_point_coordinate(0)*(x2-x1));}
    if(cut_point_coordinate(1) >= 0.-1e-6 && cut_point_coordinate(1) <= 1.+1e-8) {boundary_points.push_back(x2 + cut_point_coordinate(1)*(x3-x2));}
    if(cut_point_coordinate(2) >= 0.-1e-6 && cut_point_coordinate(2) <= 1.+1e-8) {boundary_points.push_back(x3 + cut_point_coordinate(2)*(x1-x3));}

    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel = [variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / (2 * M_PI *std::sqrt(variance_matrix.determinant()));
    };

    TV weighted_traction = TV::Zero();
    T weights = 0;
    int n = std::max((int)((boundary_points.at(1) - boundary_points.at(0)).norm()/(std/6)), 5); // Discretization points for cut segment
    for(int i = 0; i <= n; ++i){
        TV point = (boundary_points.at(1) - boundary_points.at(0))*i/n + boundary_points.at(0);
        if((sample_loc- point).norm() > 4*std) continue;
        TV2 beta = findBarycentricCoord(point, undeformed_vertices);
        // TV2 beta({0.3, 0.3});
        TV X_p = undeformed_vertices.transpose() * get_shape_function(beta(0), beta(1));
        T a = lambda, b = mu;
        setMaterialParameter(E, nu, a, b, X_p, face_idx);
        Matrix<T, 2, 2> F_p = compute2DDeformationGradient(vertices, undeformed_vertices, beta);
        TM2 GreenS = 0.5 *(F_p.transpose()*F_p - TM2::Identity());
        TM2 S = (a * GreenS.trace() *TM2::Identity() + 2 * b * GreenS)*thickness;
        weighted_traction.segment<2>(0) += S * direction_normal.segment<2>(0) * gaussian_kernel(sample_loc, point);
        weights += gaussian_kernel(sample_loc, point);
    }
    Vector<T, 4> res; res.segment<3>(0) = weighted_traction; res(3) = weights;
    res *= (boundary_points.at(1) - boundary_points.at(0)).norm() /(2*n);
    return res;
}

Vector<T, 2> QuadraticTriangle::evaluatePerTriangleStrain(const Matrix<T, 6, 3> vertices, const Matrix<T, 6, 3> undeformed_vertices, 
    const TV cut_point_coordinate, const TV direction, const TV sample_loc, int face_idx){
    std::vector<TV> boundary_points;
    TV x1 = undeformed_vertices.row(0); TV x2 = undeformed_vertices.row(1); TV x3 = undeformed_vertices.row(2);
    if(cut_point_coordinate(0) >= 0.-1e-6 && cut_point_coordinate(0) <= 1.+1e-8) {boundary_points.push_back(x1 + cut_point_coordinate(0)*(x2-x1));}
    if(cut_point_coordinate(1) >= 0.-1e-6 && cut_point_coordinate(1) <= 1.+1e-8) {boundary_points.push_back(x2 + cut_point_coordinate(1)*(x3-x2));}
    if(cut_point_coordinate(2) >= 0.-1e-6 && cut_point_coordinate(2) <= 1.+1e-8) {boundary_points.push_back(x3 + cut_point_coordinate(2)*(x1-x3));}

    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel = [variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / std::sqrt(std::pow((2 * M_PI), 2)*variance_matrix.determinant());
    };

    T weighted_strain = 0;
    T weights = 0;
    int n = std::max((int)((boundary_points.at(1) - boundary_points.at(0)).norm()/(std/6)), 5); // Discretization points for cut segment
    for(int i = 0; i <= n; ++i){
        TV point = (boundary_points.at(1) - boundary_points.at(0))*i/n + boundary_points.at(0);
        if((sample_loc- point).norm() > 4*std) continue;
        TV2 beta = findBarycentricCoord(point, undeformed_vertices);
        // TV2 beta({0.3, 0.3});
        TV X_p = undeformed_vertices.transpose() * get_shape_function(beta(0), beta(1));
        T a = lambda, b = mu;
        setMaterialParameter(E, nu, a, b, X_p, face_idx);
        Matrix<T, 2, 2> F_p = compute2DDeformationGradient(vertices, undeformed_vertices, beta);
        TM2 GreenS = 0.5 *(F_p.transpose()*F_p - TM2::Identity());
        T strain = ((direction.segment<2>(0)).transpose()) * (GreenS * direction.segment<2>(0));
        weighted_strain += strain * gaussian_kernel(sample_loc, point);
        weights += gaussian_kernel(sample_loc, point);
    }
    Vector<T, 2> res; res(0) = weighted_strain; res(1) = weights;
    res *= (boundary_points.at(1) - boundary_points.at(0)).norm() /(2*n);
    return res;
}