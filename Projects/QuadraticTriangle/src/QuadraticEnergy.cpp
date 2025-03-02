#include <Eigen/CholmodSupport>
#include <igl/readOBJ.h>
#include <igl/massmatrix.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include "../autodiff/Quadratic2DShell.h"
#include "../include/QuadraticTriangle.h"

// Define the quadrature points and weights for a triangle
struct QuadraturePoint {
    double xi, eta, weight;
};

std::vector<QuadraturePoint> get_6_point_gaussian_quadrature() {
    return {
        {0.0915762135, 0.0915762135, 0.1099517437},
        {0.8168475730, 0.0915762135, 0.1099517437},
        {0.0915762135, 0.8168475730, 0.1099517437},
        {0.4459484909, 0.1081030182, 0.2233815897},
        {0.1081030182, 0.4459484909, 0.2233815897},
        {0.4459484909, 0.4459484909, 0.2233815897}
    };
    // return {
    //     {0., 0., 1/6.},
    //     {1., 0., 1/6.},
    //     {0., 1., 1/6.},
    //     {0.5, 0., 1/6.},
    //     {0., 0.5, 1/6.},
    //     {0.5, 0.5, 1/6.}
    // };
}

Vector<T, 18> QuadraticTriangle::compute2DQuadraticShellEnergyGradient(const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices){
        
    Vector<T, 18> gradient; gradient.setZero();
    T area = ((undeformed_vertices.row(1)-undeformed_vertices.row(0)).cross(undeformed_vertices.row(2)-undeformed_vertices.row(0))).norm();
    std::vector<QuadraturePoint> quadrature_points = get_6_point_gaussian_quadrature();
    
    for(const auto& point: quadrature_points){
        T local_lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        T local_mu = E / 2.0 / (1.0 + nu);
        TV X = undeformed_vertices.row(0) * point.xi + undeformed_vertices.row(1) * point.eta + undeformed_vertices.row(2) * (1-point.eta-point.xi);
        if(heterogenuous) setMaterialParameter(E, nu, local_lambda, local_mu, X);
        gradient += computePointEnergyDensityGradient(local_lambda, local_mu, vertices, undeformed_vertices, {point.xi, point.eta}) * point.weight;
    }

    return gradient*area*thickness;
}

T QuadraticTriangle::compute2DQuadraticShellEnergy(const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices){
        
    T area = ((undeformed_vertices.row(1)-undeformed_vertices.row(0)).cross(undeformed_vertices.row(2)-undeformed_vertices.row(0))).norm();
    std::vector<QuadraturePoint> quadrature_points = get_6_point_gaussian_quadrature();
    double energy = 0.0;
    for(const auto& point: quadrature_points){
        T local_lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        T local_mu = E / 2.0 / (1.0 + nu);
        TV X = undeformed_vertices.row(0) * point.xi + undeformed_vertices.row(1) * point.eta + undeformed_vertices.row(2) * (1-point.eta-point.xi);
        if(heterogenuous) setMaterialParameter(E, nu, local_lambda, local_mu, X);
        energy += computePointEnergyDensity(local_lambda, local_mu, vertices, undeformed_vertices, {point.xi, point.eta}) * point.weight;
    }

    energy *= area*thickness;
    return energy;
}

Matrix<T, 18, 18> QuadraticTriangle::compute2DQuadraticShellEnergyHessian(const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices){
        
    Matrix<T, 18, 18> hessian; hessian.setZero();
    T area = ((undeformed_vertices.row(1)-undeformed_vertices.row(0)).cross(undeformed_vertices.row(2)-undeformed_vertices.row(0))).norm();
    std::vector<QuadraturePoint> quadrature_points = get_6_point_gaussian_quadrature();
    
    for(const auto& point: quadrature_points){
        T local_lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        T local_mu = E / 2.0 / (1.0 + nu);
        TV X = undeformed_vertices.row(0) * point.xi + undeformed_vertices.row(1) * point.eta + undeformed_vertices.row(2) * (1-point.eta-point.xi);
        if(heterogenuous) setMaterialParameter(E, nu, local_lambda, local_mu, X);
        hessian += computePointEnergyDensityHessian(local_lambda, local_mu, vertices, undeformed_vertices, {point.xi, point.eta}) * point.weight;
    }

    return hessian*area*thickness;
}

Matrix<T, 2, 2> QuadraticTriangle::compute2DDeformationGradient(const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices, const Vector<T, 2> beta){
    Matrix<T,2,6> x = vertices.block(0,0,6,2).transpose();
    return x*(compute2DdNdX(undeformed_vertices, beta).transpose());
}