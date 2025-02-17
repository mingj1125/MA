#include <Eigen/CholmodSupport>
#include <igl/readOBJ.h>
#include <igl/massmatrix.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include "../autodiff/Quadratic2DShell.h"
#include "../include/QuadraticTriangle.h"

#include <cmath> // for pi in cut directions
#include <random> // for rng in sampling for stress probing
#include <set>

void QuadraticTriangle::computeStrainAndStressPerElement(){

    iterateFaceSerial([&](int face_idx)
    {   
        
        Matrix<T, 6, 3> vertices = getFaceVtxDeformed(face_idx);
        Matrix<T, 6, 3> undeformed_vertices = getFaceVtxUndeformed(face_idx);
        T a, b;
        if(heterogenuous) setMaterialParameter(E, nu, a, b, triangleCenterofMass(undeformed_vertices.block(0,0,3,3)));

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        Matrix<T, 2, 2> F = compute2DDeformationGradient(vertices, undeformed_vertices, {1/3., 1/3.});
        // if(face_idx == 60) std::cout << "F: \n" << F << std::endl;
        // F = compute2DDeformationGradient(vertices, undeformed_vertices, {1/3., 0.});
        // if(face_idx == 60) std::cout << "F middle: \n" << F << std::endl;
        TM2 GreenS = 0.5 *(F.transpose()*F - TM2::Identity());
        space_strain_tensors[face_idx] = GreenS;
        TM2 S = (a * GreenS.trace() *TM2::Identity() + 2 * b * GreenS)*thickness;
        T areaRatio = ((x1-x0).cross(x2-x0)).norm() / ((X1-X0).cross(X2-X0)).norm();
        cauchy_stress_tensors[face_idx].block(0,0,2,2) = F*S*F.transpose()/areaRatio;
        stress_tensors[face_idx].block(0,0,2,2) = S;
        strain_tensors[face_idx].block(0,0,2,2) = GreenS;
        defomation_gradients[face_idx].block(0,0,2,2) = F;
    });
}

Matrix<T, 2, 2> QuadraticTriangle::computeCauchyStrainwrt2dXSpace(Matrix<T, 3, 2> F){
    return 0.5 * (F.transpose()*F - TM2::Identity());
}

Matrix<T, 3, 2> QuadraticTriangle::computeDeformationGradientwrt2DXSpace(
    const TV X1, const TV X2, const TV X3, 
    const TV x1, const TV x2, const TV x3)
{
    TV localSpannT1 = (X2-X1).normalized();
    TV localSpannT2 = ((X3-X1) - localSpannT1*((X3-X1).dot(localSpannT1))).normalized();

    FaceVtx x; x << x1, x2, x3;

    Matrix<T, 3, 2> dNdB; dNdB << -1, -1, 1, 0, 0, 1;

    Matrix<T, 2, 3> dBdX = computeBarycentricJacobian(X1, X2, X3);
    Matrix<T, 3, 2> dXdX_2D;  dXdX_2D << localSpannT1, localSpannT2;

    Matrix<T, 3, 2> F = x * dNdB * dBdX * dXdX_2D;

    return F;
}

// pseudoinverse of [X2-X1, X3-X1]
Matrix<T, 2, 3> QuadraticTriangle::computeBarycentricJacobian(
    const TV X1, const TV X2, const TV X3)
{
    TV v1 = (X2-X1);
    TV v2 = (X3-X1);

    T denominator = v1.squaredNorm()*v2.squaredNorm() - v1.dot(v2) * v1.dot(v2);

    Matrix<T, 3, 2> transpose_dBdX;
    transpose_dBdX << v1*v2.squaredNorm() - v2*v1.dot(v2),
                       v2*v1.squaredNorm() - v1*v1.dot(v2);

    transpose_dBdX /= denominator;                   

    return transpose_dBdX.transpose();
}
