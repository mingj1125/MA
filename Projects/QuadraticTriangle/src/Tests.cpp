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

void QuadraticTriangle::testIsotropicStretch(){
    deformed = 0.7 * undeformed;
    computeStrainAndStressPerElement();
    int A = 40;
    Matrix<T, 6, 3> vertices = getFaceVtxUndeformed(A);
    sample[1] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
    std::cout << "Found stress tensor via Probing: \n" << findBestStressTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found stress tensor via Averaging: \n" << findBestStressTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated stress tensor at sample point triangle: \n" << cauchy_stress_tensors[A] << std::endl;
    std::cout << "Found strain tensor via Probing: \n" << findBestStrainTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found strain tensor via Averaging: \n" << findBestStrainTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated strain tensor at sample point triangle: \n" << strain_tensors[A] << std::endl;
}

void QuadraticTriangle::testHorizontalDirectionStretch(){

    for(int i = 0; i < deformed.size(); i +=3) deformed[i] = 0.7 * undeformed[i];
    computeStrainAndStressPerElement();
    int A = 40;
    Matrix<T, 6, 3> vertices = getFaceVtxUndeformed(A);
    sample[1] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
    std::cout << "Found stress tensor via Probing: \n" << findBestStressTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found stress tensor via Averaging: \n" << findBestStressTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated stress tensor at sample point triangle: \n" << cauchy_stress_tensors[A] << std::endl;
    std::cout << "Found strain tensor via Probing: \n" << findBestStrainTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found strain tensor via Averaging: \n" << findBestStrainTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated strain tensor at sample point triangle: \n" << strain_tensors[A] << std::endl;

}

void QuadraticTriangle::testVerticalDirectionStretch(){

    int A = 0;
    int B = deformed.size()/3/2;
    for(int i = 1; i < deformed.size(); i +=3) deformed[i] = 0.7 * undeformed[i];
    computeStrainAndStressPerElement();
    Matrix<T, 6, 3> vertices = getFaceVtxUndeformed(A);
    sample[1] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
    sample[0] = triangleCenterofMass(getFaceVtxUndeformed(B).block(0,0,3,3));
    std::cout << "Found strain tensor via Probing: \n" << findBestStrainTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found strain tensor via Averaging: \n" << findBestStrainTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated strain tensor at sample point triangle: \n" << strain_tensors[A] << std::endl;
    std::cout << "Found stress tensor via Probing: \n" << findBestStressTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found stress tensor via Averaging: \n" << findBestStressTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated stress tensor at sample point triangle: \n" << cauchy_stress_tensors[A] << std::endl;

}

void QuadraticTriangle::testSharedEdgeStress(int A, int B, int v1, int v2) {
    
    std::cout << "Quick test for edge stresses...\n";
    TV edge = undeformed.segment<3>(v1*3) - undeformed.segment<3>(v2*3);
    TV normal; normal << -edge(1), edge(0), 0; normal = normal.normalized();
    std::cout << "normal direction: " << normal.transpose() << std::endl;
    TV stress_1 = findBestStressTensorviaProbing(triangleCenterofMass(getFaceVtxUndeformed(A).block(0,0,3,3)), direction) *normal;
    TV stress_2 = findBestStressTensorviaProbing(triangleCenterofMass(getFaceVtxUndeformed(B).block(0,0,3,3)), direction) *normal;
    std::cout << "Kernel stress from triangle " << A << " : " << stress_1.transpose() << "\nKernel stress from triangle " << B << " : " << stress_2.transpose() << std::endl; 
    stress_1 = stress_tensors[A]*normal;
    stress_2 = stress_tensors[B]*normal;
    std::cout << "Local stress from triangle " << A << " : " << stress_1.transpose() << "\nLocal stress from triangle " << B << " : " << stress_2.transpose() << std::endl; 
    stress_1 = findBestStressTensorviaAveraging(triangleCenterofMass(getFaceVtxUndeformed(A).block(0,0,3,3))) *normal;
    stress_2 = findBestStressTensorviaAveraging(triangleCenterofMass(getFaceVtxUndeformed(B).block(0,0,3,3))) *normal;
    std::cout << "Average stress from triangle " << A << " : " << stress_1.transpose() << "\nAverage stress from triangle " << B << " : " << stress_2.transpose() << std::endl; 
}

void QuadraticTriangle::testStressTensors(int A, int B){

    sample[0] = triangleCenterofMass(getFaceVtxUndeformed(A).block(0,0,3,3));
    sample[1] = triangleCenterofMass(getFaceVtxUndeformed(B).block(0,0,3,3));
    std::cout << "Tested triangle: " << A << std::endl;
    std::cout << "Found stress tensor via Probing: \n" << findBestStressTensorviaProbing(sample[0], direction) << std::endl; 
    std::cout << "Found stress tensor via Averaging: \n" << findBestStressTensorviaAveraging(sample[0]) << std::endl;
    std::cout << "Found strain tensor via Probing: \n" << findBestStrainTensorviaProbing(sample[0], direction) << std::endl;
    std::cout << "Found strain tensor via Averaging: \n" << findBestStrainTensorviaAveraging(sample[0]) << std::endl;
    std::cout << "Caculated strain tensor at sample point triangle: \n" << strain_tensors[A] << std::endl;

}
