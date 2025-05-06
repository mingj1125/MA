#ifndef ENFORCE_MATRIX_CONSTRAINTS_H
#define ENFORCE_MATRIX_CONSTRAINTS_H

#include "vector_manipulation.h"
#include <iostream>

enum RectangularMatrixConstraintType
{
	CONSTRAIN_ROWS,
	CONSTRAIN_COLUMNS
};

std::vector<Eigen::Triplet<AScalar>> SparseMatrixToTriplets(const Eigen::SparseMatrix<AScalar>& A);

Eigen::SparseMatrix<AScalar> ComputeAffineConstrainedHessian(const Eigen::SparseMatrix<AScalar>& H, const VectorXa& x, int cut_off = 0);

Eigen::SparseMatrix<AScalar> ComputeFakeAffineConstrainedHessian(const Eigen::SparseMatrix<AScalar>& H, const VectorXa& x);

Eigen::SparseMatrix<AScalar> EnforceSquareMatrixConstraints(Eigen::SparseMatrix<AScalar>& old, std::vector<int>& constraints, bool fill_ones=true);

Eigen::SparseMatrix<AScalar> EnforceRectangularMatrixConstraints(Eigen::SparseMatrix<AScalar>& old, std::vector<int>& constraints, RectangularMatrixConstraintType type);

template<typename T, int S>
Eigen::Matrix<T, S, S> ProjectEigenvalues(Eigen::Matrix<T,S,S>& m, AScalar th, AScalar pr)
{
	Eigen::EigenSolver<Eigen::Matrix<T, S, S> > es(m);
	Eigen::Matrix<T, S, S>  D = es.pseudoEigenvalueMatrix();
	Eigen::Matrix<T, S, S>  V = es.pseudoEigenvectors();

	for(int i=0; i<S; ++i)
	{
		if(D(i,i)<th)
			D(i,i) = pr;
	}

	// std::cout << "/////////////" << std::endl;
	// std::cout << singular << std::endl;
	// std::cout << m << std::endl << std::endl;
	// std::cout << svd.matrixU()*sI*svd.matrixV().transpose() << std::endl;
	// std::cout << "/////////////" << std::endl;

	return V*D*V.inverse();
}

template <typename Scalar, int size>
void ProjectSPD_IGL(Eigen::Matrix<Scalar, size, size>& symMtr, AScalar eps = 0.)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if (eigenSolver.eigenvalues()[0] >= 0.0) {
        return;
    }
    Eigen::DiagonalMatrix<Scalar, size> D(eigenSolver.eigenvalues());
    int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
    for (int i = 0; i < rows; i++) {
        if (D.diagonal()[i] <= eps) {
            D.diagonal()[i] = eps;
        }
        else {
            break;
        }
        //D.diagonal()[i] = abs(D.diagonal()[i]);
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

#endif
