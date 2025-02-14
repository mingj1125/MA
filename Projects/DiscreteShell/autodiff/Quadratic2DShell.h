#ifndef QUADRATIC_2D_SHELL_H
#define QUADRATIC_2D_SHELL_H

#include "../include/VecMatDef.h"

Matrix<T, 2, 6> compute2DdNdX(const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, const Vector<T, 2> beta);
Matrix<T, 2, 6> compute2DdNdx(const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, const Vector<T, 2> beta);

Matrix<T, 2, 2> compute2DDeformationGradient(const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, const Vector<T, 2> beta);

T computePointEnergyDensity(T lambda, T mu, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, const Vector<T, 2> beta);	
T compute2DQuadraticShellEnergy(T lambda, T mu, T thickness, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef);

Vector<T, 9> computePointEnergyDensityGradient(T lambda, T mu, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, const Vector<T, 2> beta);
Vector<T, 9> compute2DQuadraticShellEnergyGradient(T lambda, T mu, T thickness, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef);	

Matrix<T, 9, 9> computePointEnergyDensityHessian(T lambda, T mu, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, const Vector<T, 2> beta);
Matrix<T, 9, 9> compute2DQuadraticShellEnergyHessian(T lambda, T mu, T thickness, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef);		

#endif