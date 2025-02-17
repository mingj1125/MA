#ifndef QUADRATIC_2D_SHELL_H
#define QUADRATIC_2D_SHELL_H

#include "../include/VecMatDef.h"

Matrix<T, 2, 6> compute2DdNdX(const Matrix<T,6,3> & undeformed_vertices, const Vector<T, 2> beta);
Matrix<T, 2, 6> compute2DdNdx(const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, const Vector<T, 2> beta);

T computePointEnergyDensity(T lambda, T mu, const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices, const Vector<T, 2> beta);	

Vector<T, 18> computePointEnergyDensityGradient(T lambda, T mu, const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices, const Vector<T, 2> beta);

Matrix<T, 18, 18> computePointEnergyDensityHessian(T lambda, T mu, const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices, const Vector<T, 2> beta);

#endif