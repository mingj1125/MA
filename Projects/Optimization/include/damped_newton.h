#ifndef DAMPED_NEWTON_HPP
#define DAMPED_NEWTON_HPP

#include "cost_function.h"

enum damped_newton_solver_type
{
	DN_SOLVER_LLT,
	DN_SOLVER_LU
};

struct damped_newton_options 
{
    int max_iterations = 1000; //k_max
    AScalar damping = 1e-6; //tao
    AScalar global_stopping_criteria = 1e-8; //e2
    AScalar change_stopping_criteria = 1e-8; //e1

    bool use_log = true;
    bool check_matrix = false;
    bool benchmark = true;

    bool accept_everything = false;
    bool sherman_morrison = false;
    bool woodbury = false;

    //Back-compatibility
    bool simplified = false;

    int n_constraints = 0;

    Eigen::SparseMatrix<AScalar> damp_matrix;

    damped_newton_solver_type solver_type = DN_SOLVER_LLT;
};

struct damped_newton_result {
    int n_iterations;
    int n_iterations_accepted;
    AScalar cost;
    AScalar gradient;
};

class DampedNewtonSolver {

private:
    VectorXa parameters;
    damped_newton_options options;
    CostFunction* function;

    AScalar MaxJtJDiagonalValue(Eigen::SparseMatrix<AScalar>& J);
	Eigen::SparseMatrix<AScalar> DiagonalMatrix(Eigen::SparseMatrix<AScalar>& A);

public:
    void SetParameters(VectorXa& parameters_p);
    void SetOptions(damped_newton_options&& options_p);
    void SetCostFunction(CostFunction* function_p);

    damped_newton_result Solve();

};

   
#endif