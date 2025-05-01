#ifndef COST_FUNCTION_HPP
#define COST_FUNCTION_HPP

#include <tuple>
#include "vector_manipulation.h"

typedef std::tuple<AScalar, VectorXa, Eigen::SparseMatrix<AScalar>> cost_evaluation; //cost, gradient, hessian
typedef std::tuple<AScalar, VectorXa> reduced_cost_evaluation; //cost, gradient, hessian

struct TerminateNowException : public std::exception {
   const char * what () const throw () {
      return "Terminating ...";
   }
};

class CostFunction
{
public:
    virtual cost_evaluation Evaluate(const VectorXa& parameters)=0;

    virtual reduced_cost_evaluation EvaluateReduced(const VectorXa& parameters);

    virtual AScalar ComputeEnergy(const VectorXa& parameters);
    virtual VectorXa ComputeGradient(const VectorXa& parameters);
    virtual Eigen::SparseMatrix<AScalar> ComputeHessian(const VectorXa& parameters);

    virtual Eigen::SparseMatrix<AScalar> InexpensiveHessian();
    virtual Eigen::SparseMatrix<AScalar> ComputeJacobiPreconditioner();

    virtual void TakeStep(const VectorXa& step, const VectorXa& prev_parameters, VectorXa& new_parameters);

    virtual void TakePartialStep(AScalar multiplier);

    virtual void AcceptStep();
    virtual void RejectStep();

    virtual void PreProcess(VectorXa& parameters);

    virtual void Finalize(const VectorXa& parameters);

    virtual void TestGradient(VectorXa& parameters);
    virtual void TestGradient(VectorXa& parameters, int offset);

	virtual void TestHessian(VectorXa& parameters, int offset = 0);

  virtual void ShermanMorrisonVectors(VectorXa& u, VectorXa& v);

  virtual void WoodburyMatrices(MatrixXa& U, MatrixXa& C, MatrixXa& V);

};

#endif
