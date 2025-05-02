#ifndef OPTIMIZATION_PROBLEM_H
#define OPTIMIZATION_PROBLEM_H

#include "Scene.h"
#include "cost_function.h"
#include "ObjectiveEnergy.h"
#include <memory>

class OptimizationProblem
{
public:
	std::vector<std::shared_ptr<ObjectiveEnergy>> objective_energies;

	Scene* scene;

	VectorXa x;
	VectorXa p;

	VectorXa full_p;

	Eigen::SparseMatrix<AScalar> weights_p;

    std::string output_loc;
    const double cut_lower_bound = 0.001;

	std::function<bool(VectorXa&)> check_if_valid_p = [](VectorXa& parameters){return true;};
	std::function<void(VectorXa&)> on_iteration_accept = [](VectorXa& parameters){};

    bool Optimize();
    OptimizationProblem(Scene* scene_m, std::string out_m, std::string initial_file="");

	// bool OptimizeLBFGS();

	void TestSensitivityGradient();
};

class OptimizationProblemCostFunction : public CostFunction
{
public:
	OptimizationProblem* data;

	Eigen::SparseMatrix<AScalar> d2fdx2;
	Eigen::SparseMatrix<AScalar> d2fdxp;
	Eigen::SparseMatrix<AScalar> d2fdp2;
	Eigen::SparseMatrix<AScalar> dcdx;
	Eigen::SparseMatrix<AScalar> dcdp;
	VectorXa dfdx;
	VectorXa dfdp;
	VectorXa mark;

	AScalar damping = 100;

	void UpdateSensitivities();

	AScalar ComputeEnergy();
	VectorXa ComputeGradient();
	Eigen::SparseMatrix<AScalar> ComputeHessian();

	OptimizationProblemCostFunction(OptimizationProblem* data_m);

	virtual cost_evaluation Evaluate(const VectorXa& parameters);

	virtual void AcceptStep();

	virtual void RejectStep();

	virtual void TakeStep(const VectorXa& step, const VectorXa& prev_parameters, VectorXa& new_parameters);

    virtual void Finalize(const VectorXa& parameters);
};

#endif