#ifndef OPTIMIZATION_PROBLEM_H
#define OPTIMIZATION_PROBLEM_H

#include "Scene.h"
#include "cost_function.h"
#include "ObjectiveEnergy.h"
#include <memory>
#include <ceres/ceres.h>
#include <glog/logging.h>

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

	bool OptimizeGD();
	bool OptimizeGDFD();
	bool OptimizeFD();

	void TestOptimizationGradient();
	void ShowEnergyLandscape();
	void TestOptimizationSensitivity();
};

class OptimizationProblemCostFunction : public CostFunction
{
public:
	OptimizationProblem* data;

	VectorXa dfdp;
	VectorXa dfdx_sim;
	std::vector<Eigen::SparseMatrix<AScalar>> d2fdx2_sim;
	std::vector<Eigen::SparseMatrix<AScalar>> d2fdxp_sim;
	std::vector<Eigen::SparseMatrix<AScalar>> d2fdp2_sim;

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

class OptimizationProblemUpdateCallback : public ceres::IterationCallback
{
public:
	OptimizationProblem* data;
	double* parameters;

	ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary);
};

class OptimizationProblemCostFunctionCeres
{
public:
	OptimizationProblem* data;

	AScalar ComputeEnergy() const;
	VectorXa ComputeGradient() const;

	OptimizationProblemCostFunctionCeres(OptimizationProblem* data_m);

	bool Evaluate(double* parameters, double* cost, double* gradient) const;

	int NumParameters() const;
};

class OptimizationProblemCostFunctionFD : public CostFunction
{
public:
	OptimizationProblem* data;
	Eigen::SparseMatrix<AScalar> hessian_0;

	AScalar ComputeEnergy() const;
	VectorXa ComputeGradient() const;
	Eigen::SparseMatrix<AScalar> ComputeHessian();

	OptimizationProblemCostFunctionFD(OptimizationProblem* data_m);

	bool Evaluate(double* parameters, double* cost, double* gradient) const;
	virtual cost_evaluation Evaluate(const VectorXa& parameters);

	virtual void AcceptStep();

	virtual void RejectStep();

	virtual void TakeStep(const VectorXa& step, const VectorXa& prev_parameters, VectorXa& new_parameters);

    virtual void Finalize(const VectorXa& parameters);

	int NumParameters() const;
};

#endif