#ifndef LINEAR_SHELL_H
#define LINEAR_SHELL_H

#include "Simulation.h"
#include "damped_newton.h"
#include "cost_function.h"

class LinearShell : public Simulation
{
    
    public:

    int n_nodes;
    VectorXa deformed_states;
    VectorXa rest_states;
    VectorXa youngsmodulus_each_element;
    Eigen::MatrixXi faces;
    AScalar initial_youngsmodulus = 3.5e6;
    AScalar nu = 0.45;
    AScalar thickness = 0.001;
    std::vector<int> fixed_vertices;

    AScalar kernel_std = 0.1;

    struct EvaluationInfo
    {
        MatrixXa stress_gradients_wrt_spring_thickness;
        MatrixXa stress_gradients_wrt_x;
        Matrix3a stress_tensor;

        std::vector<Matrix3a> F_gradients_wrt_x;
        Matrix3a strain_tensor;
    };
    EvaluationInfo eval_info_of_sample; // not using a vector for samples since have to do sample eval one by one anyways

    virtual void initializeScene(const std::string& filename);
    virtual VectorXa get_initial_parameter(){return VectorXa p(youngsmodulus_each_element.rows()); p.setConstant(initial_youngsmodulus); return p;}
    virtual VectorXa get_undeformed_nodes(){return rest_states;}
    virtual VectorXa get_deformed_nodes(){return deformed_states;}
    virtual std::vector<int> get_constraint_map(){return fixed_vertices;}
    virtual std::vector<std::array<size_t, 2>> get_edges(){}
    virtual Matrix3a findBestStressTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){}
    virtual Matrix3a findBestStrainTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){}
    virtual void setOptimizationParameter(VectorXa parameters){
        youngsmodulus_each_element = parameters;
    }
    virtual void build_d2Edx2(Eigen::SparseMatrix<AScalar>& K){}
    virtual void build_d2Edxp(Eigen::SparseMatrix<AScalar>& K){}
    virtual void applyBoundaryStretch(int i, AScalar strain = 0);
    virtual MatrixXa getStressGradientWrtParameter(){return eval_info_of_sample.stress_gradients_wrt_spring_thickness;}
    virtual MatrixXa getStressGradientWrtx(){return eval_info_of_sample.stress_gradients_wrt_x;}
    virtual MatrixXa getStrainGradientWrtx(){}

    // Native function
    void resetSimulation();  
    void stretchX(AScalar strain);    
    void stretchY(AScalar strain);
    void stretchDiagonal(AScalar strain);
    void stretchSlidingY(AScalar strain);
    void stretchSlidingX(AScalar strain);

    // Simulator    
    AScalar global_stopping_criteria = 1e-5;
    int n_iter_hessian = 100;
    bool use_jacobi = false;
    int max_iterations = 100;

    std::function<bool()> are_parameters_ok = [](){return true;};

    std::function<void(VectorXa&)> pre_process = [](VectorXa& parameters){};
    std::function<void()> accept_step = [](){};
    std::function<void()> reject_step = [](){};

    std::function<void(const VectorXa&, const VectorXa&, VectorXa&)> take_step = 
        [](const VectorXa& step, const VectorXa& prev_parameters, VectorXa& new_parameters)
            {
                new_parameters = prev_parameters + step;
            };

    virtual damped_newton_result Simulate(bool use_log = true);
};

class LinearShellCostFunction : public CostFunction
{
private:
    LinearShell* data;

	std::vector<int> constraints;

public:
    LinearShellCostFunction(LinearShell* data_p): data(data_p){constraints = data->fixed_vertices;}

	bool is_xdef = true;

	AScalar ComputeEnergy();
	VectorXa ComputeGradient();
	Eigen::SparseMatrix<AScalar> ComputeHessian();

	virtual cost_evaluation Evaluate(const VectorXa& parameters);

	virtual void PreProcess(VectorXa& parameters);

	virtual void TakeStep(const VectorXa& step, const VectorXa& prev_parameters, VectorXa& new_parameters);
	virtual void AcceptStep();
	virtual void RejectStep();

	virtual void Finalize(const VectorXa& parameters);
};
#endif