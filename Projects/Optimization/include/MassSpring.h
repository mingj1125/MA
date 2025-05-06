#ifndef MASS_SPRING_H
#define MASS_SPRING_H

#include "Simulation.h"
#include "damped_newton.h"
#include "cost_function.h"

class MassSpring : public Simulation {

    public:

    struct Spring {
        int p1, p2;
        int spring_id;
        const AScalar YoungsModulus = 3.5e6;
        Vector3a rest_tangent;
        AScalar width;
        AScalar height = 0.01;
    
        Spring(int u_id, int v_id, int id, Vector3a u_pos, Vector3a v_pos){
            p1 = u_id;
            p2 = v_id;
            spring_id = id;
            rest_tangent = (v_pos - u_pos).normalized();
        }
        
        void set_width(AScalar w){
            width = w;
        }

        AScalar k_s(){
            return YoungsModulus * width * height;
        }

        Matrix3a computeSecondPiolaStress(Vector3a xi, Vector3a xj, Vector3a Xi, Vector3a Xj);
    };

    int n_nodes;
    VectorXa deformed_states;
    VectorXa rest_states;
    std::vector<Spring*> springs;
    VectorXa spring_widths;
    AScalar initial_width = 1e-2;
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
    virtual VectorXa get_initial_parameter(){VectorXa p(springs.size()); p.setConstant(initial_width); return p;}
    virtual VectorXa get_undeformed_nodes(){return rest_states;}
    virtual VectorXa get_deformed_nodes(){return deformed_states;}
    virtual std::vector<int> get_constraint_map(){return fixed_vertices;}
    virtual std::vector<std::array<size_t, 2>> get_edges();
    virtual Matrix3a findBestStressTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions);
    virtual Matrix3a findBestStrainTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions);
    virtual void setOptimizationParameter(VectorXa parameters){
        spring_widths = parameters;
        for(auto spring: springs){
            spring->set_width(spring_widths(spring->spring_id));
        }
    }
    virtual void build_d2Edx2(Eigen::SparseMatrix<AScalar>& K);
    virtual void build_d2Edxp(Eigen::SparseMatrix<AScalar>& K);
    virtual void applyBoundaryStretch(int i, AScalar strain = 0);
    virtual MatrixXa getStressGradientWrtParameter(){return eval_info_of_sample.stress_gradients_wrt_spring_thickness;}
    virtual MatrixXa getStressGradientWrtx(){return eval_info_of_sample.stress_gradients_wrt_x;}
    virtual MatrixXa getStrainGradientWrtx();

    // Native function for mass spring 
    void addPoint(std::vector<Vector3a>& existing_nodes, 
        const Vector3a& point, int& full_dof_cnt, int& node_cnt);
    
    void resetSimulation();  
    void computeBoundingBox(Vector3a& top_right, Vector3a& bottom_left);
    void stretchX(AScalar strain);    
    void stretchY(AScalar strain);
    void stretchDiagonal(AScalar strain);
    void stretchSlidingY(AScalar strain);
    void stretchSlidingX(AScalar strain);

    // Native function for evaluation
    Vector3a computeWeightedStress(const Vector3a sample_loc, const Vector3a direction, 
        std::vector<Vector3a>& gradients_wrt_thickness, std::vector<Vector3a>& gradients_wrt_nodes);
    bool lineCutRodinSegment(Spring* spring, Vector3a sample_loc, Vector3a direction, AScalar& cut_point_barys);
    Vector3a integrateOverCrossSection(Spring* spring, const Vector3a normal, const AScalar center_line_distance_to_sample, 
        Vector3a& gradient_wrt_thickness, std::vector<Vector3a>& gradient_wrt_nodes);
    Eigen::Matrix<AScalar, 18, 3> SGradientWrtx(Spring* spring, Vector3a xi, Vector3a xj, Vector3a Xi, Vector3a Xj);
    AScalar integrateKernelOverDomain(const Vector3a sample_loc, const Vector3a line_direction);
    Matrix3a computeWeightedDeformationGradient(const Vector3a sample_loc, const std::vector<Vector3a> line_directions);

    // Simulator    
    AScalar global_stopping_criteria = 1e-5;
    int n_iter_hessian = 100;
    bool use_jacobi = false;
    int max_iterations = 1000;

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

class MassSpringCostFunction : public CostFunction
{
private:
	MassSpring* data;

	std::vector<int> constraints;

public:
	MassSpringCostFunction(MassSpring* data_p): data(data_p){constraints = data->fixed_vertices;}

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