#include "../include/OptimizationProblem.h"
#include "../include/damped_newton.h"
#include "../include/enforce_matrix_constraints.h"
#include <Eigen/CholmodSupport>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>

bool OptimizationProblem::Optimize()
{
	// so that the parameter is updated in its order of magnitude
	if(weights_p.rows() == 0)
	{
		weights_p = Eigen::SparseMatrix<AScalar>(p.rows(), p.rows());
		for(int i=0; i<p.rows(); ++i)
			weights_p.coeffRef(i,i) = 1.0*p(0);
	}

	OptimizationProblemCostFunction cost_function(this);

	VectorXa parameters(x.rows()*2 + p.rows());
	parameters.setZero();
	parameters.segment(x.rows(), p.rows()) = p/p(0);

	Eigen::SparseMatrix<AScalar> damp_matrix(parameters.rows(), parameters.rows());
	for(int i=0; i<p.rows(); ++i)
		damp_matrix.coeffRef(x.rows()+i, x.rows()+i) = 1.0;

	damped_newton_options options;
	options.solver_type = DN_SOLVER_LU;
	options.damping = 5e-6;
	options.global_stopping_criteria = 7e-3;
	options.change_stopping_criteria = 1e-9;
	options.damp_matrix = damp_matrix;
	options.max_iterations = 300;
	options.output_log = output_loc+"_sgn";

	DampedNewtonSolver solver;
	solver.SetParameters(parameters);
	solver.SetCostFunction(&cost_function);
	solver.SetOptions(std::move(options));
	damped_newton_result result = solver.Solve();

	VectorXa opti_end_params = scene->parameters;
    std::ofstream out_file(output_loc+"_sgn_params.dat");
    if (!out_file) {
        std::cerr << "Error opening file for writing: " << output_loc << std::endl;
    }
    out_file << opti_end_params << "\n";
    out_file.close();

	return result.gradient < 1e-2;
}

struct GradientDescentOptions {
    int max_iterations = 1000;
    double initial_step_size = 1.0;
    double tolerance = 1e-3;
    double alpha = 0.3; // Armijo rule
    double beta = 0.8;  // backtracking
	std::string output_file;
};

struct GradientDescentSummary {
    int num_iterations;
    bool converged;
    double final_cost;

    std::string BriefReport() const {
        return converged ? 
            "Gradient descent converged." : 
            "Gradient descent did NOT converge.";
    }
};

template <typename CostFunctor>
GradientDescentSummary GradientDescent(
    const GradientDescentOptions& options,
    CostFunctor cost_function,
    std::vector<double>& x)
{
    GradientDescentSummary summary;
    int n = x.size();
    std::vector<double> grad(n);
    std::vector<double> new_grad(n);
    std::vector<double> x_new(n);
	std::ofstream log(options.output_file);

    // std::cout << std::fixed << std::setprecision(10);
    std::cout << " Iter     h_norm         step size         Cost         New_Cost       r_norm     r_new_norm      T_time        Status\n";
	// log << std::fixed << std::setprecision(10);
    log << " Iter     h_norm         step size         Cost         New_Cost       r_norm     r_new_norm      T_time        Status\n";

    for (int iter = 1; iter <= options.max_iterations; ++iter) {
        auto t_start = std::chrono::high_resolution_clock::now();

        double cost = 0.0;
        cost_function.Evaluate(x.data(), &cost, grad.data());

        double grad_norm = 0.0;
        for (double g : grad) grad_norm += g * g;
        grad_norm = std::sqrt(grad_norm);

        if (grad_norm < options.tolerance) {
            summary.converged = true;
            summary.num_iterations = iter - 1;
            summary.final_cost = cost;
            return summary;
        }

        // Descent direction
        std::vector<double> dir(n);
        for (int i = 0; i < n; ++i)
            dir[i] = -grad[i];

        double dir_norm = 0.0;
        for (double d : dir) dir_norm += d * d;
        dir_norm = std::sqrt(dir_norm);

        // Line search
        double step = options.initial_step_size;
        bool accepted = false;
        double new_cost = cost;

        while (true) {
            for (int i = 0; i < n; ++i)
                x_new[i] = x[i] + step * dir[i];

            cost_function.Evaluate(x_new.data(), &new_cost, new_grad.data());

            double dot = 0.0;
            for (int i = 0; i < n; ++i)
                dot += grad[i] * (x_new[i] - x[i]);

            if (new_cost <= cost + options.alpha * dot) {
                accepted = true;
                break;
            }

            step *= options.beta;
            if (step < 1e-8) break;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double time_sec = std::chrono::duration<double>(t_end - t_start).count();

        // Output logging
        double new_grad_norm = 0.0;
        for (double g : new_grad) new_grad_norm += g * g;
        new_grad_norm = std::sqrt(new_grad_norm);

        std::cout << std::setw(5) << iter << "   "
                  << std::setw(12) << dir_norm << "   "
                  << std::setw(12) << step << "   "
                  << std::setw(10) << cost << "   "
                  << std::setw(12) << new_cost << "   "
                  << std::setw(10) << grad_norm << "   "
                  << std::setw(14) << new_grad_norm << "   "
                  << "T=" << std::setw(5) /*<< std::setprecision(15)*/ << time_sec << "s   "
                  << (accepted ? "ACCEPTED" : "REJECTED") << "\n";
			
		log << std::setw(5) << iter << "   "
                  << std::setw(12) << dir_norm << "   "
                  << std::setw(12) << step << "   "
                  << std::setw(10) << cost << "   "
                  << std::setw(12) << new_cost << "   "
                  << std::setw(10) << grad_norm << "   "
                  << std::setw(14) << new_grad_norm << "   "
                  << "T=" << std::setw(5) /*<< std::setprecision(15)*/ << time_sec << "s   "
                  << (accepted ? "ACCEPTED" : "REJECTED") << "\n";		  

        if (!accepted) {
            summary.converged = false;
            summary.num_iterations = iter;
            summary.final_cost = new_cost;
            return summary;
        }

        x = x_new;
        grad = new_grad;
    }

    summary.converged = false;
    summary.num_iterations = options.max_iterations;
    cost_function.Evaluate(x.data(), &summary.final_cost, grad.data());
	log.close();
	return summary;
}

bool OptimizationProblem::OptimizeGD()
{
	if(weights_p.rows() == 0)
	{
		weights_p = Eigen::SparseMatrix<AScalar>(p.rows(), p.rows());
		for(int i=0; i<p.rows(); ++i)
			weights_p.coeffRef(i,i) = p(0);
	}

	std::vector<AScalar> parameters(p.rows());
	for(int i=0; i<p.rows(); ++i)
		parameters[i] = p(i)/p(0);;	


	OptimizationProblemCostFunctionCeres cost_function(this);
	GradientDescentOptions options;
	options.initial_step_size = 1e-2;
	options.tolerance = 5e-4;
	options.output_file = output_loc + "_gd.log";
	auto summary = GradientDescent<OptimizationProblemCostFunctionCeres>(options, cost_function, parameters);
	std::cout << summary.BriefReport() << "\n";
	std::string op_result = output_loc + "_gd_params.dat";
	std::ofstream out(op_result);
	for(int i=0; i<p.rows(); ++i)
		out << parameters[i]*weights_p.coeffRef(i,i) << std::endl;
	out.close();	
    
	return true;
}

OptimizationProblem::OptimizationProblem(Scene* scene_m, std::string out_m, std::string initial_file): scene(scene_m), output_loc(out_m){

	if(initial_file != ""){
        std::vector<double> parameter_from_file;
        std::ifstream in_file(initial_file);
        if (!in_file) {
            std::cerr << "Error opening file for reading: " << initial_file << std::endl;
        }

        AScalar a;
        while (in_file >> a) {
            parameter_from_file.push_back(a);
        }
        for(int i = 0; i < parameter_from_file.size(); ++i){
            scene->parameters(i) = parameter_from_file[i];
        }
    }
	p = scene->parameters;
	x.resize(scene->x_dof());

}

AScalar OptimizationProblemCostFunction::ComputeEnergy()
{
	AScalar energy = 0;

	for(int i=0; i<data->objective_energies.size(); ++i)
	{
		AScalar energy_ele = data->objective_energies[i]->ComputeEnergy(data->scene);
		// std::cout << "Energy " << i << ": " << energy_ele << std::endl;
		energy += energy_ele;
	}

	return energy;
}

void OptimizationProblemCostFunction::UpdateSensitivities(){

	dcdx = Eigen::SparseMatrix<AScalar>(data->x.rows(), data->x.rows()); dcdx.setZero();
	dfdx = VectorXa(data->x.rows()); dfdx.setZero();
	dfdp = VectorXa(data->full_p.rows()); dfdp.setZero();
	dcdp = Eigen::SparseMatrix<AScalar>(data->x.rows(), data->full_p.rows()); dcdp.setZero();
	d2fdx2 = Eigen::SparseMatrix<AScalar>(data->x.rows(), data->x.rows()); d2fdx2.setZero();
	d2fdxp = Eigen::SparseMatrix<AScalar>(data->full_p.rows(), data->x.rows()); d2fdxp.setZero();
	d2fdp2 = Eigen::SparseMatrix<AScalar>(data->full_p.rows(), data->full_p.rows()); d2fdp2.setZero();

	dfdx_sim = VectorXa(data->x.rows()*data->scene->num_test); dfdx_sim.setZero();

	for(int i=0; i<data->objective_energies.size(); ++i)
	{
		dfdx_sim += data->objective_energies[i]->Compute_dfdx_sim(data->scene);
		// Note that this is kinda ugly as Compute_dcdx setup a new simulation for constraints related modification (fixed dof)
		dcdx += data->objective_energies[i]->Compute_dcdx(data->scene);
		dcdp += data->objective_energies[i]->Compute_dcdp(data->scene) * data->weights_p;
		dfdx += data->objective_energies[i]->Compute_dfdx(data->scene);
		dfdp += data->objective_energies[i]->Compute_dfdp(data->scene) * data->weights_p;
		d2fdx2 += data->objective_energies[i]->Compute_d2fdx2(data->scene);
		d2fdxp += data->objective_energies[i]->Compute_d2fdxp(data->scene) * data->weights_p;
		d2fdp2 += data->weights_p.transpose() * data->objective_energies[i]->Compute_d2fdp2(data->scene) * data->weights_p;

	}
	// std::cout << "A: " << d2fdx2.norm() << std::endl;
	// // // std::cout << "dfdx norm: " << dfdx.norm() << std::endl;
	// std::cout << "B: " << d2fdxp.norm() << std::endl;
	// std::cout << "C: " << d2fdp2.norm() << std::endl;

}

VectorXa OptimizationProblemCostFunction::ComputeGradient()
{
	VectorXa gradient(data->x.rows()*2+data->p.rows()); gradient.setZero();

	VectorXa sum(data->p.rows()); sum.setZero();
	for(int i = 0; i < data->scene->num_test; ++i){
		Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<AScalar>> solver_sim(data->scene->hessian_sims[i]);
		VectorXa dd_sim = solver_sim.solve(dfdx_sim.segment(data->x.rows()*i, data->x.rows()));
		sum += (data->scene->hessian_p_sims[i]* data->weights_p).transpose() * dd_sim;
	}

	gradient.segment(data->x.rows(), data->p.rows()) = dfdp-sum;
	std::cout << "simulation gradient: " << sum.norm() << std::endl;
	std::cout << "param gradient: " << dfdp.norm() << std::endl;

	return gradient;
}

Eigen::SparseMatrix<AScalar> OptimizationProblemCostFunction::ComputeHessian()
{
	Eigen::SparseMatrix<AScalar> hessian(data->x.rows()*2 + data->p.rows(), data->x.rows()*2 + data->p.rows());

	std::vector<Eigen::Triplet<AScalar>> triplets;

	std::vector<Eigen::Triplet<AScalar> > A = SparseMatrixToTriplets(d2fdx2);
	std::vector<Eigen::Triplet<AScalar> > B = SparseMatrixToTriplets(d2fdxp.transpose());
	std::vector<Eigen::Triplet<AScalar> > C = SparseMatrixToTriplets(d2fdp2);
	std::vector<Eigen::Triplet<AScalar> > dcdx_t = SparseMatrixToTriplets(dcdx);
	std::vector<Eigen::Triplet<AScalar> > dcdp_t = SparseMatrixToTriplets(dcdp);

	triplets = A;

	int ob_col = data->x.rows();
	for(int i=0; i<B.size(); ++i)
	{
		triplets.push_back(Eigen::Triplet<AScalar>(ob_col+B[i].col(), B[i].row(),        B[i].value())); //Bt
		triplets.push_back(Eigen::Triplet<AScalar>(B[i].row(), 		  ob_col+B[i].col(), B[i].value())); //B
	}

	int oc_col = data->x.rows();
	int oc_row = data->x.rows();
	for(int i=0; i<C.size(); ++i)
		triplets.push_back(Eigen::Triplet<AScalar>(oc_row+C[i].row(), oc_row+C[i].col(), C[i].value())); //C
	
	int odcdx_row = data->x.rows() + data->p.rows();
	for(int i=0; i<dcdx_t.size(); ++i)
	{
		triplets.push_back(Eigen::Triplet<AScalar>(odcdx_row+dcdx_t[i].row(), dcdx_t[i].col(),           dcdx_t[i].value())); //dcdx
		triplets.push_back(Eigen::Triplet<AScalar>(dcdx_t[i].col(),           odcdx_row+dcdx_t[i].row(), dcdx_t[i].value())); //dcdxt
	}

	int odcdp_row = data->x.rows()+data->p.rows();
	int odcdp_col = data->x.rows();
	for(int i=0; i<dcdp_t.size(); ++i)
	{
		triplets.push_back(Eigen::Triplet<AScalar>(odcdp_row+dcdp_t[i].row(), odcdp_col+dcdp_t[i].col(), dcdp_t[i].value())); //dcdp
		triplets.push_back(Eigen::Triplet<AScalar>(odcdp_col+dcdp_t[i].col(), odcdp_row+dcdp_t[i].row(), dcdp_t[i].value())); //dcdpt
	}

	hessian.setFromTriplets(triplets.begin(), triplets.end());

	// std::cout << "test A norm " << (hessian.block(0,0,data->x.rows(),data->x.rows())).norm() << std::endl;

	// std::cout << "test B norm " << (hessian.block(data->x.rows(),0,data->p.rows(),data->x.rows())).norm() << std::endl;
	// std::cout << "test Bt norm " << (hessian.block(0,data->x.rows(),data->x.rows(),data->p.rows())).norm() << std::endl;

	// std::cout << "test C norm " << (hessian.block(data->x.rows(),data->x.rows(),data->p.rows(),data->p.rows())).norm() << std::endl;

	// std::cout << "test dcdx norm " << (hessian.block(data->x.rows()+data->p.rows(),0,data->x.rows(),data->x.rows())).norm() << std::endl;
	// std::cout << "test dcdxt norm " << (hessian.block(data->x.rows()+data->p.rows(),0,data->x.rows(),data->x.rows())).norm() << std::endl;

	// std::cout << "dcdp norm " << dcdp.norm() << std::endl;
	// std::cout << "test dcdp norm " << (hessian.block(data->x.rows()+data->p.rows(),data->x.rows(),data->x.rows(),data->p.rows())).norm() << std::endl;
	// std::cout << "test dcdpt norm " << (hessian.block(data->x.rows(),data->x.rows()+data->p.rows(),data->p.rows(),data->x.rows())).norm() << std::endl;

	   // for(int i=0; i<hessian.rows(); ++i)
	   // 	hessian.coeffRef(i,i)=1.0;

	return hessian;
}

OptimizationProblemCostFunction::OptimizationProblemCostFunction(OptimizationProblem* data_m): data(data_m){}

void OptimizationProblemCostFunction::TakeStep(const VectorXa& step, const VectorXa& prev_parameters, VectorXa& new_parameters)
{
	new_parameters = prev_parameters + step;
	std::cout << "residual on rest 0: " << step.segment(0, data->x.rows()).lpNorm<Eigen::Infinity>() << std::endl;
	std::cout << "residual on rest 1: " << step.segment(data->x.rows(), data->p.rows()).lpNorm<Eigen::Infinity>() << std::endl;
	std::cout << "residual on rest 2: " << step.segment(data->x.rows() + data->p.rows(), data->x.rows()).lpNorm<Eigen::Infinity>() << std::endl;
}

cost_evaluation OptimizationProblemCostFunction::Evaluate(const VectorXa& parameters)
{
	data->p = parameters.segment(data->x.rows(), data->p.rows());

	data->full_p = data->weights_p*data->p;

	if(!data->check_if_valid_p(data->full_p))
		throw std::exception();

	std::cout << "---------------------------- SIMULATION ----------------------------" << std::endl;
	data->scene->parameters = data->full_p;
	// data->scene->parameters = data->scene->parameters.cwiseMax(data->cut_lower_bound);
	for(int i=0; i<data->objective_energies.size(); ++i)
	{
		data->objective_energies[i]->SimulateAndCollect(data->scene);
	}
	std::cout << "--------------------------------------------------------------------" << std::endl;

	UpdateSensitivities();

	AScalar energy = ComputeEnergy();
	VectorXa gradient = ComputeGradient();
	Eigen::SparseMatrix<AScalar> hessian = ComputeHessian();

	std::cout << "Sensitivities computed" << std::endl;

	return std::tie(energy, gradient, hessian);
}

void OptimizationProblemCostFunction::RejectStep()
{
	
}

void OptimizationProblemCostFunction::AcceptStep()
{
	data->on_iteration_accept(data->full_p);
}

void OptimizationProblemCostFunction::Finalize(const VectorXa& parameters)
{
	data->p = parameters.segment(data->x.rows(), data->p.rows()).cwiseMax(data->cut_lower_bound);
}

void OptimizationProblem::TestOptimizationGradient(){

	if(weights_p.rows() == 0)
	{
		weights_p = Eigen::SparseMatrix<AScalar>(p.rows(), p.rows());
		for(int i=0; i<p.rows(); ++i)
			weights_p.coeffRef(i,i) = p(0); //1.0;
	}
	VectorXa parameters(x.rows()*2 + p.rows());
	parameters.setZero();

	OptimizationProblemCostFunction cost_function(this);

	std::cout << std::endl << std::endl;
	std::cout << "<---------------------------------- TESTING GRADIENT ----------------------------------> " << std::endl;
	parameters.segment(x.rows(), p.rows()) = p/p(0);
	VectorXa init_p = p/p(0);
	AScalar cost;
	VectorXa gradient;
	Eigen::SparseMatrix<AScalar> hessian;
	std::tie(cost, gradient, hessian) = cost_function.Evaluate(parameters);

	int test_size = 10; 
    VectorXa errors(test_size); 
	// AScalar step = init_p(0)/100;
	AScalar step = 10;
    VectorXa delta_h(p.rows()); delta_h.setConstant(step);
    for(int i = 0; i < test_size; ++i){

        AScalar obj_1 = cost + (gradient.segment(x.rows(), p.rows())).transpose() * (delta_h/std::pow(2, i));
        parameters.segment(x.rows(), p.rows()) = init_p + delta_h/std::pow(2, i);
		AScalar cost_fd;
		VectorXa gradient_fd;
		Eigen::SparseMatrix<AScalar> hessian_fd;
        std::tie(cost_fd, gradient_fd, hessian_fd) = cost_function.Evaluate(parameters);
        
        errors(i) = std::abs(cost_fd-obj_1);
    }
    for(int i = 1; i < test_size; ++i){
        std::cout <<  step/std::pow(2, i)  << " - " << errors(i-1)/errors(i) << std::endl;
    }
}

void OptimizationProblem::ShowEnergyLandscape(){

	if(weights_p.rows() == 0)
	{
		weights_p = Eigen::SparseMatrix<AScalar>(p.rows(), p.rows());
		for(int i=0; i<p.rows(); ++i)
			weights_p.coeffRef(i,i) = p(0); //1.0;
	}
	VectorXa parameters(x.rows()*2 + p.rows());
	parameters.setZero();

	OptimizationProblemCostFunction cost_function(this);

	std::cout << std::endl;
	std::cout << "<---------------------------------- Energy Landscape ----------------------------------> " << std::endl;
	parameters.segment(x.rows(), p.rows()) = p/p(0);
	VectorXa init_p = p/p(0);
	AScalar cost;
	VectorXa gradient;
	Eigen::SparseMatrix<AScalar> hessian;
	std::tie(cost, gradient, hessian) = cost_function.Evaluate(parameters);

	int test_size = 10; 
    VectorXa errors(test_size); 
	// AScalar step = init_p(0)/100;
	AScalar step = 0.01;
    VectorXa delta_h(p.rows()); delta_h.setConstant(step);
    for(int i = -test_size; i < test_size; ++i){

        parameters.segment(x.rows(), p.rows()) = init_p + delta_h * i;
		AScalar cost_fd;
		VectorXa gradient_fd;
		Eigen::SparseMatrix<AScalar> hessian_fd;
        std::tie(cost_fd, gradient_fd, hessian_fd) = cost_function.Evaluate(parameters);
		std::cout << i << " : " << cost_fd << std::endl;
    }

}

void OptimizationProblem::TestOptimizationSensitivity(){

	if(weights_p.rows() == 0)
	{
		weights_p = Eigen::SparseMatrix<AScalar>(p.rows(), p.rows());
		for(int i=0; i<p.rows(); ++i)
			weights_p.coeffRef(i,i) = p(0); //1.0;
	}
	VectorXa parameters(x.rows()*2 + p.rows());
	parameters.setZero();

	OptimizationProblemCostFunction cost_function(this);
	std::cout << std::endl;
	std::cout << "<---------------------------------- Sensitivity for Objective ----------------------------------> " << std::endl;
	parameters.segment(x.rows(), p.rows()) = p/p(0);
	AScalar cost;
	VectorXa gradient;
	Eigen::SparseMatrix<AScalar> hessian;
	std::tie(cost, gradient, hessian) = cost_function.Evaluate(parameters);

	int test_size = 10; 
    VectorXa errors(test_size); 
	// AScalar step = init_p(0)/100;
	AScalar step = 0.01;
    MatrixXa delta_h(scene->num_test, x.rows()); delta_h.setZero();
	Vector3a s; s.setConstant(step);
	// delta_h(2, 54*3) = step;
	delta_h.col(52*3) = s;
    for(int i = 0; i < test_size; ++i){

		AScalar correction = ((cost_function.dfdx_sim).segment(0, x.rows())).transpose() * (delta_h.row(0).transpose()/std::pow(2, i));
		// std::cout << "correction: " << correction << std::endl; 
		for(int j = 1; j < scene->num_test; ++j){
			correction += ((cost_function.dfdx_sim).segment(x.rows()*j, x.rows())).transpose() * (delta_h.row(j).transpose()/std::pow(2, i));
		}
		// std::cout << "correction: " << correction << std::endl; 
        AScalar obj_1 = cost + correction;
		for(int j=0; j<objective_energies.size(); ++j)
		{
			objective_energies[j]->OnlyCollect(scene, delta_h/std::pow(2, i));
		}
	
		AScalar energy = cost_function.ComputeEnergy();
		std::cout << "energy: " << energy << std::endl; 
        
        errors(i) = std::abs(energy-obj_1);
    }
    for(int i = 1; i < test_size; ++i){
		// std::cout << errors(i) << std::endl;
        std::cout <<  step/std::pow(2, i)  << " - " << errors(i-1)/errors(i) << std::endl;
    }

}

ceres::CallbackReturnType OptimizationProblemUpdateCallback::operator()(const ceres::IterationSummary& summary)
{
	VectorXa params(data->p.rows());

	for(int i=0; i<params.rows(); ++i)
		params[i] = parameters[i];

	data->on_iteration_accept(params);

	return ceres::SOLVER_CONTINUE;
}

AScalar OptimizationProblemCostFunctionCeres::ComputeEnergy() const
{
	AScalar energy = 0;

	for(int i=0; i<data->objective_energies.size(); ++i)
	{
		AScalar energy_ele = data->objective_energies[i]->ComputeEnergy(data->scene);
		// std::cout << "Energy " << i << ": " << energy_ele << std::endl;
		energy += energy_ele;
	}

	return energy;
}

VectorXa OptimizationProblemCostFunctionCeres::ComputeGradient() const
{	
	VectorXa dfdp(data->full_p.rows()); dfdp.setZero();
	VectorXa dfdx_sim = VectorXa(data->x.rows()*data->scene->num_test); dfdx_sim.setZero();
	for(int i=0; i<data->objective_energies.size(); ++i)
	{	
		dfdx_sim += data->objective_energies[i]->Compute_dfdx_sim(data->scene);
		dfdp += data->objective_energies[i]->Compute_dfdp(data->scene) * data->weights_p;
	}
	VectorXa gradient(data->p.rows()); gradient.setZero();

	VectorXa sum(data->p.rows()); sum.setZero();
	for(int i = 0; i < data->scene->num_test; ++i){
		Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<AScalar>> solver_sim(data->scene->hessian_sims[i]);
		VectorXa dd_sim = solver_sim.solve(dfdx_sim.segment(data->x.rows()*i, data->x.rows()));
		sum += (data->scene->hessian_p_sims[i] * data->weights_p).transpose() * dd_sim;
		// std::cout << "sim " << i << " gradient : " << ((data->scene->hessian_p_sims[i] * data->weights_p).transpose() * dd_sim).norm() << std::endl;
	}

	std::cout << "simulation gradient: " << sum.norm() << std::endl;
	std::cout << "param gradient: " << dfdp.norm() << std::endl;

	gradient = dfdp-sum;

	return gradient;
}

OptimizationProblemCostFunctionCeres::OptimizationProblemCostFunctionCeres(OptimizationProblem* data_m) : data(data_m)
{

}

bool OptimizationProblemCostFunctionCeres::Evaluate(double* parameters, double* cost, double* gradient) const
{
	for(int i=0; i<data->p.rows(); ++i)
		data->p[i] =  parameters[i];
	
	data->full_p = data->weights_p*data->p;

	if(!data->check_if_valid_p(data->full_p))
		return false;

	std::cout << "---------------------------- SIMULATION ----------------------------" << std::endl;
	data->scene->parameters = data->full_p;
	// data->scene->parameters = data->scene->parameters.cwiseMax(data->cut_lower_bound);
	for(int i=0; i<data->objective_energies.size(); ++i)
	{
		data->objective_energies[i]->SimulateAndCollect(data->scene);
	}
	std::cout << "--------------------------------------------------------------------" << std::endl;

	AScalar energy = ComputeEnergy();
	std::cout << "Computing gradient" << std::endl;
	VectorXa gradient_e = ComputeGradient();
	std::cout << "Done" << std::endl;

	*cost = energy;

	for(int i=0; i<gradient_e.rows(); ++i)
		gradient[i] = gradient_e[i];

	return true;
}

int OptimizationProblemCostFunctionCeres::NumParameters() const
{
	return data->p.rows();
}