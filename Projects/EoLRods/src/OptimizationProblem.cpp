#include "../include/OptimizationProblem.h"
#include "../include/damped_newton.h"
#include <Eigen/CholmodSupport>
#include <fstream>

std::vector<Eigen::Triplet<AScalar>> SparseMatrixToTriplets(const Eigen::SparseMatrix<AScalar>& A)
{
	std::vector<Eigen::Triplet<AScalar> > triplets;

	for (int k=0; k < A.outerSize(); ++k)
        for (Eigen::SparseMatrix<AScalar>::InnerIterator it(A,k); it; ++it)
        	triplets.push_back(Eigen::Triplet<AScalar>(it.row(), it.col(), it.value()));

    return triplets;
}

bool OptimizationProblem::Optimize()
{
	if(weights_p.rows() == 0)
	{
		weights_p = Eigen::SparseMatrix<AScalar>(p.rows(), p.rows());
		for(int i=0; i<p.rows(); ++i)
			weights_p.coeffRef(i,i) = 1.0;
	}

	OptimizationProblemCostFunction cost_function(this);

	VectorXa parameters(x.rows()*2 + p.rows());
	parameters.setZero();
	parameters.segment(x.rows(), p.rows()) = p;

	// gauss_newton_options options;
	// options.damping = 100.0;
	// options.global_stopping_criteria = 1e-3;
	// options.change_stopping_criteria = 1e-9;

	// GaussNewtonSolver solver;
	// solver.SetParameters(parameters);
	// solver.SetCostFunction(&cost_function);
	// solver.SetOptions(std::move(options));
	// gauss_newton_result result = solver.Solve();

	Eigen::SparseMatrix<AScalar> damp_matrix(parameters.rows(), parameters.rows());
	for(int i=0; i<p.rows(); ++i)
		damp_matrix.coeffRef(x.rows()+i, x.rows()+i) = 1.0;

	damped_newton_options options;
	options.solver_type = DN_SOLVER_LU;
	options.damping = 5e-5;
	options.global_stopping_criteria = 1e-3;
	options.change_stopping_criteria = 1e-9;
	options.damp_matrix = damp_matrix;
	options.output_log = output_loc;

	DampedNewtonSolver solver;
	solver.SetParameters(parameters);
	solver.SetCostFunction(&cost_function);
	solver.SetOptions(std::move(options));
	damped_newton_result result = solver.Solve();
	solver.GetParameters(parameters);

	// std::cout << result.gradient_vec.segment(x.rows(), p.rows()).transpose() << std::endl;
	// std::cout << std::endl;
	AScalar Fx; VectorXa r; Eigen::SparseMatrix<AScalar> J;
	AScalar step = 1e-11;
	for(int i = -10; i < 15; i++){
		VectorXa test_param = i * step * result.gradient_vec + parameters;
		std::tie(Fx, r, J) = cost_function.Evaluate(test_param);
		std::cout << "Step " << i << " : " << Fx << std::endl;
		std::cout << "Grad " << i << " : " << r.lpNorm<Eigen::Infinity>() << std::endl;
	}

	VectorXa rods_radii = parameters.segment(x.rows(), p.rows());
    std::cout << rods_radii.transpose() << std::endl; 
    std::ofstream out_file(output_loc+"_radii.dat");
    if (!out_file) {
        std::cerr << "Error opening file for writing: " << output_loc << std::endl;
    }
    out_file << rods_radii << "\n";
    out_file.close();

	return result.gradient < 1e-3;
}

OptimizationProblem::OptimizationProblem(Scene* scene_m, std::string out_m, std::string initial_file): scene(scene_m), output_loc(out_m){

	if(initial_file == ""){
        scene->rods_radii.resize(scene->num_rods());
        scene->rods_radii.setConstant(1e-2);
    } else {
        std::vector<double> rods_radius;
        std::ifstream in_file(initial_file);
        if (!in_file) {
            std::cerr << "Error opening file for reading: " << initial_file << std::endl;
        }

        T a;
        while (in_file >> a) {
            rods_radius.push_back(a);
        }
        scene->rods_radii.resize(rods_radius.size());
        for(int i = 0; i < rods_radius.size(); ++i){
            scene->rods_radii(i) = rods_radius[i];
        }
    }
	p = scene->rods_radii;
	x.resize(scene->simulationW().cols());

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
	for(int i=0; i<data->objective_energies.size(); ++i)
	{
		dcdx += data->objective_energies[i]->Compute_dcdx(data->scene);
		dcdp += data->objective_energies[i]->Compute_dcdp(data->scene);
		dfdx += data->objective_energies[i]->Compute_dfdx(data->scene);
		dfdp += data->objective_energies[i]->Compute_dfdp(data->scene);
		d2fdx2 += data->objective_energies[i]->Compute_d2fdx2(data->scene);
		d2fdxp += data->objective_energies[i]->Compute_d2fdxp(data->scene);
		d2fdp2 += data->objective_energies[i]->Compute_d2fdp2(data->scene);
	}
	data->scene->solveForAdjoint(dcdx, dfdx);
	
}

VectorXa OptimizationProblemCostFunction::ComputeGradient()
{
	VectorXa gradient(data->x.rows()*2+data->p.rows()); gradient.setZero();

	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<AScalar> > solver(dcdx);
	VectorXa dd = solver.solve(dfdx);

	gradient.segment(data->x.rows(), data->p.rows()) = (dfdp - dcdp.transpose() * dd);
	std::cout << "Gradient 2-norm: " << gradient.norm() << std::endl;

	return gradient;
}

Eigen::SparseMatrix<AScalar> OptimizationProblemCostFunction::ComputeHessian()
{
	Eigen::SparseMatrix<AScalar> hessian(data->x.rows()*2 + data->p.rows(), data->x.rows()*2 + data->p.rows());

	std::vector<Eigen::Triplet<AScalar>> triplets;

	std::vector<Eigen::Triplet<AScalar> > A = SparseMatrixToTriplets(d2fdx2);
	std::vector<Eigen::Triplet<AScalar> > B = SparseMatrixToTriplets(d2fdxp);
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
	// new_parameters.segment(data->x.rows(), data->p.rows()) = new_parameters.segment(data->x.rows(), data->p.rows()).cwiseMax(cut_lower_bound);
}

cost_evaluation OptimizationProblemCostFunction::Evaluate(const VectorXa& parameters)
{
	data->p = parameters.segment(data->x.rows(), data->p.rows());

	data->full_p = data->weights_p*data->p;

	if(!data->check_if_valid_p(data->full_p))
		throw std::exception();

	std::cout << "---------------------------- SIMULATION ----------------------------" << std::endl;
	data->scene->rods_radii = data->full_p;
	data->scene->rods_radii = data->scene->rods_radii.cwiseMax(data->cut_lower_bound);
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

void OptimizationProblem::TestSensitivityGradient(){
	scene->finiteDifferenceEstimation({-0.4,  0.53, 0}, {634576, 181359, 4336.13, 726504, 40214.4, 380732});
}