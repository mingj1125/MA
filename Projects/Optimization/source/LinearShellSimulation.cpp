#include "../include/LinearShell.h"
#include "../include/LinearShell_Stretching.h"
#include "../include/enforce_matrix_constraints.h"
#include <igl/readOBJ.h>
#include <Eigen/Eigen>
#include <iostream>

void LinearShell::initializeScene(const std::string& filename){
    
    MatrixXa V; Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);
    n_nodes = V.rows();
    Vector3a min_corner = V.colwise().minCoeff();
    Vector3a max_corner = V.colwise().maxCoeff();

    // normalize to unit patch
    AScalar length = max_corner(0)-min_corner(0);
    V.rowwise() -= min_corner.transpose();
    V /= length;

    rest_states = V.reshaped<Eigen::RowMajor>();
    faces = F;
    deformed_states = rest_states;
    std::cout << "# Nodes initialized: " << n_nodes << " with dofs: " << rest_states.rows() << std::endl;

    youngsmodulus_each_element.resize(faces.rows());
    youngsmodulus_each_element.setConstant(initial_youngsmodulus);
    std::cout << "# faces initialized: " << faces.rows() << std::endl;

    strain_tensors_each_element.resize(faces.rows());
    stress_tensors_each_element.resize(faces.rows());
}

void LinearShell::resetSimulation(){
    deformed_states = rest_states;
    fixed_vertices = std::vector<int>(0);
}

void LinearShell::applyBoundaryStretch(int i, AScalar strain){

    AScalar strain_apply = 1.5;
    if(strain > 0.) strain_apply = strain;
    switch (i)
    {
    case 5:
        stretchSlidingX(strain_apply);
        break;
    case 4:    
        stretchSlidingY(strain_apply);
        break;
    case 6:
        stretchDiagonal(strain_apply); 
        break;   
    case 2:
        stretchY(strain_apply);
        break;    
    case 1:
        stretchX(strain_apply);
        break;  
    case 3:
        stretchShear(1.5);
        break;          
    default:
        break;
    }
}

void LinearShell::stretchX(AScalar strain){

    resetSimulation();
    AScalar tol = 1e-9;
    int count = 0;

    for(int i = 0; i < n_nodes; ++i){
        Vector3a X = rest_states.segment(3*i, 3);
        if(X(0) < tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i) = X(0)*strain;
            ++count;
        }
        else if(X(0) > 1.0 - tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i) = X(0)*strain;
            ++count;
        }
        fixed_vertices.push_back(i*3+2);
    }
}

void LinearShell::stretchY(AScalar strain){

    resetSimulation();
    AScalar tol = 1e-9;

    for(int i = 0; i < n_nodes; ++i){
        Vector3a X = rest_states.segment(3*i, 3);
        if(X(1) < tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i+1, 0) = X(1,0)*strain;
        }
        else if(X(1) > 1.0 - tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i+1, 0) = X(1,0)*strain;
        }
        fixed_vertices.push_back(i*3+2);
    }
}

void LinearShell::stretchSlidingY(AScalar strain){

    resetSimulation();
    AScalar tol = 1e-9;

    for(int i = 0; i < n_nodes; ++i){
        Vector3a X = rest_states.segment(3*i, 3);
        if(X(1) < tol) {
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i+1, 0) = X(1,0)*strain;
        }
        else if(X(1) > 1.0 - tol) {
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i+1, 0) = X(1,0)*strain;
        }
        fixed_vertices.push_back(i*3+2);
    }
}

void LinearShell::stretchSlidingX(AScalar strain){

    resetSimulation();
    AScalar tol = 1e-9;

    for(int i = 0; i < n_nodes; ++i){
        Vector3a X = rest_states.segment(3*i, 3);
        if(X(0) < tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i, 0) = X(0,0)*strain;
        }
        else if(X(0) > 1.0 - tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i, 0) = X(0,0)*strain;
        }
        fixed_vertices.push_back(i*3+2);
    }
}

void LinearShell::stretchShear(AScalar strain){

    resetSimulation();
    AScalar tol = 1e-9;

    // for(int i = 0; i < n_nodes; ++i){
    //     Vector3a X = rest_states.segment(3*i, 3);
    //     if(X(0) < tol && X(1) < tol || X(1) > 1.0 - tol && X(0) < tol) {
    //         fixed_vertices.push_back(i*3);
    //         fixed_vertices.push_back(i*3+1);
    //         fixed_vertices.push_back(i*3+2);
    //     }
    //     else if(X(0) > 1.0 - tol && X(1) < tol || X(0) > 1.0 - tol && X(1) > 1.0 - tol) {
    //         fixed_vertices.push_back(i*3);
    //         fixed_vertices.push_back(i*3+1);
    //         fixed_vertices.push_back(i*3+2);

    //         deformed_states(3*i+1) = X(1)*strain+X(0)*strain-1;
    //         deformed_states(3*i) = X(0)*strain;
    //     }
    // }
    for(int i = 0; i < n_nodes; ++i){
        Vector3a X = rest_states.segment(3*i, 3);
        if(X(0) < tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);
        }
        else if(X(0) > 1.0 - tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i+1) = X(1)+strain-1;
            deformed_states(3*i) = X(0)+strain-1;
        }
        fixed_vertices.push_back(i*3+2);
    }
}

void LinearShell::stretchDiagonal(AScalar strain){

    resetSimulation();
    AScalar tol = 1e-9;

    for(int i = 0; i < n_nodes; ++i){
        Vector3a X = rest_states.segment(3*i, 3);
        if(X(0) < tol && X(1) < tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);
        }
        else if(X(0) > 1.0 - tol && X(1) < tol || X(1) > 1.0 - tol && X(0) < tol || X(0) > 1.0 - tol && X(1) > 1.0 - tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i+1, 0) = X(1,0)*strain + 0.001;
            deformed_states(3*i, 0) = X(0,0)*strain;
        }
        fixed_vertices.push_back(i*3+2);
    }
}

damped_newton_result LinearShell::Simulate(bool use_log)
{
	Eigen::SparseMatrix<AScalar> damp_matrix(n_nodes*3, n_nodes*3);
    for(int i=0; i<n_nodes*3 ; ++i)
        damp_matrix.coeffRef(i, i) = 1.0;

    LinearShellCostFunction cost_function(this);

    VectorXa& parameters = deformed_states;

    damped_newton_options options;
    options.global_stopping_criteria = global_stopping_criteria;
    options.change_stopping_criteria = 1e-8;
    options.max_iterations = max_iterations;
    options.damp_matrix = damp_matrix;
    options.damping = 1e-2;
    options.solver_type = DN_SOLVER_LLT;
    options.woodbury = false;
    options.sherman_morrison = false;
    options.simplified = false;
    options.use_log = use_log;

    DampedNewtonSolver solver;
    solver.SetParameters(parameters);
    solver.SetCostFunction(&cost_function);
    solver.SetOptions(std::move(options));

    // cost_function.TestHessian(parameters);

    damped_newton_result result = solver.Solve();
    computeStressAndStraininTriangles();

    return result;
}

AScalar LinearShellCostFunction::ComputeEnergy(){
    
    AScalar energy = 0.;
    for(int i = 0; i < data->faces.rows(); ++i){
        Eigen::Vector3i indices = data->faces.row(i);
        Vector9a q;
        Vector6a p;
        for(int j = 0; j < 3; ++j){
            q.segment(j*3,3) = data->deformed_states.segment(indices(j)*3, 3);
            p.segment(j*2,2) = data->rest_states.segment(indices(j)*3, 2);
        }

        AScalar E = data->youngsmodulus_each_element(i);
        AScalar nu = data->nu;
        AScalar lambda = E * nu /((1+nu)*(1-2*nu));
        AScalar mu = E / (2*(1+nu));

        energy += PlanarStVenantKirchhoffEnergyImpl_(q, p, data->thickness, lambda, mu);
    }
    return energy;
}

VectorXa LinearShellCostFunction::ComputeGradient(){

    int n_params = is_xdef ? data->n_nodes*3 : data->youngsmodulus_each_element.rows();

	VectorXa gradient(n_params);
    gradient.setZero();

    for(int i = 0; i < data->faces.rows(); ++i){
        Eigen::Vector3i indices = data->faces.row(i);
        Vector9a q;
        Vector6a p;
        for(int j = 0; j < 3; ++j){
            q.segment(j*3,3) = data->deformed_states.segment(indices(j)*3, 3);
            p.segment(j*2,2) = data->rest_states.segment(indices(j)*3, 2);
        }

        AScalar E = data->youngsmodulus_each_element(i);
        AScalar nu = data->nu;
        AScalar lambda = E * nu /((1+nu)*(1-2*nu));
        AScalar mu = E / (2*(1+nu));

        Vector9a F = PlanarStVenantKirchhoffGradientImpl_(q, p, data->thickness, lambda, mu);
        // std::cout << F.transpose() << std::endl;

        gradient.segment(indices(0)*3, 3) += F.segment(0,3);
        gradient.segment(indices(1)*3, 3) += F.segment(3,3);
        gradient.segment(indices(2)*3, 3) += F.segment(6,3);

    }

    return gradient;
}

Eigen::SparseMatrix<AScalar> LinearShellCostFunction::ComputeHessian(){
    int n_params = is_xdef ? data->n_nodes*3 : data->youngsmodulus_each_element.rows();

	Eigen::SparseMatrix<AScalar> hessian(n_params, n_params);
    hessian.setZero();

    std::vector<Eigen::Triplet<AScalar>> triplets;

    for(int i = 0; i < data->faces.rows(); ++i){
        Eigen::Vector3i indices = data->faces.row(i);
        Vector9a q;
        Vector6a p;
        for(int j = 0; j < 3; ++j){
            q.segment(j*3,3) = data->deformed_states.segment(indices(j)*3, 3);
            p.segment(j*2,2) = data->rest_states.segment(indices(j)*3, 2);
        }

        AScalar E = data->youngsmodulus_each_element(i);
        AScalar nu = data->nu;
        AScalar lambda = E * nu /((1+nu)*(1-2*nu));
        AScalar mu = E / (2*(1+nu));

        Matrix9a J = PlanarStVenantKirchhoffHessianImpl_(q, p, data->thickness, lambda, mu);

        for(int k = 0; k < 3; k++)
                for(int l = 0; l < 3; l++)
                    for(int a = 0; a < 3; a++)
                        for (int j = 0; j < 3; j++){
                            triplets.emplace_back(indices[k]*3+a, indices[l]*3+j, J(k*3 + a, l * 3 + j));
                        }

    }

    hessian.setFromTriplets(triplets.begin(), triplets.end());
    return hessian;
}

cost_evaluation LinearShellCostFunction::Evaluate(const VectorXa& parameters)
{
  if(is_xdef)
    data->deformed_states = parameters;

	if(!data->are_parameters_ok())
		throw std::exception();

	AScalar energy = ComputeEnergy();
    if(std::isnan(energy))
    {
        std::cout << "Energy nan" << std::endl;
        //throw std::exception();
    }

    VectorXa gradient = ComputeGradient();
     for(int i=0; i<gradient.rows(); ++i)
    {
        if(std::isnan(gradient[i]))
        {
            int g_idx = i/3;
            std::cout << "Gradient nan " << i << " " << i/3 << " : " << parameters.segment<3>(g_idx).transpose() << std::endl;
            throw std::exception();
        }
    }

    Eigen::SparseMatrix<AScalar> hessian = ComputeHessian();

    constraints = data->fixed_vertices;

    for(int i=0; i<constraints.size(); ++i)
    	gradient[constraints[i]] = 0.0;

    std::vector<Eigen::Triplet<AScalar>> triplets = SparseMatrixToTriplets(hessian);

    for(int i=0; i<triplets.size(); ++i)
    {
    	if(std::isnan(triplets[i].value()))
        {
            std::cout << triplets[i].row() << " " << triplets[i].col() << " is nan" << std::endl;
    		throw std::exception();
        }
    }

    hessian = EnforceSquareMatrixConstraints(hessian, constraints, true);

    return std::tie(energy, gradient, hessian);
}

void LinearShellCostFunction::PreProcess(VectorXa& parameters){
    data->pre_process(parameters);
}

void LinearShellCostFunction::AcceptStep()
{
	data->accept_step();
}

void LinearShellCostFunction::RejectStep()
{
	data->reject_step();
}

void LinearShellCostFunction::TakeStep(const VectorXa& step, const VectorXa& prev_parameters, VectorXa& new_parameters)
{
	data->take_step(step, prev_parameters, new_parameters);
}

void LinearShellCostFunction::Finalize(const VectorXa& parameters)
{
    if(is_xdef)
      data->deformed_states = parameters;
}

void LinearShell::build_sim_hessian(Eigen::SparseMatrix<AScalar>& K){
    int n_params = n_nodes*3;

	K.resize(n_params, n_params);
    K.setZero();

    std::vector<Eigen::Triplet<AScalar>> triplets;

    for(int i = 0; i < faces.rows(); ++i){
        Eigen::Vector3i indices = faces.row(i);
        Vector9a q;
        Vector6a p;
        for(int j = 0; j < 3; ++j){
            q.segment(j*3,3) = deformed_states.segment(indices(j)*3, 3);
            p.segment(j*2,2) = rest_states.segment(indices(j)*3, 2);
        }

        AScalar E = youngsmodulus_each_element(i);
        AScalar lambda = E * nu /((1.0+nu)*(1.0-2.0*nu));
        AScalar mu = E / (2.0*(1.0+nu));

        Matrix9a J = PlanarStVenantKirchhoffHessianImpl_(q, p, thickness, lambda, mu);

        for(int k = 0; k < 3; k++)
                for(int l = 0; l < 3; l++)
                    for(int a = 0; a < 3; a++)
                        for (int j = 0; j < 3; j++){
                            triplets.emplace_back(indices[k]*3+a, indices[l]*3+j, J(k*3 + a, l * 3 + j));
                        }

    }

    K.setFromTriplets(triplets.begin(), triplets.end());

    for(int i=0; i<triplets.size(); ++i)
    {
    	if(std::isnan(triplets[i].value()))
        {
            std::cout << triplets[i].row() << " " << triplets[i].col() << " is nan" << std::endl;
    		throw std::exception();
        }
    }

    K = EnforceSquareMatrixConstraints(K, fixed_vertices, true);
}

void LinearShell::build_d2Edxp(Eigen::SparseMatrix<AScalar>& K){

	K.resize(n_nodes*3, faces.rows());
    std::vector<Eigen::Triplet<AScalar>> triplets;

    for(int i = 0; i < faces.rows(); ++i){
        Eigen::Vector3i indices = faces.row(i);
        Vector9a q;
        Vector6a p;
        for(int j = 0; j < 3; ++j){
            q.segment(j*3,3) = deformed_states.segment(indices(j)*3, 3);
            p.segment(j*2,2) = rest_states.segment(indices(j)*3, 2);
        }

        // AScalar E = youngsmodulus_each_element(i);
        // AScalar lambda = E * nu /((1+nu)*(1-2*nu));
        // AScalar mu = E / (2*(1+nu));

        AScalar lambda = 1.0 * nu /((1+nu)*(1-2*nu));
        AScalar mu = 1.0 / (2*(1+nu));

        Vector9a F = PlanarStVenantKirchhoffGradientImpl_(q, p, thickness, lambda, mu);

        std::vector<bool> constrained(deformed_states.rows(), false);
	    for(int j=0; j<fixed_vertices.size(); ++j) constrained[fixed_vertices[j]] = true;
        for(int j = 0; j < 3; ++j){
            for(int k = 0; k < 3; ++k){
                if(!constrained[indices(j)*3+k]) triplets.emplace_back(indices(j)*3+k, i, F(3*j+k));
            }
        }
    }
    K.setFromTriplets(triplets.begin(), triplets.end());
}