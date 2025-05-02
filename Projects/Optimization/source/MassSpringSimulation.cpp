#include "../include/MassSpring.h"
#include "../include/MassSpring_Stretching.h"
#include <igl/readOBJ.h>
#include <unordered_set>
#include <iostream>
#include <Eigen/Eigen>

// std::vector<Eigen::Triplet<AScalar>> SparseMatrixToTriplets(const Eigen::SparseMatrix<AScalar>& A)
// {
// 	std::vector<Eigen::Triplet<AScalar> > triplets;

// 	for (int k=0; k < A.outerSize(); ++k)
//         for (Eigen::SparseMatrix<AScalar>::InnerIterator it(A,k); it; ++it)
//         	triplets.push_back(Eigen::Triplet<AScalar>(it.row(), it.col(), it.value()));

//     return triplets;
// }

Eigen::SparseMatrix<AScalar> EnforceSquareMatrixConstraints(Eigen::SparseMatrix<AScalar>& old, std::vector<int>& constraints, bool fill_ones = false)
{

	std::vector<Eigen::Triplet<AScalar>> triplets;// = SparseMatrixToTriplets(old);
    for (int k=0; k < old.outerSize(); ++k)
        for (Eigen::SparseMatrix<AScalar>::InnerIterator it(old,k); it; ++it)
        	triplets.push_back(Eigen::Triplet<AScalar>(it.row(), it.col(), it.value()));

	std::vector<Eigen::Triplet<AScalar>> new_triplets;

	std::vector<bool> constrained(old.rows(), false);
	for(int i=0; i<constraints.size(); ++i)
		constrained[constraints[i]] = true;

	for(int i=0; i<triplets.size(); ++i)
	{
		if(!constrained[triplets[i].row()] && !constrained[triplets[i].col()] )
			new_triplets.push_back(triplets[i]);
	}

	if(fill_ones)
	{
		for(int i=0; i<constraints.size(); ++i)
			new_triplets.push_back(Eigen::Triplet<AScalar>(constraints[i], constraints[i], 1.0));
	}

	Eigen::SparseMatrix<AScalar> new_matrix(old.rows(), old.cols());
	new_matrix.setFromTriplets(new_triplets.begin(), new_triplets.end());

	return new_matrix;
}

struct MeshEdge{
    int u_, v_;

    bool operator==(const MeshEdge& other) const {
        return (u_ == other.u_ && v_ == other.v_) || (u_ == other.v_ && v_ == other.u_);
    }
    MeshEdge(int u, int v): u_(u), v_(v){}
};

struct MeshEdgeHash {
    size_t operator()(const MeshEdge& e) const {
        return std::hash<int>()(e.u_) ^ std::hash<int>()(e.v_);
    }
};

void MassSpring::initializeScene(const std::string& filename){

    MatrixXa V; Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);
    n_nodes = V.rows();
    Vector3a min_corner = V.colwise().minCoeff();
    Vector3a max_corner = V.colwise().maxCoeff();

    // normalize to unit patch
    AScalar length = max_corner(0)-min_corner(0);
    V.rowwise() -= min_corner.transpose();
    V /= length;

    std::unordered_set<MeshEdge, MeshEdgeHash> edges;
    for(int i = 0; i < F.rows(); ++i){
        Eigen::Vector<int, 3> face = F.row(i);
        edges.insert(MeshEdge(face(0), face(1)));
        edges.insert(MeshEdge(face(0), face(2)));
        edges.insert(MeshEdge(face(2), face(1)));
    }

    int full_dof_cnt = 0;
    int node_cnt = 0;
    int spring_cnt = 0;
    std::vector<Vector3a> nodal_positions(n_nodes);
    deformed_states.resize(n_nodes*3, 1);

    for(int i = 0; i < V.rows(); ++i){
        Vector3a node_pos = V.row(i); 
        addPoint(nodal_positions, node_pos, full_dof_cnt, node_cnt);
    }
    std::cout << "# Nodes initialized: " << node_cnt << " with dofs: " << full_dof_cnt << std::endl;
    rest_states = deformed_states;

    for(auto edge: edges){
        Vector3a node1 = V.row(edge.u_);
        Vector3a node2 = V.row(edge.v_);
        if(node1(0,0) > node2(0,0)) {
            std::swap(node1, node2);
            std::swap(edge.u_, edge.v_);
        }

        Spring* spring = new Spring(edge.u_, edge.v_, spring_cnt, node1, node2); 
        spring->set_width(initial_width);
        ++spring_cnt;
        springs.push_back(spring);
    }
    spring_widths.resize(spring_cnt); spring_widths.setConstant(initial_width);
    std::cout << "# Spring initialized: " << spring_cnt << std::endl;

    // Vector3a top_right, bottom_left;
    // computeBoundingBox(top_right, bottom_left);
    // std::cout << top_right.transpose() << " bt: " << bottom_left.transpose() << std::endl;

}

void MassSpring::addPoint(std::vector<Vector3a>& existing_nodes, 
    const Vector3a& point, int& full_dof_cnt, int& node_cnt)
{
    deformed_states.segment(full_dof_cnt, 3) = point;
    existing_nodes.push_back(point);
    full_dof_cnt += 3;
    node_cnt++;
}

std::vector<std::array<size_t, 2>> MassSpring::get_edges(){
    std::vector<std::array<size_t, 2>> edges;
    for(auto spring: springs){
        edges.push_back({spring->p1, spring->p2});
    }
    return edges;
}

void MassSpring::resetSimulation(){
    deformed_states = rest_states;
    fixed_vertices = std::vector<int>(0);
}

void MassSpring::computeBoundingBox(Vector3a& top_right, Vector3a& bottom_left){
    
    bottom_left.setConstant(1e6);
    top_right.setConstant(-1e6);
    for(int i = 0; i < n_nodes; ++i){
        Vector3a X = rest_states.segment(3*i, 3);
        bottom_left = bottom_left.cwiseMin(X);
        top_right = top_right.cwiseMax(X);
    }

}

void MassSpring::ApplyBoundaryStretch(int i){

    switch (i)
    {
    case 3:
        stretchX(1.1);
        break;
    case 2:    
        stretchY(1.1);
        break;
    case 1:
        stretchDiagonal(1.1); 
        break;   
    default:
        break;
    }
}

void MassSpring::stretchX(AScalar strain){

    resetSimulation();
    AScalar tol = 1e-9;
    int count = 0;

    for(int i = 0; i < n_nodes; ++i){
        Vector3a X = rest_states.segment(3*i, 3);
        if(X(0) < tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i, 0) = X(0,0)*strain;
            ++count;
        }
        else if(X(0) > 1.0 - tol) {
            fixed_vertices.push_back(i*3);
            fixed_vertices.push_back(i*3+1);
            fixed_vertices.push_back(i*3+2);

            deformed_states(3*i, 0) = X(0,0)*strain;
            ++count;
        }
    }
    // std::cout << count << std::endl;
}

void MassSpring::stretchY(AScalar strain){

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
    }
}

void MassSpring::stretchDiagonal(AScalar strain){

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

            deformed_states(3*i+1, 0) = X(1,0)*strain;
            deformed_states(3*i, 0) = X(0,0)*strain;
        }
    }
}

damped_newton_result MassSpring::Simulate(bool use_log)
{
	Eigen::SparseMatrix<AScalar> damp_matrix(n_nodes*3, n_nodes*3);
    for(int i=0; i<n_nodes*3 ; ++i)
        damp_matrix.coeffRef(i, i) = 1.0;

    MassSpringCostFunction cost_function(this);

    VectorXa& parameters = deformed_states;

    damped_newton_options options;
    options.global_stopping_criteria = global_stopping_criteria;
    options.change_stopping_criteria = 1e-8;
    options.max_iterations = max_iterations;
    options.damp_matrix = damp_matrix;
    // options.damping = 1e-2;
    options.solver_type = DN_SOLVER_LLT;
    options.woodbury = false;
    options.sherman_morrison = false;
    options.simplified = false;
    options.use_log = use_log;

    DampedNewtonSolver solver;
    solver.SetParameters(parameters);
    solver.SetCostFunction(&cost_function);
    solver.SetOptions(std::move(options));

    // //cost_function.TestHessian(parameters);

    damped_newton_result result = solver.Solve();

    return result;
}

AScalar MassSpringCostFunction::ComputeEnergy(){
    
    AScalar energy = 0.;
    for(auto spring: data->springs){
        Vector3a xi, xj, Xi, Xj;
        xi = data->deformed_states.segment(spring->p1*3, 3);
        xj = data->deformed_states.segment(spring->p2*3, 3);
        Xi = data->rest_states.segment(spring->p1*3, 3);
        Xj = data->rest_states.segment(spring->p2*3, 3);

        energy += stretchingEnergyLocal(spring->k_s(), Xi, Xj, xi, xj);
    }
    return energy;
}

VectorXa MassSpringCostFunction::ComputeGradient(){

    int n_params = is_xdef ? data->n_nodes*3 : data->spring_widths.rows();

	VectorXa gradient(n_params);
    gradient.setZero();

    for(auto spring: data->springs){
        Vector3a xi, xj, Xi, Xj;
        xi = data->deformed_states.segment(spring->p1*3, 3);
        xj = data->deformed_states.segment(spring->p2*3, 3);
        Xi = data->rest_states.segment(spring->p1*3, 3);
        Xj = data->rest_states.segment(spring->p2*3, 3);

        Vector12a F;
        F.setZero();
        computeStretchingEnergyGradient(spring->k_s(), Xi, Xj, xi, xj, F);

        gradient.segment(spring->p1*3, 3) += F.segment(0,3);
        gradient.segment(spring->p2*3, 3) += F.segment(3,3);
    }

    return gradient;
}

Eigen::SparseMatrix<AScalar> MassSpringCostFunction::ComputeHessian(){
    int n_params = is_xdef ? data->n_nodes*3 : data->spring_widths.rows();

	Eigen::SparseMatrix<AScalar> hessian(n_params, n_params);
    hessian.setZero();

    std::vector<Eigen::Triplet<AScalar>> triplets;
    for(auto spring: data->springs){
        Vector3a xi, xj, Xi, Xj;
        xi = data->deformed_states.segment(spring->p1*3, 3);
        xj = data->deformed_states.segment(spring->p2*3, 3);
        Xi = data->rest_states.segment(spring->p1*3, 3);
        Xj = data->rest_states.segment(spring->p2*3, 3);
        std::vector<int> offsets = {spring->p1*3, spring->p2*3};

        Matrix12a J;
        computeStretchingEnergyHessian(spring->k_s(), Xi, Xj, xi, xj, J);

        for(int k = 0; k < 2; k++)
                for(int l = 0; l < 2; l++)
                    for(int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++){
                            triplets.emplace_back(offsets[k]+i, offsets[l]+j, J(k*3 + i, l * 3 + j));
                        }
    }
    hessian.setFromTriplets(triplets.begin(), triplets.end());
    return hessian;
}

cost_evaluation MassSpringCostFunction::Evaluate(const VectorXa& parameters)
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

    std::vector<Eigen::Triplet<AScalar>> triplets; // = SparseMatrixToTriplets(hessian);
    for (int k=0; k < hessian.outerSize(); ++k)
        for (Eigen::SparseMatrix<AScalar>::InnerIterator it(hessian,k); it; ++it)
        	triplets.push_back(Eigen::Triplet<AScalar>(it.row(), it.col(), it.value()));

    for(int i=0; i<triplets.size(); ++i)
    {
    	if(std::isnan(triplets[i].value()))
        {
            std::cout << triplets[i].row() << " " << triplets[i].col() << " is nan" << std::endl;
    		throw std::exception();
        }
    }

    hessian = EnforceSquareMatrixConstraints(hessian, constraints);

    return std::tie(energy, gradient, hessian);
}

void MassSpringCostFunction::PreProcess(VectorXa& parameters){
    data->pre_process(parameters);
}

void MassSpringCostFunction::AcceptStep()
{
	data->accept_step();
}

void MassSpringCostFunction::RejectStep()
{
	data->reject_step();
}

void MassSpringCostFunction::TakeStep(const VectorXa& step, const VectorXa& prev_parameters, VectorXa& new_parameters)
{
	data->take_step(step, prev_parameters, new_parameters);
}

void MassSpringCostFunction::Finalize(const VectorXa& parameters)
{
    if(is_xdef)
      data->deformed_states = parameters;
}

void MassSpring::build_d2Edx2(Eigen::SparseMatrix<AScalar>& K){
    int n_params = n_nodes*3;

	K.resize(n_params, n_params);
    K.setZero();

    stretchX(1.001); Simulate(false);

    std::vector<Eigen::Triplet<AScalar>> triplets;
    for(auto spring: springs){
        Vector3a xi, xj, Xi, Xj;
        xi = deformed_states.segment(spring->p1*3, 3);
        xj = deformed_states.segment(spring->p2*3, 3);
        Xi = rest_states.segment(spring->p1*3, 3);
        Xj = rest_states.segment(spring->p2*3, 3);
        std::vector<int> offsets = {spring->p1*3, spring->p2*3};

        Matrix12a J;
        computeStretchingEnergyHessian(spring->k_s(), Xi, Xj, xi, xj, J);

        for(int k = 0; k < 2; k++)
                for(int l = 0; l < 2; l++)
                    for(int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++){
                            triplets.emplace_back(offsets[k]+i, offsets[l]+j, J(k*3 + i, l * 3 + j));
                        }
    }
    K.setFromTriplets(triplets.begin(), triplets.end());

    std::vector<Eigen::Triplet<AScalar>> triplets_k; // = SparseMatrixToTriplets(hessian);
    for (int k=0; k < K.outerSize(); ++k)
        for (Eigen::SparseMatrix<AScalar>::InnerIterator it(K,k); it; ++it)
        	triplets_k.push_back(Eigen::Triplet<AScalar>(it.row(), it.col(), it.value()));

    for(int i=0; i<triplets_k.size(); ++i)
    {
    	if(std::isnan(triplets_k[i].value()))
        {
            std::cout << triplets_k[i].row() << " " << triplets_k[i].col() << " is nan" << std::endl;
    		throw std::exception();
        }
    }

    K = EnforceSquareMatrixConstraints(K, fixed_vertices, true);
}

void MassSpring::build_d2Edxp(Eigen::SparseMatrix<AScalar>& K){

	K.resize(n_nodes*3, spring_widths.rows());
    std::vector<Eigen::Triplet<AScalar>> triplets;

    for(auto spring: springs){
        Vector3a xi, xj, Xi, Xj;
        xi = deformed_states.segment(spring->p1*3, 3);
        xj = deformed_states.segment(spring->p2*3, 3);
        Xi = rest_states.segment(spring->p1*3, 3);
        Xj = rest_states.segment(spring->p2*3, 3);

        Vector12a F;
        F.setZero();
        computeStretchingEnergyGradient(spring->k_s(), Xi, Xj, xi, xj, F);
        F /= spring->width;

        std::vector<bool> constrained(deformed_states.rows(), false);
	    for(int i=0; i<fixed_vertices.size(); ++i) constrained[fixed_vertices[i]] = true;
        for(int i = 0; i < 3; ++i){
            if(!constrained[spring->p1*3+i])
                triplets.emplace_back(spring->p1*3+i, spring->spring_id, F(i));
            if(!constrained[spring->p2*3+i])    
                triplets.emplace_back(spring->p2*3+i, spring->spring_id, F(3+i));
        }
    }
    K.setFromTriplets(triplets.begin(), triplets.end());
}
