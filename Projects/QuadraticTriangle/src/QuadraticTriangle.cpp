#include <Eigen/CholmodSupport>
#include <igl/readOBJ.h>
#include <igl/massmatrix.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include "../autodiff/CST3DShell.h"
#include "../autodiff/Quadratic2DShell.h"
#include "../include/QuadraticTriangle.h"

#include <cmath> // for pi in cut directions
#include <random> // for rng in sampling for stress probing
#include <set>
// quadratic node
#include <map> 
#include <algorithm>

void QuadraticTriangle::setMaterialParameter(T& E, T& nu, T& local_lambda, T& local_mu, TV X){
    // nu = 0.45;
    nu = 0.;
    E = 1e6;
    // nu = (1-(face_idx%10)/10.)*nu;
    // E = (1-(face_idx/18)/10.)*E;
    // FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
    // T x = 0.5*(undeformed_vertices.col(1).minCoeff()+undeformed_vertices.col(1).maxCoeff());
    T x = X(1);
    E = (1-x*1.5)*E;
    local_lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);;
    local_mu = E / 2.0 / (1.0 + nu);
}

bool QuadraticTriangle::advanceOneTimeStep()
{
    int iter = 0;
    while (true)
    {
        VectorXT residual = external_force;
        T residual_norm = computeResidual(residual);
        // residual_norms.push_back(residual_norm);
        if (verbose)
            std::cout << "[NEWTON] iter " << iter << "/" 
                << max_newton_iter << ": residual_norm " 
                << residual_norm << " tol: " << newton_tol << std::endl;
        if (residual_norm < newton_tol || iter == max_newton_iter)
        {
            std::cout << "[NEWTON] iter " << iter << "/" 
                << max_newton_iter << ": residual_norm " 
                << residual_norm << " tol: " << newton_tol << std::endl;
            break;
        }
        T du_norm = 1e10;
        du_norm = lineSearchNewton(residual);
        if (du_norm < 1e-10)
            break;
        iter ++;
        
    }

    return true;
}

bool QuadraticTriangle::advanceOneStep(int step)
{
    if (dynamics)
    {
        std::cout << "=================== Time STEP " << step * dt << "s===================" << std::endl;
        bool finished = advanceOneTimeStep();
        // updateDynamicStates();
        // computeStrainAndStressPerElement();
        // computeHomogenization();
        // TM vertices = getFaceVtxUndeformed(10);
        // sample[0] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
        // vertices = getFaceVtxUndeformed(60);
        // sample[1] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
        // visualizeCuts(sample, direction);
        if (step * dt > simulation_duration)
        {
            return true;
        }
        return false;
    }
    else
    {
        std::cout << "===================STEP " << step << "===================" << std::endl;
        VectorXT residual = external_force;
        T residual_norm = computeResidual(residual);
        residual_norms.push_back(residual_norm);
        std::cout << "[NEWTON] iter " << step << "/" 
            << max_newton_iter << ": residual_norm " 
            << residual_norm << " tol: " << newton_tol << std::endl;    
        if (residual_norm < newton_tol || step >= max_newton_iter)
        {   
            computeStrainAndStressPerElement();
            // Matrix<T, 6, 3> vertices = getFaceVtxUndeformed(1288);
            Matrix<T, 3, 3> vertices = getFaceVtxUndeformed(60).block(0,0,3,3);
            sample[1] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
            // TM E = TM::Zero();
            // TV CoM = {-0.21, 0.185, 0};
            // E.block(0,0,2,2) = findBestStrainTensorviaProbing(CoM, direction);
            // std::cout << "sample loc: " << CoM.transpose() << std::endl;
            // std::cout << E << std::endl;
            // std::cout << "stress S: \n" << findBestStressTensorviaProbing(CoM, direction) << std::endl;
            // CoM = {-0.21, 0.30, 0};
            // E.block(0,0,2,2) = findBestStrainTensorviaProbing(CoM, direction);
            // std::cout << "sample loc: " << CoM.transpose() << std::endl;
            // std::cout << E << std::endl;
            // std::cout << "stress S: \n" << findBestStressTensorviaProbing(CoM, direction) << std::endl;
            // CoM = {-0.05, 0.26, 0};
            // E.block(0,0,2,2) = findBestStrainTensorviaProbing(CoM, direction);
            // std::cout << "sample loc: " << CoM.transpose() << std::endl;
            // std::cout << E << std::endl;
            // std::cout << "stress S: \n" << findBestStressTensorviaProbing(CoM, direction) << std::endl;
            // sample[0] = CoM;

            // coarse mesh 
            // testStressTensors(0, 45);
            // testSharedEdgeStress(44, 45, 24, 35);
            // testSharedEdgeStress(45, 42, 24, 34);
            // testSharedEdgeStress(45, 62, 34, 35);

            // refined mesh
            // testStressTensors(0, 679);
            // testSharedEdgeStress(677, 679, 537, 640);
            // testSharedEdgeStress(678, 679, 537, 641);
            // testSharedEdgeStress(676, 679, 640, 641);
            return true;
        }

        T du_norm = 1e10;
        du_norm = lineSearchNewton(residual);
        return false;
    }
}

bool QuadraticTriangle::linearSolve(StiffnessMatrix& K, const VectorXT& residual, VectorXT& du)
{
    START_TIMING(LinearSolve)
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    
    T alpha = 1e-6;
    if (!dynamics)
    {
        StiffnessMatrix H(K.rows(), K.cols());
        H.setIdentity(); H.diagonal().array() = 1e-10;
        K += H;
    }
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;

    // std::cout << K << std::endl;
    
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    int i = 0;
    T dot_dx_g = 0.0;
    for (; i < 50; i++)
    {
        solver.factorize(K);
        // std::cout << "factorize" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        du = solver.solve(residual);
        
        dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        if (!search_dir_correct_sign)
        {   
            invalid_search_dir_cnt++;
        }
        
        // bool solve_success = true;
        // bool solve_success = (K * du - residual).norm() / residual.norm() < 1e-6;
        bool solve_success = du.norm() < 1e3;
        
        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            
            if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K.nonZeros() << std::endl;
                // std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i 
                    << " indefinite " << indefinite_count_reg_cnt 
                    << " invalid search dir " << invalid_search_dir_cnt
                    << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                std::cout << "\t======================== " << std::endl;
                FINISH_TIMING_PRINT(LinearSolve)
            }
            return true;
        }
        else
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    if (verbose)
    {
        std::cout << "\t===== Linear Solve ===== " << std::endl;
        std::cout << "\tnnz: " << K.nonZeros() << std::endl;
        // std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
        std::cout << "\t# regularization step " << i 
            << " indefinite " << indefinite_count_reg_cnt 
            << " invalid search dir " << invalid_search_dir_cnt
            << " invalid solve " << invalid_residual_cnt << std::endl;
        std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
        std::cout << "\t======================== " << std::endl;
        FINISH_TIMING_PRINT(LinearSolve)
    }
    return false;
}

T QuadraticTriangle::computeTotalEnergy()
{
    deformed = undeformed + u;

    T energy = 0.0;
    addShellEnergy(energy);
    if (add_gravity)
        addShellGravitionEnergy(energy);
    if (dynamics)
        addInertialEnergy(energy);
    energy -= u.dot(external_force);
    return energy;
}

T QuadraticTriangle::computeResidual(VectorXT& residual)
{
    deformed = undeformed + u;
    addShellForceEntry(residual);
    if (add_gravity)
        addShellGravitionForceEntry(residual);
    if (dynamics)
        addInertialForceEntry(residual);

    iterateDirichletDoF([&](int offset, T target)
    {
        residual[offset] = target;
        // interesting function of setting Dirichlet BC
        // residual[offset] = 0;
    });

    return residual.norm();
}

void QuadraticTriangle::buildSystemMatrix(StiffnessMatrix& K)
{
    deformed = undeformed + u;
    std::vector<Entry> entries;
    addShellHessianEntries(entries);
    if (add_gravity)
        addShellGravitionHessianEntry(entries);
    if (dynamics)
        addInertialHessianEntries(entries); 
  
    int n_dof = deformed.rows();
    K.resize(n_dof, n_dof);
    K.setFromTriplets(entries.begin(), entries.end());
    projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();
}

void QuadraticTriangle::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }
}

T QuadraticTriangle::lineSearchNewton(const VectorXT& residual)
{
    VectorXT du = residual;
    du.setZero();

    du = residual;
    StiffnessMatrix K(residual.rows(), residual.rows());
    buildSystemMatrix(K);
    bool success = linearSolve(K, residual, du);
    if (!success)
    {
        std::cout << "Linear Solve Failed" << std::endl;
        return 1e16;
    }

    T norm = du.norm();
    if (verbose)
        std::cout << "\t|du | " << norm << std::endl;
    
    T E0 = computeTotalEnergy();
    // std::cout << "obj: " << E0 << std::endl;
    T alpha = 1.0;
    int cnt = 0;
    VectorXT u_current = u;
    while (true)
    {
        u = u_current + alpha * du;
        T E1 = computeTotalEnergy();
        if (E1 - E0 < 0 || cnt > 10)
        {
            // if (cnt > 10)
                // std::cout << "cnt > 10" << " |du| " << norm << " |g| " << residual.norm() << std::endl;
            break;
        }
        alpha *= 0.5;
        cnt += 1;
    }

    return alpha * du.norm();
}

void QuadraticTriangle::initializeFromFile(const std::string& filename)
{
    MatrixXT V; MatrixXi F;
    igl::readOBJ(filename, V, F);

    TV min_corner = V.colwise().minCoeff();
    TV max_corner = V.colwise().maxCoeff();

    T bb_diag = (max_corner - min_corner).norm();

    V *= 1.0 / bb_diag;

    V *= 0.5;

    faces.resize(F.rows(), 6);
    mesh_nodes = V.rows();
    std::map<std::pair<int, int>, int> edgeMidpointIndex;
    std::vector<TV> extra_nodes;
    for(int i = 0; i < F.rows(); ++i){
        int v0 = F(i,0);
        int v1 = F(i,1);
        int v2 = F(i,2);
        faces(i, 0) = v0;
        faces(i, 1) = v1;
        faces(i, 2) = v2;

        std::pair<int, int> edge01 = std::minmax(v0, v1);
        std::pair<int, int> edge12 = std::minmax(v1, v2);
        std::pair<int, int> edge20 = std::minmax(v2, v0);

        TV mid01 = 0.5*(V.row(v0) + V.row(v1));
        TV mid12 = 0.5*(V.row(v1) + V.row(v2));
        TV mid02 = 0.5*(V.row(v0) + V.row(v2));

        if (edgeMidpointIndex.find(edge01) == edgeMidpointIndex.end()) {
            edgeMidpointIndex[edge01] = mesh_nodes+extra_nodes.size();
            extra_nodes.push_back(mid01);
        }
        if (edgeMidpointIndex.find(edge12) == edgeMidpointIndex.end()) {
            edgeMidpointIndex[edge12] = mesh_nodes+extra_nodes.size();
            extra_nodes.push_back(mid12);
        }
        if (edgeMidpointIndex.find(edge20) == edgeMidpointIndex.end()) {
            edgeMidpointIndex[edge20] = mesh_nodes+extra_nodes.size();
            extra_nodes.push_back(mid02);
        }

        faces(i, 3) = edgeMidpointIndex[edge01];
        faces(i, 4) = edgeMidpointIndex[edge20];
        faces(i, 5) = edgeMidpointIndex[edge12];
    }
    MatrixXT V_1(mesh_nodes+extra_nodes.size(), 3);
    for(int i = 0; i < extra_nodes.size(); ++i){
        V_1.row(mesh_nodes+i) = extra_nodes[i];
    }
    V_1.block(0,0,mesh_nodes,3) = V;

    iglMatrixFatten<T, 3>(V_1, undeformed);

    deformed = undeformed;
    u = VectorXT::Zero(deformed.rows());
    stress_tensors = std::vector<TM>(F.rows());
    cauchy_stress_tensors = std::vector<TM>(F.rows());

    strain_tensors = std::vector<TM>(F.rows());
    space_strain_tensors = std::vector<TM2>(F.rows()); 
    defomation_gradients = std::vector<TM>(F.rows()); 

    cut_coloring = VectorXi::Zero(F.rows());
    kernel_coloring_prob = VectorXT::Zero(F.rows());
    kernel_coloring_avg = VectorXT::Zero(F.rows());
    sample = std::vector<TV>(2);
    setProbingLineDirections(8);

    external_force = VectorXT::Zero(deformed.rows());

    // natural BC (traction)
    // for (int j = 0; j < 10; j++)
    // {
    //     external_force[j * 3 + 1] = -5;
    // }
    // for (int j = 90; j < 100; j++)
    // {
    //     external_force[j * 3 + 1] = 5;
    // }
    // for (int j = 0; j < undeformed.size()/3; j++)
    // {
    //     if(undeformed(j*3+2) <= 0 && undeformed(j*3+1) >= V.colwise().maxCoeff()(1)-1e-5){
    //         // for (int d = 0; d < 3; d++)
    //         {
    //             external_force[j * 3 + 1] = 35;
    //         }
    //     }
    //     if(undeformed(j*3+2) <= 0 && undeformed(j*3+1) <= V.colwise().minCoeff()(1)+1e-5){
    //         // for (int d = 0; d < 3; d++)
    //         {
    //             external_force[j * 3 + 1] = -35;
    //         }
    //     }
    // }
    // or single node natural BC
    // external_force[5 * 3 + 2] = -10.0;

    // Essential BC (displacement)

    T shell_len = max_corner(1) - min_corner(1);
    T displacement = -0.01*shell_len;

    for (int j = 0; j < undeformed.size()/3; j++)
    {
        if(undeformed(j*3+2) <= 0 && undeformed(j*3+1) >= V.colwise().maxCoeff()(1)-1e-5){
            for (int d = 0; d < 3; d++)
            {   
                // if(d == 0) continue;
                dirichlet_data[j * 3 + d] = 0.;
            }
        }
    }
    setEssentialBoundaryCondition(0, displacement);

    nu_visualization = VectorXT::Zero(F.rows());
    E_visualization = VectorXT::Zero(F.rows());

    // dynamics = true;
    dynamics = false;
    add_gravity = false;
    use_consistent_mass_matrix = true;
    // E = 0.0;
    dt = 1e-2;
    simulation_duration = 1000000;
    
    if (dynamics)
    {
        initializeDynamicStates();
    }
    
}

void QuadraticTriangle::addShellInplaneEnergy(T& energy)
{
    iterateFaceSerial([&](int face_idx)
    {
        Matrix<T, 6, 3> vertices = getFaceVtxDeformed(face_idx);
        Matrix<T, 6, 3> undeformed_vertices = getFaceVtxUndeformed(face_idx);
        Vector<int, 6> indices = faces.row(face_idx);

        energy += compute2DQuadraticShellEnergy(vertices, undeformed_vertices);
    });
}

void QuadraticTriangle::addShellEnergy(T& energy)
{
    T in_plane_energy = 0.0;
    addShellInplaneEnergy(in_plane_energy);

    energy += in_plane_energy;
}

void QuadraticTriangle::addShellInplaneForceEntries(VectorXT& residual)
{
    iterateFaceSerial([&](int face_idx)
    {
        Matrix<T, 6, 3> vertices = getFaceVtxDeformed(face_idx);
        Matrix<T, 6, 3> undeformed_vertices = getFaceVtxUndeformed(face_idx);
        Vector<int, 6> indices = faces.row(face_idx);
       
        Vector<T, 18> dedx;
        dedx = compute2DQuadraticShellEnergyGradient(vertices, undeformed_vertices);
        T a, b;
        if(heterogenuous) setMaterialParameter(E, nu, a, b, triangleCenterofMass(undeformed_vertices.block(0,0,3,3)));
        nu_visualization[face_idx] = nu;
        E_visualization[face_idx] = E;
        
        addForceEntry<3>(residual, {indices[0],indices[1],indices[2],indices[3],indices[4],indices[5]}, -dedx);
    });
}

void QuadraticTriangle::addShellForceEntry(VectorXT& residual)
{
    addShellInplaneForceEntries(residual);
}

void QuadraticTriangle::addShellInplaneHessianEntries(std::vector<Entry>& entries)
{
    iterateFaceSerial([&](int face_idx)
    {   
        Matrix<T, 6, 3> vertices = getFaceVtxDeformed(face_idx);
        Matrix<T, 6, 3> undeformed_vertices = getFaceVtxUndeformed(face_idx);
        Vector<int, 6> indices = faces.row(face_idx);

        Matrix<T, 18, 18> hessian;
        hessian = compute2DQuadraticShellEnergyHessian(vertices, undeformed_vertices);
        // if(face_idx == 0) std::cout << hessian << "\n and \n" << compute2DQuadraticShellEnergyHessian(nu, E, thickness, x0, x1, x2, X0, X1, X2) << std::endl;

        addHessianEntry<3, 3>(entries, {indices[0],indices[1],indices[2],indices[3],indices[4],indices[5] }, hessian);

    });
}

void QuadraticTriangle::addShellHessianEntries(std::vector<Entry>& entries)
{
    addShellInplaneHessianEntries(entries);
}

void QuadraticTriangle::computeBoundingBox(TV& min_corner, TV& max_corner)
{
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);
    int num_nodes = deformed.rows() / 3;
    for (int i = 0; i < num_nodes; i++)
    {
        for (int d = 0; d < 3; d++)
        {
            max_corner[d] = std::max(max_corner[d], undeformed[i * 3 + d]);
            min_corner[d] = std::min(min_corner[d], undeformed[i * 3 + d]);
        }
    }
}

void QuadraticTriangle::addShellGravitionEnergy(T& energy)
{
    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx).block(0,0,3,3);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);

        FaceIdx indices = faces.row(face_idx).segment(0,3);

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        energy += compute3DCSTGravitationalEnergy(density, thickness, 
            gravity, x0, x1, x2, X0, X1, X2);

        
    });
}

void QuadraticTriangle::addShellGravitionForceEntry(VectorXT& residual)
{
    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx).block(0,0,3,3);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);

        FaceIdx indices = faces.row(face_idx).segment(0,3);

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        Vector<T, 9> dedx;
        compute3DCSTGravitationalEnergyGradient(density, thickness, gravity, x0, x1, x2, X0, X1, X2, dedx);

        addForceEntry<3>(residual, {indices[0], indices[1], indices[2]}, -dedx);
    });
}

void QuadraticTriangle::addShellGravitionHessianEntry(std::vector<Entry>& entries)
{
    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx).block(0,0,3,3);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);

        FaceIdx indices =  faces.row(face_idx).segment(0,3);

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        
        Matrix<T, 9, 9> hessian;
        compute3DCSTGravitationalEnergyHessian(density, thickness, gravity, x0, x1, x2, X0, X1, X2, hessian);
    });
}

// ============================= Dynamics =============================
void QuadraticTriangle::addInertialEnergy(T& energy)
{
    T kinetic_energy = 0.0;
    if (use_consistent_mass_matrix)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx).block(0,0,3,3);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);
            FaceIdx indices = faces.row(face_idx).segment(0,3);
            Matrix<T, 9, 9> mass_mat;
            computeConsistentMassMatrix(undeformed_vertices, mass_mat);
            Vector<T, 9> x_n_plus_1_vec, xn_vec, vn_vec;
            for (int i = 0; i < 3; i++)
            {
                x_n_plus_1_vec.segment<3>(i * 3) = vertices.row(i);
                xn_vec.segment<3>(i * 3) = xn.segment<3>(indices[i] * 3);
                vn_vec.segment<3>(i * 3) = vn.segment<3>(indices[i] * 3);
            }

            T xTMx = x_n_plus_1_vec.transpose() * mass_mat * x_n_plus_1_vec;
            T xTMxn_vn_dt = 2.0 * x_n_plus_1_vec.transpose() * mass_mat * (xn_vec + vn_vec * dt);
            kinetic_energy += (xTMx - xTMxn_vn_dt) / (2.0 * std::pow(dt, 2));
            
        });
    }
    else
    {
        for (int i = 0; i < deformed.rows() / 3; i++)
        {
            TV x_n_plus_1 = deformed.segment<3>(i * 3);
            kinetic_energy += (density * mass_diagonal[i] * (x_n_plus_1.dot(x_n_plus_1)
                                                    - 2.0 * x_n_plus_1.dot(xn.segment<3>(i * 3) + vn.segment<3>(i * 3) * dt)
                                                    )) / (2.0 * std::pow(dt, 2));
        }
    }
    
    energy += kinetic_energy;
}

void QuadraticTriangle::addInertialForceEntry(VectorXT& residual)
{
    if (use_consistent_mass_matrix)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx).block(0,0,3,3);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);
            FaceIdx indices =  faces.row(face_idx).segment(0,3);
            Matrix<T, 9, 9> mass_mat;
            computeConsistentMassMatrix(undeformed_vertices, mass_mat);
            Vector<T, 9> x_n_plus_1_vec, xn_vec, vn_vec;
            for (int i = 0; i < 3; i++)
            {
                x_n_plus_1_vec.segment<3>(i * 3) = vertices.row(i);
                xn_vec.segment<3>(i * 3) = xn.segment<3>(indices[i] * 3);
                vn_vec.segment<3>(i * 3) = vn.segment<3>(indices[i] * 3);
            }
            Vector<T, 9> dedx = mass_mat * (2.0 * x_n_plus_1_vec - 2.0 * (xn_vec + vn_vec * dt)) / (2.0 * std::pow(dt, 2));
            addForceEntry<3>(residual, {indices[0], indices[1], indices[2]}, -dedx);
        });
    }
    else
    {
        for (int i = 0; i < deformed.rows() / 3; i++)
        {
            TV x_n_plus_1 = deformed.segment<3>(i * 3);
            residual.segment<3>(i * 3) -= (density * mass_diagonal[i] * (2.0 * x_n_plus_1
                                                    - 2.0 * (xn.segment<3>(i * 3) + vn.segment<3>(i * 3) * dt)
                                                    )) / (2.0 * std::pow(dt, 2));
        }
    }
}

void QuadraticTriangle::addInertialHessianEntries(std::vector<Entry>& entries)
{
    if (use_consistent_mass_matrix)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx).block(0,0,3,3);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);
            Matrix<T, 9, 9> mass_mat;
            computeConsistentMassMatrix(undeformed_vertices, mass_mat);
            FaceIdx indices = faces.row(face_idx).segment(0,3);
            addHessianEntry<3, 3>(entries, {indices[0], indices[1], indices[2]}, mass_mat / std::pow(dt, 2));
        });
    }
    else
    {
        for (int i = 0; i < deformed.rows() / 3; i++)
        {
            TM hess = density * TM::Identity() * mass_diagonal[i] / std::pow(dt, 2);
            addHessianEntry<3, 3>(entries, {i}, hess);
        }
    }
}

void QuadraticTriangle::updateDynamicStates()
{
    vn = (deformed - xn) / dt;
    xn = deformed;
}

void QuadraticTriangle::initializeDynamicStates()
{
    if (!use_consistent_mass_matrix)
        computeMassMatrix();
    xn = undeformed;
    vn = VectorXT::Zero(undeformed.rows());
}

void QuadraticTriangle::computeMassMatrix()
{
    MatrixXT V; MatrixXi F;
    vectorToIGLMatrix<T, 3>(undeformed, V);
    vectorToIGLMatrix<int, 3>(faces, F);
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
    mass_diagonal.resize(deformed.rows() / 3);
    mass_diagonal.setZero();
    mass_diagonal = M.diagonal();   
}

void QuadraticTriangle::computeConsistentMassMatrix(const FaceVtx& vtx_pos, Matrix<T, 9, 9>& mass_mat)
{
    double m[81];
    T t1 = vtx_pos(0, 0) - vtx_pos(2, 0);
    T t2 = vtx_pos(0, 1) - vtx_pos(2, 1);
    T t3 = vtx_pos(0, 2) - vtx_pos(2, 2);
    T t4 = vtx_pos(1, 0) - vtx_pos(2, 0);
    T t5 = vtx_pos(1, 1) - vtx_pos(2, 1);
    T t6 = vtx_pos(1, 2) - vtx_pos(2, 2);
    T t7 = t1 * t4 + t2 * t5 + t3 * t6;
    t1 = (pow(t1, 0.2e1) + pow(t2, 0.2e1) + pow(t3, 0.2e1)) * (pow(t4, 0.2e1) + pow(t5, 0.2e1) + pow(t6, 0.2e1)) - pow(t7, 0.2e1);
    t1 = sqrt(t1);
    t2 = t1 / 0.12e2;
    t1 = t1 / 0.24e2;
    m[0] = t2;
    m[1] = 0;
    m[2] = 0;
    m[3] = t1;
    m[4] = 0;
    m[5] = 0;
    m[6] = t1;
    m[7] = 0;
    m[8] = 0;
    m[9] = 0;
    m[10] = t2;
    m[11] = 0;
    m[12] = 0;
    m[13] = t1;
    m[14] = 0;
    m[15] = 0;
    m[16] = t1;
    m[17] = 0;
    m[18] = 0;
    m[19] = 0;
    m[20] = t2;
    m[21] = 0;
    m[22] = 0;
    m[23] = t1;
    m[24] = 0;
    m[25] = 0;
    m[26] = t1;
    m[27] = t1;
    m[28] = 0;
    m[29] = 0;
    m[30] = t2;
    m[31] = 0;
    m[32] = 0;
    m[33] = t1;
    m[34] = 0;
    m[35] = 0;
    m[36] = 0;
    m[37] = t1;
    m[38] = 0;
    m[39] = 0;
    m[40] = t2;
    m[41] = 0;
    m[42] = 0;
    m[43] = t1;
    m[44] = 0;
    m[45] = 0;
    m[46] = 0;
    m[47] = t1;
    m[48] = 0;
    m[49] = 0;
    m[50] = t2;
    m[51] = 0;
    m[52] = 0;
    m[53] = t1;
    m[54] = t1;
    m[55] = 0;
    m[56] = 0;
    m[57] = t1;
    m[58] = 0;
    m[59] = 0;
    m[60] = t2;
    m[61] = 0;
    m[62] = 0;
    m[63] = 0;
    m[64] = t1;
    m[65] = 0;
    m[66] = 0;
    m[67] = t1;
    m[68] = 0;
    m[69] = 0;
    m[70] = t2;
    m[71] = 0;
    m[72] = 0;
    m[73] = 0;
    m[74] = t1;
    m[75] = 0;
    m[76] = 0;
    m[77] = t1;
    m[78] = 0;
    m[79] = 0;
    m[80] = t2;

    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            mass_mat(i, j) = m[i * 9 + j] * density;
        }
        
    }
    
}
// ============================= Dynamics End =============================

// ============================= Boundary Utilities =================================

// assume for now I only set explicit BC for the lower boundary (node_idx 0-9)
void QuadraticTriangle::setEssentialBoundaryCondition(T displacement_x, T displacement_y){

    for (int j = 0; j < undeformed.size()/3; j++)
    {
        if(undeformed(j*3+2) <= 0 && undeformed(j*3+1) <= 0){
            u[j * 3 + 1] = displacement_y;
            u[j * 3] = displacement_x;
            for (int d = 0; d < 3; d++)
            {   
                // if(d == 0) continue;
                dirichlet_data[j * 3 + d] = 0.;
            }
        }
    }

}

std::vector<Matrix<T, 3, 3>> QuadraticTriangle::returnStrainTensors(int A){

    TM vertices = getFaceVtxUndeformed(A).block(0,0,3,3);
    auto CoM = triangleCenterofMass(vertices);
    TM E = TM::Zero();
    E.block(0,0,2,2) = findBestStrainTensorviaProbing(CoM, direction);
    // T x = triangleCenterofMass(vertices)(1);
    // T e = 1e6;
    // e = (1-x*1.5)*e;
    // std::cout << A << " " << e << " " << findBestStrainTensorviaAveraging(CoM)(1,1) << std::endl;

    return {strain_tensors.at(A), E, findBestStrainTensorviaAveraging(CoM)};
}

std::vector<Matrix<T, 3, 3>> QuadraticTriangle::returnStressTensors(int A){

    TM vertices = getFaceVtxUndeformed(A).block(0,0,3,3);
    auto CoM = triangleCenterofMass(vertices);

    return {stress_tensors.at(A), findBestStressTensorviaProbing(CoM, direction), findBestStressTensorviaAveraging(CoM)};
}

T QuadraticTriangle::areaRatio(int A){
    FaceVtx vertices = getFaceVtxDeformed(A).block(0,0,3,3);
    FaceVtx undeformed_vertices = getFaceVtxUndeformed(A).block(0,0,3,3);

    TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
    TV X0 = undeformed_vertices.row(0); 
    TV X1 = undeformed_vertices.row(1); 
    TV X2 = undeformed_vertices.row(2);

    return ((x1-x0).cross(x2-x0)).norm() / ((X1-X0).cross(X2-X0)).norm();
}

Vector<T, 3> QuadraticTriangle::triangleCenterofMass(FaceVtx vertices){
    TV CoM; CoM << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
    return CoM;
}

int QuadraticTriangle::pointInTriangle(const TV sample_loc){

    for (int i = 0; i < faces.rows(); i++){
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(i).block(0,0,3,3);

        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);
        TM2 X; X.col(0) = (X1-X0).segment(0,2); X.col(1) = (X2-X0).segment(0,2); 
        T denom = X.determinant();
        X.col(0) = (X1-sample_loc).segment(0,2); X.col(1) = (X2-sample_loc).segment(0,2); 
        T alpha = X.determinant()/denom;
        X.col(0) = (X1-X0).segment(0,2); X.col(1) = (sample_loc-X0).segment(0,2); 
        T beta = X.determinant()/denom;
        T gamma = 1-alpha-beta;

        if (alpha >= 0 && beta >= 0 && gamma >= 0) {
            return i;  // Return the index of the containing triangle
        }
    }

    return -1;
}

std::vector<Vector<T, 3>> QuadraticTriangle::pointInDeformedTriangle(){
    std::vector<TV> update;
    for(auto sample_loc: sample){
        update.push_back(pointInDeformedTriangle(sample_loc));
    }

    return update;
}

Vector<T, 3> QuadraticTriangle::pointInDeformedTriangle(const TV sample_loc){

    for (int i = 0; i < faces.rows()/3; i++){
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(i).block(0,0,3,3);

        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);
        TM2 X; X.col(0) = (X1-X0).segment(0,2); X.col(1) = (X2-X0).segment(0,2); 
        T denom = X.determinant();
        X.col(0) = (X1-sample_loc).segment(0,2); X.col(1) = (X2-sample_loc).segment(0,2); 
        T alpha = X.determinant()/denom;
        X.col(0) = (X1-X0).segment(0,2); X.col(1) = (sample_loc-X0).segment(0,2); 
        T beta = X.determinant()/denom;
        T gamma = 1-alpha-beta;

        if (alpha >= 0 && beta >= 0 && gamma >= 0) {
            FaceVtx vertices = getFaceVtxDeformed(i).block(0,0,3,3);
            TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
            return alpha*x0 + gamma*x1 + beta*x2;  
        }
    }

    return TV::Zero();
}