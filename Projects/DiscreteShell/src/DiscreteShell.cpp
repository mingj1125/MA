#include <Eigen/CholmodSupport>
#include <igl/readOBJ.h>
#include <igl/massmatrix.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include "../autodiff/CST3DShell.h"
#include "../autodiff/Quadratic2DShell.h"
#include "../include/DiscreteShell.h"

#include <cmath> // for pi in cut directions
#include <random> // for rng in sampling for stress probing
#include <set>

void DiscreteShell::setMaterialParameter(T& E, T& nu, int face_idx){
    // nu = 0.45;
    nu = 0.;
    E = 1e6;
    // nu = (1-(face_idx%10)/10.)*nu;
    // E = (1-(face_idx/18)/10.)*E;
    FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
    // T x = 0.5*(undeformed_vertices.col(1).minCoeff()+undeformed_vertices.col(1).maxCoeff());
    T x = triangleCenterofMass(undeformed_vertices)(1);
    E = (1-x*1.5)*E;
}

bool DiscreteShell::advanceOneTimeStep()
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

bool DiscreteShell::advanceOneStep(int step)
{
    if (dynamics)
    {
        std::cout << "=================== Time STEP " << step * dt << "s===================" << std::endl;
        bool finished = advanceOneTimeStep();
        updateDynamicStates();
        computeStrainAndStressPerElement();
        computeHomogenization();
        TM vertices = getFaceVtxDeformed(10);
        sample[0] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
        vertices = getFaceVtxDeformed(60);
        sample[1] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
        visualizeCuts(sample, direction);
        std::cout << "Found stress tensor: \n" << findBestStressTensorviaProbing(sample[0], direction) << std::endl;
        std::cout << "Caculated stress tensor at sample point triangle: \n" << stress_tensors[10] << std::endl;
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
            computeHomogenization();
            // TM vertices = getFaceVtxUndeformed(1288);
            TM vertices = getFaceVtxUndeformed(60);
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

bool DiscreteShell::linearSolve(StiffnessMatrix& K, const VectorXT& residual, VectorXT& du)
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

T DiscreteShell::computeTotalEnergy()
{
    deformed = undeformed + u;

    T energy = 0.0;
    addShellEnergy(energy);
    if (add_gravity)
        addShellGravitionEnergy(energy);
    if (dynamics)
        addInertialEnergy(energy);
    if (set_window)
        addEnergyforDesiredTarget(energy);    
    energy -= u.dot(external_force);
    return energy;
}

T DiscreteShell::computeResidual(VectorXT& residual)
{
    deformed = undeformed + u;
    addShellForceEntry(residual);
    if (add_gravity)
        addShellGravitionForceEntry(residual);
    if (dynamics)
        addInertialForceEntry(residual);
    if (set_window)
        addGradientForDesiredTarget(residual);     

    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = target;
            // interesting function of setting Dirichlet BC
            // residual[offset] = 0;
        });

    return residual.norm();
}

void DiscreteShell::computeLinearModes(MatrixXT& eigen_vectors, VectorXT& eigen_values)
{
    int nmodes = 10;
    StiffnessMatrix K;
    run_diff_test = true;
    buildSystemMatrix(K);
    run_diff_test = false;
    Spectra::SparseSymShiftSolve<T, Eigen::Lower> op(K);
    
    T shift = -1e-4;
    Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<T, Eigen::Lower>> 
            eigs(op, nmodes, 2 * nmodes, shift);

    eigs.init();

    int nconv = eigs.compute(Spectra::SortRule::LargestMagn);

    if (eigs.info() == Spectra::CompInfo::Successful)
    {
        eigen_vectors = eigs.eigenvectors().real();
        eigen_values = eigs.eigenvalues().real();
    }

    MatrixXT tmp_vec = eigen_vectors;
    VectorXT tmp_val = eigen_values;
    for (int i = 0; i < nmodes; i++)
    {
        eigen_vectors.col(i) = tmp_vec.col(nmodes-i-1);
        eigen_values[i] = tmp_val[nmodes-i-1];
    }
    
}

void DiscreteShell::buildSystemMatrix(StiffnessMatrix& K)
{
    deformed = undeformed + u;
    std::vector<Entry> entries;
    addShellHessianEntries(entries);
    if (add_gravity)
        addShellGravitionHessianEntry(entries);
    if (dynamics)
        addInertialHessianEntries(entries); 
    // if (set_window)
    //     addHessianForDesiredStrain(entries);    
    int n_dof = deformed.rows();
    K.resize(n_dof, n_dof);
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();
}

void DiscreteShell::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }
}

T DiscreteShell::lineSearchNewton(const VectorXT& residual)
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

void DiscreteShell::initializeFromFile(const std::string& filename)
{
    MatrixXT V; MatrixXi F;
    igl::readOBJ(filename, V, F);

    TV min_corner = V.colwise().minCoeff();
    TV max_corner = V.colwise().maxCoeff();

    T bb_diag = (max_corner - min_corner).norm();

    V *= 1.0 / bb_diag;

    V *= 0.5;

    auto rotationMatrixFromEulerAngle = [](T angle_z, T angle_y, T angle_x)
    {
        Eigen::Matrix3d R, yaw, pitch, roll;
        yaw.setZero(); pitch.setZero(); roll.setZero();
        yaw(0, 0) = cos(angle_z);	yaw(0, 1) = -sin(angle_z);
        yaw(1, 0) = sin(angle_z);	yaw(1, 1) = cos(angle_z);
        yaw(2, 2) = 1.0;
        //y rotation
        pitch(0, 0) = cos(angle_y); pitch(0, 2) = sin(angle_y);
        pitch(1, 1) = 1.0;
        pitch(2, 0) = -sin(angle_y); pitch(2, 2) = cos(angle_y);
        //x rotation
        roll(0, 0) = 1.0;
        roll(1, 1) = cos(angle_x); roll(1, 2) = -sin(angle_x);
        roll(2, 1) = sin(angle_x); roll(2, 2) = cos(angle_x);
        R = yaw * pitch * roll;
        return R;
    };

    // TM R = rotationMatrixFromEulerAngle(0, 0, M_PI_2);
    // for (int i = 0; i < V.rows(); i++)
    // {
    //     V.row(i) = (R * V.row(i).transpose()).transpose();
    // }
    

    iglMatrixFatten<T, 3>(V, undeformed);
    iglMatrixFatten<int, 3>(F, faces);
    deformed = undeformed;
    u = VectorXT::Zero(deformed.rows());
    stress_tensors = std::vector<TM>(F.rows());
    cauchy_stress_tensors = std::vector<TM>(F.rows());
    homo_stress_tensors = std::vector<TM>(F.rows());
    homo_stress_magnitudes = std::vector<T>(F.rows());

    strain_tensors = std::vector<TM>(F.rows());
    space_strain_tensors = std::vector<TM2>(F.rows()); 
    optimization_homo_target_tensors = std::vector<TM2>(F.rows()); 
    homo_strain_tensors = std::vector<TM>(F.rows());
    homo_strain_magnitudes = std::vector<T>(F.rows());

    weights << 1e3, 1e3, 1e3, 1e3;
    epsilons << 0.9, 0.1, 0.1, 1.259;

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
    // set_window = true; // for window testing
    window_x = 1;
    window_y = 1;
    window_length = 5;
    window_height = 5;

    T shell_len = max_corner(1) - min_corner(1);
    T displacement = -0.01*shell_len;

    if (!set_window) { // if no window testing required we stretch sheet
   
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
    }

    nu_visualization = VectorXT::Zero(F.rows());
    E_visualization = VectorXT::Zero(F.rows());

    buildHingeStructure();
    // dynamics = true;
    dynamics = false;
    add_gravity = false;
    use_consistent_mass_matrix = true;
    // E = 0.0;
    dt = 1e-2;
    simulation_duration = 1000000;
    
    hinge_stiffness.setConstant(10);
    if (dynamics)
    {
        initializeDynamicStates();
    }
    
}

void DiscreteShell::buildHingeStructure()
{
    struct Hinge
	{
		Hinge()
		{
			for (int i = 0; i < 2; i++)
			{
				edge[i] = -1;
				flaps[i] = -1;
				tris[i] = -1;
			}
		}
		int edge[2];
		int flaps[2];
		int tris[2];
	};
	
	std::vector<Hinge> hinges_temp;
	
	hinges_temp.clear();
	std::map<std::pair<int, int>, int> edge2index;
	for (int i = 0; i < faces.size() / 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int i1 = faces(3 * i + j);
			int i2 = faces(3 * i + (j + 1) % 3);
			int i1t = i1;
			int i2t = i2;
			bool swapped = false;
			if (i1t > i2t)
			{
				std::swap(i1t, i2t);
				swapped = true;
			}
			
			auto ei = std::make_pair(i1t, i2t);
			auto ite = edge2index.find(ei);
			if (ite == edge2index.end())
			{
				//insert new hinge
				edge2index[ei] = hinges_temp.size();
				hinges_temp.push_back(Hinge());
				Hinge& hinge = hinges_temp.back();
				hinge.edge[0] = i1t;
				hinge.edge[1] = i2t;
				int itmp = swapped ? 1 : 0;
				hinge.tris[itmp] = i;
				hinge.flaps[itmp] = faces(3 * i + (j + 2) % 3);
			}
			else
			{
				//hinge for this edge already exists, add missing information for this triangle
				Hinge& hinge = hinges_temp[ite->second];
				int itmp = swapped ? 1 : 0;
				hinge.tris[itmp] = i;
				hinge.flaps[itmp] = faces(3 * i + (j + 2) % 3);
			}
		}
	}
	//ordering for edges
	
	hinges.resize(hinges_temp.size(), Eigen::NoChange);
	int ii = 0;
	/*
      auto diff code takes
           x3
         /   \
        x2---x1
         \   /
           x0	

      hinge is 
           x2
         /   \
        x0---x1
         \   /
           x3	
    */
    for(Hinge & hinge : hinges_temp) {
		if ((hinge.tris[0] == -1) || (hinge.tris[1] == -1)) {
			continue; //skip boundary edges
		}
		hinges(ii, 2) = hinge.edge[0]; //x0
		hinges(ii, 1) = hinge.edge[1]; //x1
		hinges(ii, 3) = hinge.flaps[0]; //x2
		hinges(ii, 0) = hinge.flaps[1]; //x3
		++ii;
	}
	hinges.conservativeResize(ii, Eigen::NoChange);
    hinge_stiffness.resize(hinges.rows());
    hinge_stiffness.setOnes();
}

void DiscreteShell::addShellInplaneEnergy(T& energy)
{
    iterateFaceSerial([&](int face_idx)
    {
        if(heterogenuous) {
            setMaterialParameter(E, nu, face_idx);
        }
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

        FaceIdx indices = faces.segment<3>(face_idx * 3);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        T k_s = E * thickness / (1.0 - nu * nu);
        T lambda = E * nu /((1+nu)*(1-2*nu));
        T mu = E / (2*(1+nu));
        if(quadratic) energy += compute2DQuadraticShellEnergy(lambda, mu, thickness, x0, x1, x2, X0, X1, X2);
        else energy += compute3DCSTShellEnergy(nu, k_s, x0, x1, x2, X0, X1, X2);

        
    });
}

void DiscreteShell::addShellBendingEnergy(T& energy)
{
    iterateHingeSerial([&](const HingeIdx& hinge_idx, int hinge_cnt){

        HingeVtx deformed_vertices = getHingeVtxDeformed(hinge_idx);
        HingeVtx undeformed_vertices = getHingeVtxUndeformed(hinge_idx);


        T k_bend = hinge_stiffness[hinge_cnt] * E * std::pow(thickness, 3) / (24 * (1.0 - std::pow(nu, 2)));

        TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

        energy += computeDSBendingEnergy(k_bend, x0, x1, x2, x3, X0, X1, X2, X3);

    });
}

void DiscreteShell::addShellEnergy(T& energy)
{
    T in_plane_energy = 0.0;
    T bending_energy = 0.0;
    addShellInplaneEnergy(in_plane_energy);
    addShellBendingEnergy(bending_energy);    

    energy += in_plane_energy;
    energy += bending_energy;
}

void DiscreteShell::addShellInplaneForceEntries(VectorXT& residual)
{
    iterateFaceSerial([&](int face_idx)
    {
        if(heterogenuous) {
            setMaterialParameter(E, nu, face_idx);
        }
        nu_visualization[face_idx] = nu;
        E_visualization[face_idx] = E;
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
        FaceIdx indices = faces.segment<3>(face_idx * 3);

        T k_s = E * thickness / (1.0 - nu * nu);
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);
        
        Vector<T, 9> dedx;
        T lambda = E * nu /((1+nu)*(1-2*nu));
        T mu = E / (2*(1+nu));
        if(quadratic) dedx = compute2DQuadraticShellEnergyGradient(lambda, mu, thickness, x0, x1, x2, X0, X1, X2);
        else compute3DCSTShellEnergyGradient(nu, k_s, x0, x1, x2, X0, X1, X2, dedx);
        
        addForceEntry<3>(residual, {indices[0], indices[1], indices[2]}, -dedx);
    });
}

void DiscreteShell::addShellBendingForceEntries(VectorXT& residual)
{
    iterateHingeSerial([&](const HingeIdx& hinge_idx, int hinge_cnt){
                
        HingeVtx deformed_vertices = getHingeVtxDeformed(hinge_idx);
        HingeVtx undeformed_vertices = getHingeVtxUndeformed(hinge_idx);


        T k_bend = hinge_stiffness[hinge_cnt] * E * std::pow(thickness, 3) / (24 * (1.0 - std::pow(nu, 2)));

        Vector<T, 12> dedx;
        
        TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

        computeDSBendingEnergyGradient(k_bend, x0, x1, x2, x3, X0, X1, X2, X3, dedx);
        addForceEntry<3>(residual, 
            {hinge_idx[0], hinge_idx[1], hinge_idx[2], hinge_idx[3]}, -dedx);
    });
}
void DiscreteShell::addShellForceEntry(VectorXT& residual)
{
    addShellInplaneForceEntries(residual);
    addShellBendingForceEntries(residual);
}

void DiscreteShell::addShellInplaneHessianEntries(std::vector<Entry>& entries)
{
    iterateFaceSerial([&](int face_idx)
    {   
        if(heterogenuous) {
            setMaterialParameter(E, nu, face_idx);
        }
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

        FaceIdx indices = faces.segment<3>(face_idx * 3);

        T k_s = E * thickness / (1.0 - nu * nu);

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);

        Matrix<T, 9, 9> hessian;
        T lambda = E * nu /((1+nu)*(1-2*nu));
        T mu = E / (2*(1+nu));
        if(quadratic) hessian = compute2DQuadraticShellEnergyHessian(lambda, mu, thickness, x0, x1, x2, X0, X1, X2);
        else compute3DCSTShellEnergyHessian(nu, k_s, x0, x1, x2, X0, X1, X2, hessian);
        // if(face_idx == 0) std::cout << hessian << "\n and \n" << compute2DQuadraticShellEnergyHessian(nu, E, thickness, x0, x1, x2, X0, X1, X2) << std::endl;


        addHessianEntry<3, 3>(entries, {indices[0], indices[1], indices[2]}, hessian);

    });
}

void DiscreteShell::addShellBendingHessianEntries(std::vector<Entry>& entries)
{
    iterateHingeSerial([&](const HingeIdx& hinge_idx, int hinge_cnt){
        
        HingeVtx deformed_vertices = getHingeVtxDeformed(hinge_idx);
        HingeVtx undeformed_vertices = getHingeVtxUndeformed(hinge_idx);

        T k_bend = hinge_stiffness[hinge_cnt] * E * std::pow(thickness, 3) / (24 * (1.0 - std::pow(nu, 2)));
        
        Matrix<T, 12, 12> hess;
        TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

        computeDSBendingEnergyHessian(k_bend, x0, x1, x2, x3, X0, X1, X2, X3, hess);
        addHessianEntry<3, 3>(entries, 
                            {hinge_idx[0], hinge_idx[1], hinge_idx[2], hinge_idx[3]}, 
                            hess);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

    });
}

void DiscreteShell::addShellHessianEntries(std::vector<Entry>& entries)
{
    addShellInplaneHessianEntries(entries);
    addShellBendingHessianEntries(entries);
}

void DiscreteShell::setHingeStiffness()
{
    int dir = 1;
    T eps = 0.02;
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    TV center = 0.5 * (min_corner + max_corner);
    iterateHingeSerial([&](const HingeIdx& hinge_idx, int hinge_cnt){
        
        HingeVtx undeformed_vertices = getHingeVtxUndeformed(hinge_idx);
        TV v0 = undeformed_vertices.row(1);
        TV v1 = undeformed_vertices.row(2);
        bool center_v0 = v0[dir] < center[dir] + eps && v0[dir] > center[dir] - eps;
        bool center_v1 = v1[dir] < center[dir] + eps && v1[dir] > center[dir] - eps;
        if (center_v0 && center_v1)
        {
            hinge_stiffness[hinge_cnt] = E;
            std::cout << "center" << std::endl;
        }
    });
}

void DiscreteShell::computeBoundingBox(TV& min_corner, TV& max_corner)
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

void DiscreteShell::addShellGravitionEnergy(T& energy)
{
    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

        FaceIdx indices = faces.segment<3>(face_idx * 3);

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        energy += compute3DCSTGravitationalEnergy(density, thickness, 
            gravity, x0, x1, x2, X0, X1, X2);

        
    });
}

void DiscreteShell::addShellGravitionForceEntry(VectorXT& residual)
{
    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

        FaceIdx indices = faces.segment<3>(face_idx * 3);

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        Vector<T, 9> dedx;
        compute3DCSTGravitationalEnergyGradient(density, thickness, gravity, x0, x1, x2, X0, X1, X2, dedx);

        addForceEntry<3>(residual, {indices[0], indices[1], indices[2]}, -dedx);
    });
}

void DiscreteShell::addShellGravitionHessianEntry(std::vector<Entry>& entries)
{
    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

        FaceIdx indices = faces.segment<3>(face_idx * 3);

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        
        Matrix<T, 9, 9> hessian;
        compute3DCSTGravitationalEnergyHessian(density, thickness, gravity, x0, x1, x2, X0, X1, X2, hessian);
    });
}

// ============================= DERIVATIVE TESTS =============================

void DiscreteShell::checkTotalGradient(bool perturb)
{
    run_diff_test = true;

    int n_dof = deformed.rows();

    if (perturb)
    {
        VectorXT du(n_dof);
        du.setRandom();
        du *= 1.0 / du.norm();
        du *= 0.001;
        u += du;
    }

    std::cout << "======================== CHECK GRADIENT ========================" << std::endl;
    std::cout << "****** Only mismatching entries are printed ******" << std::endl;
    
    T epsilon = 1e-6;
    VectorXT gradient(n_dof);
    gradient.setZero();

    computeResidual(gradient);

    // std::cout << gradient.transpose() << std::endl;
    
    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();

    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        u(dof_i) += epsilon;
        // std::cout << W * dq << std::endl;
        T E0 = computeTotalEnergy();
        
        u(dof_i) -= 2.0 * epsilon;
        T E1 = computeTotalEnergy();
        u(dof_i) += epsilon;
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E1 - E0) / (2.0 *epsilon);
        if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            continue;
        if (std::abs( gradient_FD(dof_i) - gradient(dof_i)) < 1e-3 * std::abs(gradient(dof_i)))
            continue;
        std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << gradient(dof_i) << std::endl;
        std::getchar();
        cnt++;   
    }
    std::cout << "Gradient Diff Test Finished" << std::endl;
    run_diff_test = false;
}

void DiscreteShell::checkTotalGradientScale(bool perturb)
{
    
    run_diff_test = true;
    
    std::cout << "===================== Check Gradient Scale =====================" << std::endl;
    std::cout << "********************You Should Be Seeing 4s********************" << std::endl;
    
    int n_dof = deformed.rows();

    if (perturb)
    {
        VectorXT du(n_dof);
        du.setRandom();
        du *= 1.0 / du.norm();
        du *= 0.001;
        u += du;
    }

    VectorXT gradient(n_dof);
    gradient.setZero();
    computeResidual(gradient);
    gradient *= -1;
    T E0 = computeTotalEnergy();
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.01;
    T previous = 0.0;
    VectorXT u_backup = u;
    for (int i = 0; i < 10; i++)
    {
        u = u_backup + dx;
        T E1 = computeTotalEnergy();
        T dE = E1 - E0;
        
        dE -= gradient.dot(dx);
        // std::cout << "dE " << dE << std::endl;
        if (i > 0)
        {
            std::cout << (previous/dE) << std::endl;
        }
        previous = dE;
        dx *= 0.5;
    }
    run_diff_test = false;
}


void DiscreteShell::checkTotalHessian(bool perturb)
{
    std::cout << "======================== CHECK Hessian ========================" << std::endl;
    std::cout << "****** Only mismatching entries are printed ******" << std::endl;
    run_diff_test = true;
    T epsilon = 1e-5;
    int n_dof = deformed.rows();

    if (perturb)
    {
        VectorXT du(n_dof);
        du.setRandom();
        du *= 1.0 / du.norm();
        du *= 0.001;
        u += du;
    }
    StiffnessMatrix A(n_dof, n_dof);
    buildSystemMatrix(A);
    
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        // std::cout << dof_i << std::endl;
        u(dof_i) += epsilon;
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();
        
        computeResidual(g0); 
        
        u(dof_i) -= 2.0 * epsilon;
        
        computeResidual(g1); 
        
        u(dof_i) += epsilon;
        VectorXT row_FD = (g1 - g0) / (2.0 * epsilon);

        for(int i = 0; i < n_dof; i++)
        {
            
            if(A.coeff(i, dof_i) == 0 && row_FD(i) == 0)
                continue;
            if (std::abs( A.coeff(i, dof_i) - row_FD(i)) < 1e-3 * std::abs(row_FD(i)))
                continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    std::cout << "Hessian Diff Test Finished" << std::endl;
    run_diff_test = false;
}


void DiscreteShell::checkTotalHessianScale(bool perturb)
{
    
    std::cout << "===================== check Hessian Scale =====================" << std::endl;
    std::cout << "********************You Should Be Seeing 4s********************" << std::endl;
    std::cout << "************** Unless your function is quadratic **************" << std::endl;
    run_diff_test = true;
    int n_dof = deformed.rows();

    if (perturb)
    {
        VectorXT du(n_dof);
        du.setRandom();
        du *= 1.0 / du.norm();
        du *= 0.001;
        u += du;
    }
    
    StiffnessMatrix A(n_dof, n_dof);
    
    buildSystemMatrix(A);

    VectorXT f0(n_dof);
    f0.setZero();
    computeResidual(f0);
    f0 *= -1;
    
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    for(int i = 0; i < n_dof; i++) dx[i] += 0.5;
    dx *= 0.001;
    T previous = 0.0;
    VectorXT u_backup = u;
    for (int i = 0; i < 10; i++)
    {
        VectorXT f1(n_dof);
        f1.setZero();
        u = u_backup + dx;
        computeResidual(f1);
        f1 *= -1;
        T df_norm = (f0 + (A * dx) - f1).norm();
        // std::cout << "df_norm " << df_norm << std::endl;
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dx *= 0.5;
    }
    run_diff_test = false;
}
// ============================= DERIVATIVE TESTS END =============================


// ============================= Dynamics =============================
void DiscreteShell::addInertialEnergy(T& energy)
{
    T kinetic_energy = 0.0;
    if (use_consistent_mass_matrix)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
            FaceIdx indices = faces.segment<3>(face_idx * 3);
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

void DiscreteShell::addInertialForceEntry(VectorXT& residual)
{
    if (use_consistent_mass_matrix)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
            FaceIdx indices = faces.segment<3>(face_idx * 3);
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

void DiscreteShell::addInertialHessianEntries(std::vector<Entry>& entries)
{
    if (use_consistent_mass_matrix)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
            Matrix<T, 9, 9> mass_mat;
            computeConsistentMassMatrix(undeformed_vertices, mass_mat);
            FaceIdx indices = faces.segment<3>(face_idx * 3);
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

void DiscreteShell::updateDynamicStates()
{
    vn = (deformed - xn) / dt;
    xn = deformed;
}

void DiscreteShell::initializeDynamicStates()
{
    if (!use_consistent_mass_matrix)
        computeMassMatrix();
    xn = undeformed;
    vn = VectorXT::Zero(undeformed.rows());
}

void DiscreteShell::computeMassMatrix()
{
    MatrixXT V; MatrixXi F;
    vectorToIGLMatrix<T, 3>(undeformed, V);
    vectorToIGLMatrix<int, 3>(faces, F);
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
    mass_diagonal.resize(deformed.rows() / 3);
    mass_diagonal.setZero();
    mass_diagonal = M.diagonal();   
}

void DiscreteShell::computeConsistentMassMatrix(const FaceVtx& vtx_pos, Matrix<T, 9, 9>& mass_mat)
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

// ============================= Stress Tensor Utilities ============================

void DiscreteShell::computeStrainAndStressPerElement(){

    iterateFaceSerial([&](int face_idx)
    {   
        if(heterogenuous) {
            setMaterialParameter(E, nu, face_idx);
        }
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        T k_s = E * thickness / (1.0 - nu * nu);

        if(quadratic) {
            Matrix<T, 2, 2> F = compute2DDeformationGradient(x0, x1, x2, X0, X1, X2, {1.0, 0.});
            if(face_idx == 60) std::cout << "F: \n" << F << std::endl;
            F = compute2DDeformationGradient(x0, x1, x2, X0, X1, X2, {1/3., 1/3.});
            if(face_idx == 60) std::cout << "F middle: \n" << F << std::endl;
            TM2 GreenS = 0.5 *(F.transpose()*F - TM2::Identity());
            space_strain_tensors[face_idx] = GreenS;
            TM2 S = thickness*(nu * GreenS.trace() *TM2::Identity() + 2 * E * GreenS);
            T areaRatio = ((x1-x0).cross(x2-x0)).norm() / ((X1-X0).cross(X2-X0)).norm();
            cauchy_stress_tensors[face_idx].block(0,0,2,2) = F*S*F.transpose()/areaRatio;
            stress_tensors[face_idx].block(0,0,2,2) = S;
            strain_tensors[face_idx].block(0,0,2,2) = GreenS;
            optimization_homo_target_tensors[face_idx] = F;

        } else {
            Matrix<T, 3, 2> F = computeDeformationGradientwrt2DXSpace(X0, X1, X2, x0, x1, x2);
            TM2 GreenS = computeCauchyStrainwrt2dXSpace(F);
            // if(face_idx == 0) std::cout << "GS: \n" << GreenS << std::endl;
            space_strain_tensors[face_idx] = GreenS;
            TM2 S = k_s * ((1-nu)*2*GreenS + nu*2*GreenS.trace()*TM2::Identity());
            // if(face_idx == 0) std::cout << "S: \n" << S << std::endl;
            T areaRatio = ((x1-x0).cross(x2-x0)).norm() / ((X1-X0).cross(X2-X0)).norm();
            cauchy_stress_tensors[face_idx] = F*S*F.transpose()/areaRatio;

            TM2 x; x.col(0) = (vertices.row(1) - vertices.row(0)).segment(0, 2);x.col(1) = (vertices.row(2) - vertices.row(0)).segment(0, 2);
            TM2 X; X.col(0) = (undeformed_vertices.row(1) - undeformed_vertices.row(0)).segment(0, 2);X.col(1) = (undeformed_vertices.row(2) - undeformed_vertices.row(0)).segment(0, 2);
            TM2 F_2D = x*X.lu().solve(TM2::Identity());
            optimization_homo_target_tensors[face_idx] = F_2D;
            TM2 F_2D_inv = F_2D.lu().solve(TM2::Identity());
            stress_tensors[face_idx].block(0,0,2,2) = F_2D_inv * (F*S*F.transpose()).block(0,0,2,2) * F_2D_inv.transpose();
            // if(face_idx == 0) std::cout << "S post: \n" << stress_tensors[face_idx].block(0,0,2,2) << std::endl;
            TM2 E_2D = 0.5*(F_2D.transpose()*F_2D - TM2::Identity());
            // if(face_idx == 0) std::cout << "E_2D: \n" << E_2D << std::endl;
            strain_tensors[face_idx].block(0,0,2,2) = E_2D;
            // stress_tensors[face_idx].block(0,0,2,2) = k_s * ((1-nu)*2*E_2D + nu*2*E_2D.trace()*TM2::Identity());
            // if(face_idx == 0) std::cout << "S 2D post: \n" <<  k_s * ((1-nu)*2*E_2D + nu*2*E_2D.trace()*TM2::Identity()) << std::endl;
            // std::cout << face_idx << " " << E << " " << E_2D(1,1) << std::endl;
        }
    });
}

void DiscreteShell::computeHomogenization(){
    if(set_window){
        iterateFaceSerial([&](int face_idx)
        {
            TM homo_tensor = computeHomogenisedStressTensorinWindow();
            if(TriangleinsideWindow(window_x, window_y, window_length, window_height, face_idx)){
                homo_stress_tensors[face_idx] = homo_tensor;
                homo_stress_magnitudes[face_idx] = homo_stress_tensors[face_idx].norm();
            }
            homo_tensor = computeHomogenisedStrainTensorinWindow();
            if(TriangleinsideWindow(window_x, window_y, window_length, window_height, face_idx)){
                homo_strain_tensors[face_idx] = homo_tensor;
                homo_strain_magnitudes[face_idx] = homo_strain_tensors[face_idx].norm();
            }
        });
    }
}

void DiscreteShell::computeEnergyComparison(){

    iterateFaceSerial([&](int face_idx)
    {   
        if(heterogenuous) {
            setMaterialParameter(E, nu, face_idx);
        }
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        T k_s = E * thickness / (1.0 - nu * nu);

        Matrix<T, 3, 2> F = computeDeformationGradientwrt2DXSpace(X0, X1, X2, x0, x1, x2);
        TM2 E = computeCauchyStrainwrt2dXSpace(F);

        double energy = k_s * 0.5 * ((X1-X0).cross(X2-X0)).norm() * ((1-nu) * E.squaredNorm() + nu * E.trace() * E.trace());
        double energy_opt = compute3DCSTShellEnergy(nu, k_s, x0, x1, x2, X0, X1, X2);
        if (face_idx == 0) std::cout << "Energy comparison: " << energy << "  " << energy_opt << std::endl;
    });
}

Matrix<T, 2, 2> DiscreteShell::computeCauchyStrainwrt2dXSpace(Matrix<T, 3, 2> F){
    return 0.5 * (F.transpose()*F - TM2::Identity());
}

Matrix<T, 3, 2> DiscreteShell::computeDeformationGradientwrt2DXSpace(
    const TV X1, const TV X2, const TV X3, 
    const TV x1, const TV x2, const TV x3)
{
    TV localSpannT1 = (X2-X1).normalized();
    TV localSpannT2 = ((X3-X1) - localSpannT1*((X3-X1).dot(localSpannT1))).normalized();

    FaceVtx x; x << x1, x2, x3;

    Matrix<T, 3, 2> dNdB; dNdB << -1, -1, 1, 0, 0, 1;

    Matrix<T, 2, 3> dBdX = computeBarycentricJacobian(X1, X2, X3);
    Matrix<T, 3, 2> dXdX_2D;  dXdX_2D << localSpannT1, localSpannT2;

    Matrix<T, 3, 2> F = x * dNdB * dBdX * dXdX_2D;

    return F;
}

// pseudoinverse of [X2-X1, X3-X1]
Matrix<T, 2, 3> DiscreteShell::computeBarycentricJacobian(
    const TV X1, const TV X2, const TV X3)
{
    TV v1 = (X2-X1);
    TV v2 = (X3-X1);

    T denominator = v1.squaredNorm()*v2.squaredNorm() - v1.dot(v2) * v1.dot(v2);

    Matrix<T, 3, 2> transpose_dBdX;
    transpose_dBdX << v1*v2.squaredNorm() - v2*v1.dot(v2),
                       v2*v1.squaredNorm() - v1*v1.dot(v2);

    transpose_dBdX /= denominator;                   

    return transpose_dBdX.transpose();
}

// ======================================= Test Window Utilities =================================

// assume valid start coordinate for the window
Matrix<T, 3, 3> DiscreteShell::computeHomogenisedStressTensorinWindow(){
    T total_volume = 0;
    TM homogenised_stress_tensor = TM::Zero();

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        if(TriangleinsideWindow(window_x, window_y, window_length, window_height, face_idx)){
            T area = 0.5 * ((X1-X0).cross(X2-X0)).norm();
            total_volume += area * thickness;
            homogenised_stress_tensor += cauchy_stress_tensors[face_idx] * area * thickness;
        }
    });
    if (total_volume <= 0) {std::cout << "No triangle detected in the window!\n"; return TM::Zero();}
    return homogenised_stress_tensor / total_volume;
}

T DiscreteShell::computeWindowAreaRatio(){
    T area_undeformed = 0;
    T area_deformed = 0;

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        if(TriangleinsideWindow(window_x, window_y, window_length, window_height, face_idx)){
            area_undeformed += 0.5 * ((X1-X0).cross(X2-X0)).norm();
            area_deformed += 0.5 * ((x1-x0).cross(x2-x0)).norm();
        }
    });

    return area_deformed / area_undeformed;
}

// assume valid start coordinate for the window
Matrix<T, 3, 3> DiscreteShell::computeHomogenisedStrainTensorinWindow(){
    total_window_area_undeformed = 0;
    TM homogenised_strain_tensor = TM::Zero();

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        if(TriangleinsideWindow(window_x, window_y, window_length, window_height, face_idx)){
            T area = 0.5 * ((X1-X0).cross(X2-X0)).norm();
            total_window_area_undeformed += area;
            homogenised_strain_tensor.block(0,0,2,2) += space_strain_tensors[face_idx] * area;
        }
    });
    if (total_window_area_undeformed <= 0) {std::cout << "No triangle detected in the window!\n"; return TM::Zero();}
    return homogenised_strain_tensor / total_window_area_undeformed;
}

Matrix<T, 3, 3> DiscreteShell::computeHomogenisedTargetTensorinWindow(){
    total_window_area_undeformed = 0;
    TM homogenised_tensor = TM::Zero();

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        if(TriangleinsideWindow(window_x, window_y, window_length, window_height, face_idx)){
            T area = 0.5 * ((X1-X0).cross(X2-X0)).norm();
            total_window_area_undeformed += area;
            homogenised_tensor.block(0,0,2,2) += optimization_homo_target_tensors[face_idx] * area;
        }
    });
    if (total_window_area_undeformed <= 0) {std::cout << "No triangle detected in the window!\n"; return TM::Zero();}
    return homogenised_tensor / total_window_area_undeformed;
}

// assume valid start coordinate for the window
// bool DiscreteShell::TriangleinsideWindow(int start_x, int start_y, int length, int height, int face_idx){

//     int start_node_idx_bottom = 10*start_x + start_y;
//     int start_node_idx_top = 10*(start_x+height) + start_y;
//     FaceIdx indices = faces.segment<3>(face_idx * 3);

//     for(int i = 0; i < indices.size(); ++i){
//         int current_idx = indices[i];
//         if(current_idx / 10 > std::min(9, height+start_x) || current_idx/10 < start_x) return false;
//         if(current_idx % 10 < start_y || current_idx % 10 > std::min(9, length+start_y)) return false;
//     }
//     return true;
// }

bool DiscreteShell::TriangleinsideWindow(int start_x, int start_y, int length, int height, int face_idx){

    FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

    TV d = undeformed_vertices.row(0) - undeformed_vertices.row(1);
    T dx = std::max(std::abs(d(0)), std::abs(d(1)));
    TV CoM = triangleCenterofMass(undeformed_vertices);
    TV start; start << -dx*start_x, dx*start_y, 0;
    TV diff = CoM - start;
    // if(face_idx == 99) {std::cout << "CoM of 99 \n" << CoM  << "\ndiff: " << -diff(0)/dx << " " << diff(1)/dx << "\n dx: "<< dx<<std::endl;}
    if(-diff(0) > length*dx || diff(1)/dx > height || diff(0) > 0. || diff(1) < 0) return false;

    return true;
}

void DiscreteShell::addEnergyforDesiredTarget(T &energy){

    computeStrainAndStressPerElement();
    TM homo_tensor = computeHomogenisedTargetTensorinWindow(); 
    energy += (homo_tensor(0,0)-epsilons(0))*(homo_tensor(0,0)-epsilons(0))*weights(0) + 
        (homo_tensor(1,1)-epsilons(3))*(homo_tensor(1,1)-epsilons(3))*weights(3) + (homo_tensor(0,1)-epsilons(1))*(homo_tensor(0,1)-epsilons(1))*weights(1) +
        (homo_tensor(1,0)-epsilons(2))*(homo_tensor(1,0)-epsilons(2))*weights(2);
}

Vector<T, 9> DiscreteShell::computeGradientForDesiredTargetXX(const TV X1, const TV X2, const TV X3, const TV x1, const TV x2, const TV x3){
    T p[9]; T q[9];
    p[0] = X1(0); p[1] = X1(1); p[2] = X1(2);
    p[3] = X2(0); p[4] = X2(1); p[5] = X2(2);
    p[6] = X3(0); p[7] = X3(1); p[8] = X3(2);
    q[0] = x1(0); q[1] = x1(1); q[2] = x1(2);
    q[3] = x2(0); q[4] = x2(1); q[5] = x2(2);
    q[6] = x3(0); q[7] = x3(1); q[8] = x3(2);
    T t1 = (p[3] - p[6]) * p[1] - (-p[6] + p[0]) * p[4] + p[7] * (-p[3] + p[0]);
    t1 = 0.1e1 / t1;
    VectorXT gradient(9);
    gradient[0] = -(p[4] - p[7]) * t1;
    gradient[1] = 0;
    gradient[2] = 0;
    gradient[3] = (p[1] - p[7]) * t1;
    gradient[4] = 0;
    gradient[5] = 0;
    gradient[6] = -(-p[4] + p[1]) * t1;
    gradient[7] = 0;
    gradient[8] = 0;

    return gradient;
}

Vector<T, 9> DiscreteShell::computeGradientForDesiredTargetYY(const TV X1, const TV X2, const TV X3, const TV x1, const TV x2, const TV x3){
    T p[9]; T q[9];
    p[0] = X1(0); p[1] = X1(1); p[2] = X1(2);
    p[3] = X2(0); p[4] = X2(1); p[5] = X2(2);
    p[6] = X3(0); p[7] = X3(1); p[8] = X3(2);
    q[0] = x1(0); q[1] = x1(1); q[2] = x1(2);
    q[3] = x2(0); q[4] = x2(1); q[5] = x2(2);
    q[6] = x3(0); q[7] = x3(1); q[8] = x3(2);
    T t1 = (p[4] - p[7]) * p[0] - (p[1] - p[7]) * p[3] + p[6] * (-p[4] + p[1]);
    t1 = 0.1e1 / t1;

    VectorXT gradient(9);gradient[0] = 0;
    gradient[1] = -(p[3] - p[6]) * t1;
    gradient[2] = 0;
    gradient[3] = 0;
    gradient[4] = (-p[6] + p[0]) * t1;
    gradient[5] = 0;
    gradient[6] = 0;
    gradient[7] = -(-p[3] + p[0]) * t1;
    gradient[8] = 0;


    return gradient;
}

Vector<T, 9> DiscreteShell::computeGradientForDesiredTargetXY(const TV X1, const TV X2, const TV X3, const TV x1, const TV x2, const TV x3){
    T p[9]; T q[9];
    p[0] = X1(0); p[1] = X1(1); p[2] = X1(2);
    p[3] = X2(0); p[4] = X2(1); p[5] = X2(2);
    p[6] = X3(0); p[7] = X3(1); p[8] = X3(2);
    q[0] = x1(0); q[1] = x1(1); q[2] = x1(2);
    q[3] = x2(0); q[4] = x2(1); q[5] = x2(2);
    q[6] = x3(0); q[7] = x3(1); q[8] = x3(2);
    T t1 = (p[4] - p[7]) * p[0] - (p[1] - p[7]) * p[3] + p[6] * (-p[4] + p[1]);
    t1 = 0.1e1 / t1;
    VectorXT gradient(9);
    gradient[0] = -(p[3] - p[6]) * t1;
    gradient[1] = 0;
    gradient[2] = 0;
    gradient[3] = (-p[6] + p[0]) * t1;
    gradient[4] = 0;
    gradient[5] = 0;
    gradient[6] = -(-p[3] + p[0]) * t1;
    gradient[7] = 0;
    gradient[8] = 0;


    return gradient;
}

Vector<T, 9> DiscreteShell::computeGradientForDesiredTargetYX(const TV X1, const TV X2, const TV X3, const TV x1, const TV x2, const TV x3){
    T p[9]; T q[9];
    p[0] = X1(0); p[1] = X1(1); p[2] = X1(2);
    p[3] = X2(0); p[4] = X2(1); p[5] = X2(2);
    p[6] = X3(0); p[7] = X3(1); p[8] = X3(2);
    q[0] = x1(0); q[1] = x1(1); q[2] = x1(2);
    q[3] = x2(0); q[4] = x2(1); q[5] = x2(2);
    q[6] = x3(0); q[7] = x3(1); q[8] = x3(2);
    T t1 = (p[3] - p[6]) * p[1] - (-p[6] + p[0]) * p[4] + p[7] * (-p[3] + p[0]);
    t1 = 0.1e1 / t1;
    VectorXT gradient(9);
    gradient[0] = 0;
    gradient[1] = -(p[4] - p[7]) * t1;
    gradient[2] = 0;
    gradient[3] = 0;
    gradient[4] = (p[1] - p[7]) * t1;
    gradient[5] = 0;
    gradient[6] = 0;
    gradient[7] = -(-p[4] + p[1]) * t1;
    gradient[8] = 0;

    return gradient;
}

void DiscreteShell::addGradientForDesiredTarget(VectorXT& residual)
{   
    computeStrainAndStressPerElement();
    TM homo_tensor = computeHomogenisedTargetTensorinWindow(); 
    std::cout << "Current constraint energy: " << (homo_tensor(0,0)-epsilons(0))*(homo_tensor(0,0)-epsilons(0)) + 
        (homo_tensor(1,1)-epsilons(3))*(homo_tensor(1,1)-epsilons(3)) + (homo_tensor(0,1)-epsilons(1))*(homo_tensor(0,1)-epsilons(1)) +
        (homo_tensor(1,0)-epsilons(2))*(homo_tensor(1,0)-epsilons(2))
        << std::endl;
    std::cout << "Current homogenised target tensor: \n" << homo_tensor << std::endl;    
    iterateFaceSerial([&](int face_idx)
    {
        
        if(TriangleinsideWindow(window_x, window_y, window_length, window_height, face_idx)){
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
            FaceIdx indices = faces.segment<3>(face_idx * 3);

            TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);
            T area = 0.5 * ((X1-X0).cross(X2-X0)).norm();
            Vector<T, 9> dedx_xx = computeGradientForDesiredTargetXX(X0, X1, X2, x0, x1, x2)*area;
            dedx_xx *= 2*(homo_tensor(0,0)-epsilons(0))/total_window_area_undeformed;
            Vector<T, 9> dedx_xy = computeGradientForDesiredTargetXY(X0, X1, X2, x0, x1, x2)*area;
            dedx_xy *= 2*(homo_tensor(0,1)-epsilons(1))/total_window_area_undeformed;
            Vector<T, 9> dedx_yy = computeGradientForDesiredTargetYY(X0, X1, X2, x0, x1, x2)*area;
            dedx_yy *= 2*(homo_tensor(1,1)-epsilons(3))/total_window_area_undeformed;
            Vector<T, 9> dedx_yx = computeGradientForDesiredTargetYX(X0, X1, X2, x0, x1, x2)*area;
            dedx_yx *= 2*(homo_tensor(1,0)-epsilons(2))/total_window_area_undeformed;
            
            addForceEntry<3>(residual, {indices[0], indices[1], indices[2]}, -dedx_xx*weights(0)-dedx_yy*weights(3)-dedx_xy*weights(1)-dedx_yx*weights(2));
        }
    });
}

T DiscreteShell::computeTestWindowForces(VectorXT& forces)
{
    deformed = undeformed + u;
    addShellForceEntry(forces);
    if (add_gravity)
        addShellGravitionForceEntry(forces);
    if (dynamics)
        addInertialForceEntry(forces);

    return forces.norm();
}

// ============================= Boundary Utilities =================================

// assume for now I only set explicit BC for the lower boundary (node_idx 0-9)
void DiscreteShell::setEssentialBoundaryCondition(T displacement_x, T displacement_y){

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


// ============================= Kernel-weighted Cut Utilities ======================
// assume planar case for now
void DiscreteShell::setProbingLineDirections(unsigned int num_directions){
    direction = std::vector<TV>(num_directions);
    T angle = std::acos(-1) / num_directions;
    for(int i = 0; i < num_directions; ++i){
        direction[i] << std::cos(angle*i), std::sin(angle*i), 0;
    }
}

Matrix<T, 3, 3> DiscreteShell::findBestStressTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions){
    int tri = pointInTriangle(sample_loc);
    if(tri == -1) std::cout << "Sample point not in mesh!" << std::endl;
    // std::cout << "Found sample point in triangle: " << tri << std::endl;
    // TM2 F_2D_inv = optimization_homo_target_tensors[tri].lu().solve(TM2::Identity());
    int c = line_directions.size();
    MatrixXT n(3, c);
    MatrixXT t(3, c);
    for(int i = 0; i < c; ++i){
        TV direction = line_directions.at(i);
        TV2 direction_normal_2D; direction_normal_2D << direction(1), -direction(0);
        // direction_normal_2D = F_2D_inv.transpose()*direction_normal_2D;
        TV direction_normal; direction_normal.segment(0,2) = direction_normal_2D;
        // direction_normal = direction_normal.normalized(); 
        t.col(i) = computeWeightedStress(sample_loc, direction);
        n.col(i) = direction_normal;
    }

    bool fit_symmetric_constrained = true;
    TM fitted_symmetric_tensor;
    if(!fit_symmetric_constrained){
        MatrixXT A = n.transpose();
        MatrixXT b = t.transpose();
        TM x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
        fitted_symmetric_tensor = x.transpose();
    } else {
        MatrixXT A = MatrixXT::Zero(3*c,6);
        VectorXT b(3*c);
        for(int i = 0; i < c; ++i){
            MatrixXT A_block = MatrixXT::Zero(3,6);
            TV normal = n.col(i);
            A_block << normal(0), normal(1), 0, normal(2), 0, 0,
                    0, normal(0), normal(1), 0, normal(2), 0,
                    0, 0, 0, normal(0), normal(1), normal(2);
            A.block(i*3, 0, 3, 6) = A_block;
            b.segment(i*3, 3) = t.col(i);
        }
        VectorXT x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
        fitted_symmetric_tensor << x(0), x(1), x(3), 
                                    x(1), x(2), x(4),
                                    x(3), x(4), x(5);
    }

    return fitted_symmetric_tensor;
}

Matrix<T, 2, 2> DiscreteShell::findBestStrainTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions){
    int c = line_directions.size();
    MatrixXT n(3, c);
    VectorXT t(c);
    for(int i = 0; i < c; ++i){
        TV direction = line_directions.at(i);
        t(i) = computeWeightedStrain(sample_loc, direction);
        // if(t(i) >= 1e3 || t(i) <= -1e3) {std::cout << "sample " << sample_loc.transpose() <<  " with direction " << direction.transpose() << " : with strain: " << t(i) << std::endl;} 
        n.col(i) = direction;
    }

    TM2 fitted_symmetric_tensor;
    MatrixXT A = MatrixXT::Zero(c,3);
    for(int i = 0; i < c; ++i){
        MatrixXT A_block = MatrixXT::Zero(1,3);
        TV2 normal = n.col(i).segment(0,2);
        A_block << normal(0)*normal(0), 2*normal(1)*normal(0), normal(1)*normal(1);
        A.row(i) = A_block;
        }
        VectorXT x = (A.transpose()*A).ldlt().solve(A.transpose()*t);
        fitted_symmetric_tensor << x(0), x(1), 
                                    x(1), x(2);

    return fitted_symmetric_tensor;
}

Matrix<T, 3, 3> DiscreteShell::findBestStressTensorviaAveraging(const TV sample_loc){
    T pi = std::acos(-1);
    T std = 7e-3;
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel = [pi, variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / std::sqrt(std::pow((2 * pi), 2)*variance_matrix.determinant());
    };

    T sum = 0.;
    TM stress = TM::Zero();

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxUndeformed(face_idx);
            
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);

        stress.block(0,0,2,2) += gaussian_kernel(sample_loc, triangleCenterofMass(vertices)) * 
            stress_tensors[face_idx].block(0,0,2,2);
        sum += gaussian_kernel(sample_loc, triangleCenterofMass(vertices));

        if((sample_loc-sample[1]).norm() <= 1e-4) kernel_coloring_avg[face_idx] = gaussian_kernel(sample_loc, triangleCenterofMass(vertices));

    }); 
    return stress/sum;
}

Matrix<T, 3, 3> DiscreteShell::findBestStrainTensorviaAveraging(const TV sample_loc){
    T pi = std::acos(-1);
    T std = 7e-3;
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel = [pi, variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / std::sqrt(std::pow((2 * pi), 2)*variance_matrix.determinant());
    };

    T sum = 0.;
    TM strain = TM::Zero();

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxUndeformed(face_idx);
            
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);

        strain += gaussian_kernel(sample_loc, triangleCenterofMass(vertices)) * strain_tensors[face_idx];
        sum += gaussian_kernel(sample_loc, triangleCenterofMass(vertices));

    }); 
    return strain/sum;
}

Vector<T, 3> DiscreteShell::computeWeightedStress(const TV sample_loc, TV direction){
    T pi = std::acos(-1);
    T std = 7e-3;
    int choose_gaussian_kernel = 1;
    auto gaussian_kernel1 = [pi, std](T distance){
        return std::exp(-0.5*distance*distance/(std*std)) / (std * std::sqrt(2 * pi));
    };
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel2 = [pi, variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / std::sqrt(std::pow((2 * pi), 2)*variance_matrix.determinant());
    };
    T sum = 0.;
    TV stress = TV::Zero();
    TV direction_normal; direction_normal << direction(1), -direction(0), 0;
    direction_normal = direction_normal.normalized(); 

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxUndeformed(face_idx);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV cut_point_coordinate;
        if(lineCutTriangle(x0, x1, x2, sample_loc, direction, cut_point_coordinate)){
            TV middle = middlePointoflineCutTriangle(x0, x1, x2, cut_point_coordinate);
            if(choose_gaussian_kernel == 1){
                T distance = (sample_loc - middle).norm();
                stress += gaussian_kernel1(distance) * stress_tensors[face_idx] * direction_normal;
                sum += gaussian_kernel1(distance);

                // visulization
                if((sample_loc-sample[1]).norm() <= 1e-4) kernel_coloring_prob[face_idx] = gaussian_kernel1(distance);
            } else if(choose_gaussian_kernel == 2){
                stress += gaussian_kernel2(sample_loc, middle) * stress_tensors[face_idx] * direction_normal;
                sum += gaussian_kernel2(sample_loc, middle);

                // visulization
                if((sample_loc-sample[1]).norm() <= 1e-4) kernel_coloring_prob[face_idx] = gaussian_kernel2(sample_loc, middle);
            }
            
        }
    });
    if (sum <= 0.) std::cout << "Sum is 0 for direction " << direction.transpose() << std::endl; 

    return stress/sum;
}

T DiscreteShell::computeWeightedStrain(const TV sample_loc, TV direction){
    T pi = std::acos(-1);
    T std = 7e-3;
    int choose_gaussian_kernel = 1;
    auto gaussian_kernel1 = [pi, std](T distance){
        return std::exp(-0.5*distance*distance/(std*std)) / (std * std::sqrt(2 * pi));
    };
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel2 = [pi, variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / std::sqrt(std::pow((2 * pi), 2)*variance_matrix.determinant());
    };
    T sum = 0.;
    T strain = 0;

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxUndeformed(face_idx);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);

        TV cut_point_coordinate;
        bool compute_with_segments = true;
        if(lineCutTriangle(x0, x1, x2, sample_loc, direction, cut_point_coordinate)){
            TV middle = middlePointoflineCutTriangle(x0, x1, x2, cut_point_coordinate);
            // if((sample_loc-sample[1]).norm() <= 1e-4) {std::cout << "triangle "<< face_idx << " has strain : " << 
            //         strainInCut(face_idx, cut_point_coordinate) << " in direction: " << direction.transpose()<< 
            //         " with cut in " << cut_point_coordinate.transpose() << std::endl; 
            //         }
            if(choose_gaussian_kernel == 1){
                T distance = (sample_loc - middle).norm();
                if(compute_with_segments)
                    strain += gaussian_kernel1(distance)*strainInCut(face_idx, cut_point_coordinate);
                else strain += gaussian_kernel1(distance)*direction.transpose()*strain_tensors[face_idx]*direction;    
                sum += gaussian_kernel1(distance);

            } else if(choose_gaussian_kernel == 2){
                if(compute_with_segments)
                    strain += gaussian_kernel2(sample_loc, middle)*strainInCut(face_idx, cut_point_coordinate);
                else gaussian_kernel2(sample_loc, middle)*direction.transpose()*strain_tensors[face_idx]*direction; 
                sum += gaussian_kernel2(sample_loc, middle);
            }
        }
    });

    if(sum <= 0) {std::cout << "Weighted strain: "<< strain << " Something wrong with the weighting!\n";} 

    return strain/sum;
}

Vector<T, 3> DiscreteShell::triangleCenterofMass(FaceVtx vertices){
    TV CoM; CoM << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
    return CoM;
}

void DiscreteShell::visualizeCuts(const std::vector<TV> sample_points, const std::vector<TV> line_directions){
    unsigned int tag = 1;
    for(auto sample_point: sample_points){
        for(auto direction: line_directions){
            visualizeCut(sample_point, direction, tag);
            ++tag;
        }
    }
}
void DiscreteShell::visualizeCut(const TV sample_point, const TV line_direction, unsigned int line_tag){

    iterateFaceSerial([&](int face_idx)
    {
        if(cut_coloring[face_idx] == line_tag) cut_coloring[face_idx] = 0;
        FaceVtx vertices = getFaceVtxUndeformed(face_idx);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV cut_point_coordinate;
        if(lineCutTriangle(x0, x1, x2, sample_point, line_direction, cut_point_coordinate)){
            cut_coloring[face_idx] = line_tag;
        }
    });
}

// return if a line cut through a triangle and return the cut points' barycentric coordinate
bool DiscreteShell::lineCutTriangle(const TV x1, const TV x2, const TV x3, const TV sample_point, const TV line_direction, TV &cut_point_coordinate){
    
    int count = 0;
    cut_point_coordinate << -1, -1, -1;
    std::vector<TV> points(3, TV::Zero());

    TV2 r1 = solveLineIntersection(sample_point, line_direction, x3, x2);
    TV2 r2 = solveLineIntersection(sample_point, line_direction, x2, x1);
    TV2 r3 = solveLineIntersection(sample_point, line_direction, x1, x3);
    if(r1(0) >= 0.-1e-6 && r1(0) <= 1.+1e-8) {++count; cut_point_coordinate(1) = r1(0);points[1] = x2 + r1(0)*(x3-x2);}
    if(r2(0) >= 0.-1e-6 && r2(0) <= 1.+1e-8) {++count; cut_point_coordinate(0) = r2(0);points[0] = x1 + r2(0)*(x2-x1);}
    if(r3(0) >= 0.-1e-6 && r3(0) <= 1.+1e-8) {++count; cut_point_coordinate(2) = r3(0);points[2] = x3 + r3(0)*(x1-x3);}
    
    // check if intersections are at the corners
    for(int i = 0; i < 3; ++i){
        for(int j = i+1; j < 3; ++j){
            if(cut_point_coordinate(i) <= -1 || cut_point_coordinate(j) <= -1) continue;
            if((points[i]-points[j]).norm() <= 1e-8) {
                --count; cut_point_coordinate(j) = -1;
            }
        }
    }
    

    if(count > 1) return true;
    return false;

}

// find middle point of a cut segment through the triangle element
Vector<T,3> DiscreteShell::middlePointoflineCutTriangle(const TV x1, const TV x2, const TV x3, const TV cut_point_coordinate){
    
    TV middle_point = TV::Zero();
    if(cut_point_coordinate(0) >= 0.-1e-6 && cut_point_coordinate(0) <= 1.+1e-8) {middle_point += x1 + cut_point_coordinate(0)*(x2-x1);}
    if(cut_point_coordinate(1) >= 0.-1e-6 && cut_point_coordinate(1) <= 1.+1e-8) {middle_point += x2 + cut_point_coordinate(1)*(x3-x2);}
    if(cut_point_coordinate(2) >= 0.-1e-6 && cut_point_coordinate(2) <= 1.+1e-8) {middle_point += x3 + cut_point_coordinate(2)*(x1-x3);}

    return middle_point/2;

}

// calculate strain in current cut segment using barycentric coordinate
T DiscreteShell::strainInCut(const int face_idx, const TV cut_point_coordinate){

    FaceVtx vertices = getFaceVtxDeformed(face_idx);
    FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

    TM cuts = TM::Zero();
    TM cuts_undeformed = TM::Zero();
    std::vector<int> recorder;
    for(int i = 0; i < 3; ++i){
        T cut_point = cut_point_coordinate(i);
        if(cut_point >= 0.-1e-8 && cut_point <= 1.+1e-8) {
            cuts.col(i) = vertices.row(i) + cut_point*(vertices.row((i+1)%3) - vertices.row(i));
            cuts_undeformed.col(i) = undeformed_vertices.row(i) + cut_point*(undeformed_vertices.row((i+1)%3) - undeformed_vertices.row(i));
            recorder.push_back(i);
        }    
    }
    assert(recorder.size() == 2);
    T l = (cuts.col(recorder.at(0)) - cuts.col(recorder.at(1))).norm();
    T L0 = (cuts_undeformed.col(recorder.at(0)) - cuts_undeformed.col(recorder.at(1))).norm();

    return 0.5*(l*l-L0*L0)/(L0*L0);
}

Vector<T,2> DiscreteShell::solveLineIntersection(const TV sample_point, const TV line_direction, const TV v1, const TV v2){
    TV e = v1 - v2;
    TV b; b << sample_point-v2;
    Matrix<T, 3, 2> A; A.col(0) = e; A.col(1) = -line_direction;
    TV2 r = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    if(std::abs(e.normalized().transpose()*line_direction.normalized())-1 >= 0.-1e-8) r(0) = -1;

    return r;
}

void DiscreteShell::testIsotropicStretch(){
    deformed = 0.7 * undeformed;
    computeStrainAndStressPerElement();
    int A = 40;
    TM vertices = getFaceVtxUndeformed(A);
    sample[1] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
    std::cout << "Found stress tensor via Probing: \n" << findBestStressTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found stress tensor via Averaging: \n" << findBestStressTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated stress tensor at sample point triangle: \n" << cauchy_stress_tensors[A] << std::endl;
    std::cout << "Found strain tensor via Probing: \n" << findBestStrainTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found strain tensor via Averaging: \n" << findBestStrainTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated strain tensor at sample point triangle: \n" << strain_tensors[A] << std::endl;
}

void DiscreteShell::testHorizontalDirectionStretch(){

    for(int i = 0; i < deformed.size(); i +=3) deformed[i] = 0.7 * undeformed[i];
    computeStrainAndStressPerElement();
    int A = 40;
    TM vertices = getFaceVtxUndeformed(A);
    sample[1] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
    std::cout << "Found stress tensor via Probing: \n" << findBestStressTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found stress tensor via Averaging: \n" << findBestStressTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated stress tensor at sample point triangle: \n" << cauchy_stress_tensors[A] << std::endl;
    std::cout << "Found strain tensor via Probing: \n" << findBestStrainTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found strain tensor via Averaging: \n" << findBestStrainTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated strain tensor at sample point triangle: \n" << strain_tensors[A] << std::endl;

}

void DiscreteShell::testVerticalDirectionStretch(){

    int A = 0;
    int B = deformed.size()/3/2;
    for(int i = 1; i < deformed.size(); i +=3) deformed[i] = 0.7 * undeformed[i];
    computeStrainAndStressPerElement();
    TM vertices = getFaceVtxUndeformed(A);
    sample[1] << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
    sample[0] = triangleCenterofMass(getFaceVtxUndeformed(B));
    std::cout << "Found strain tensor via Probing: \n" << findBestStrainTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found strain tensor via Averaging: \n" << findBestStrainTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated strain tensor at sample point triangle: \n" << strain_tensors[A] << std::endl;
    std::cout << "Found stress tensor via Probing: \n" << findBestStressTensorviaProbing(sample[1], direction) << std::endl;
    std::cout << "Found stress tensor via Averaging: \n" << findBestStressTensorviaAveraging(sample[1]) << std::endl;
    std::cout << "Caculated stress tensor at sample point triangle: \n" << cauchy_stress_tensors[A] << std::endl;

}

std::vector<Matrix<T, 3, 3>> DiscreteShell::returnStrainTensors(int A){

    TM vertices = getFaceVtxUndeformed(A);
    auto CoM = triangleCenterofMass(vertices);
    TM E = TM::Zero();
    E.block(0,0,2,2) = findBestStrainTensorviaProbing(CoM, direction);
    // T x = triangleCenterofMass(vertices)(1);
    // T e = 1e6;
    // e = (1-x*1.5)*e;
    // std::cout << A << " " << e << " " << findBestStrainTensorviaAveraging(CoM)(1,1) << std::endl;

    return {strain_tensors.at(A), E, findBestStrainTensorviaAveraging(CoM)};
}

std::vector<Matrix<T, 3, 3>> DiscreteShell::returnStressTensors(int A){

    TM vertices = getFaceVtxUndeformed(A);
    auto CoM = triangleCenterofMass(vertices);

    return {stress_tensors.at(A), findBestStressTensorviaProbing(CoM, direction), findBestStressTensorviaAveraging(CoM)};
}

void DiscreteShell::testSharedEdgeStress(int A, int B, int v1, int v2) {
    
    std::cout << "Quick test for edge stresses...\n";
    TV edge = undeformed.segment<3>(v1*3) - undeformed.segment<3>(v2*3);
    TV normal; normal << -edge(1), edge(0), 0; normal = normal.normalized();
    std::cout << "normal direction: " << normal.transpose() << std::endl;
    TV stress_1 = findBestStressTensorviaProbing(triangleCenterofMass(getFaceVtxUndeformed(A)), direction) *normal;
    TV stress_2 = findBestStressTensorviaProbing(triangleCenterofMass(getFaceVtxUndeformed(B)), direction) *normal;
    std::cout << "Kernel stress from triangle " << A << " : " << stress_1.transpose() << "\nKernel stress from triangle " << B << " : " << stress_2.transpose() << std::endl; 
    stress_1 = stress_tensors[A]*normal;
    stress_2 = stress_tensors[B]*normal;
    std::cout << "Local stress from triangle " << A << " : " << stress_1.transpose() << "\nLocal stress from triangle " << B << " : " << stress_2.transpose() << std::endl; 
    stress_1 = findBestStressTensorviaAveraging(triangleCenterofMass(getFaceVtxUndeformed(A))) *normal;
    stress_2 = findBestStressTensorviaAveraging(triangleCenterofMass(getFaceVtxUndeformed(B))) *normal;
    std::cout << "Average stress from triangle " << A << " : " << stress_1.transpose() << "\nAverage stress from triangle " << B << " : " << stress_2.transpose() << std::endl; 
}

void DiscreteShell::testStressTensors(int A, int B){

    sample[0] = triangleCenterofMass(getFaceVtxUndeformed(A));
    sample[1] = triangleCenterofMass(getFaceVtxUndeformed(B));
    std::cout << "Tested triangle: " << A << std::endl;
    std::cout << "Found stress tensor via Probing: \n" << findBestStressTensorviaProbing(sample[0], direction) << std::endl; 
    std::cout << "Found stress tensor via Averaging: \n" << findBestStressTensorviaAveraging(sample[0]) << std::endl;
    std::cout << "Caculated stress tensor at sample point triangle: \n" << stress_tensors[A].block(0,0,2,2)*optimization_homo_target_tensors[A].transpose()/areaRatio(A) << std::endl;
    std::cout << "Found strain tensor via Probing: \n" << findBestStrainTensorviaProbing(sample[0], direction) << std::endl;
    std::cout << "Found strain tensor via Averaging: \n" << findBestStrainTensorviaAveraging(sample[0]) << std::endl;
    std::cout << "Caculated strain tensor at sample point triangle: \n" << strain_tensors[A] << std::endl;

}

T DiscreteShell::areaRatio(int A){
    FaceVtx vertices = getFaceVtxDeformed(A);
    FaceVtx undeformed_vertices = getFaceVtxUndeformed(A);

    TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
    TV X0 = undeformed_vertices.row(0); 
    TV X1 = undeformed_vertices.row(1); 
    TV X2 = undeformed_vertices.row(2);

    return ((x1-x0).cross(x2-x0)).norm() / ((X1-X0).cross(X2-X0)).norm();
}

int DiscreteShell::pointInTriangle(const TV sample_loc){

    for (int i = 0; i < faces.rows()/3; i++){
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(i);

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
std::vector<Vector<T, 3>> DiscreteShell::pointInDeformedTriangle(){
    std::vector<TV> update;
    for(auto sample_loc: sample){
        update.push_back(pointInDeformedTriangle(sample_loc));
    }

    return update;
}
Vector<T, 3> DiscreteShell::pointInDeformedTriangle(const TV sample_loc){

    for (int i = 0; i < faces.rows()/3; i++){
        FaceVtx undeformed_vertices = getFaceVtxUndeformed(i);

        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);
        TM2 X; X.col(0) = (X1-X0).segment(0,2); X.col(1) = (X2-X0).segment(0,2); 
        T denom = X.determinant();
        X.col(0) = (X1-sample_loc).segment(0,2); X.col(1) = (X2-sample_loc).segment(0,2); 
        T alpha = X.determinant()/denom;
        X.col(0) = (X1-X0).segment(0,2); X.col(1) = (sample_loc-X0).segment(0,2); 
        T beta = X.determinant()/denom;
        T gamma = 1-alpha-beta;

        if (alpha >= 0 && beta >= 0 && gamma >= 0) {
            FaceVtx vertices = getFaceVtxDeformed(i);
            TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
            return alpha*x0 + gamma*x1 + beta*x2;  
        }
    }

    return TV::Zero();
}