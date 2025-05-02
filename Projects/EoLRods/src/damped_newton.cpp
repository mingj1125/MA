#include "../include/damped_newton.h"

#include <iostream>
#include <math.h>
#include <iomanip>
#include <fstream>

#include <Eigen/CholmodSupport>

#ifdef ALUNA_USE_PARDISO
#include <Eigen/PardisoSupport>
#else
#include <Eigen/UmfPackSupport>
#include <Eigen/CholmodSupport>
#endif

#include <chrono>

void DampedNewtonSolver::SetParameters(VectorXa& parameters_p)
{
    parameters=parameters_p;
}

void DampedNewtonSolver::SetOptions(damped_newton_options&& options_p)
{
    options=std::move(options_p);
}

void DampedNewtonSolver::SetCostFunction(CostFunction* function_p)
{
    function=function_p;
}

AScalar DampedNewtonSolver::MaxJtJDiagonalValue(Eigen::SparseMatrix<AScalar>& A)
{
    AScalar max = std::abs(A.coeff(0,0));

    for(int i=1; i<A.rows(); ++i)
        max = std::max(max, std::abs(A.coeff(i,i)));

    return max;
}

Eigen::SparseMatrix<AScalar> DampedNewtonSolver::DiagonalMatrix(Eigen::SparseMatrix<AScalar>& A)
{
    Eigen::SparseMatrix<AScalar> D(A.rows(), A.rows());

    for(int i=0; i<A.rows(); ++i) {
        D.coeffRef(i, i) = abs(A.coeff(i, i));
    }

    return D;
}

damped_newton_result DampedNewtonSolver::Solve()
{
    //std::ofstream iterations_file("iterations.csv");

    damped_newton_result result;
    result.n_iterations = 0;
    result.n_iterations_accepted = 0;
    std::ofstream outputlog(options.output_log+".log");

    if(options.use_log){
        std::cout << std::setw(10) << "# Iter" << std::setw(20) << "dir" << std::setw(20) << "h norm" << std::setw(20) << "Cost" << std::setw(20) << "New Cost" << std::setw(20) << "r norm" << std::setw(20) << "r_new norm" << std::endl;
        if (!outputlog) {
            std::cerr << "Error opening file for writing: " << options.output_log << std::endl;
        } 
        outputlog << std::setw(10) << "# Iter" << std::setw(20) << "dir" << std::setw(20) << "h norm" << std::setw(20) << "Cost" << std::setw(20) << "New Cost" << std::setw(20) << "r norm" << std::setw(20) << "r_new norm" << std::endl;
    }

    Eigen::SparseMatrix<AScalar> I(parameters.rows(), parameters.rows());
    //Eigen::SparseMatrix<AScalar> D;

    if(options.damp_matrix.rows()>0)
    	I = options.damp_matrix;
    else
    {
	    for(int i=0; i<parameters.rows(); ++i)
	        I.coeffRef(i,i) = 1.0;
	}

    int k=0; AScalar nu=2; bool found=false;
    Eigen::SparseMatrix<AScalar> J(parameters.rows(), parameters.rows());
    VectorXa r(parameters.rows());
//std::cout << parameters.rows() << std::endl;
    function->PreProcess(parameters);

    AScalar Fx;
    bool first_is_valid = true;
    try{
    	std::tie(Fx, r, J) = function->Evaluate(parameters);
    }
    catch(TerminateNowException e)
    {
    	found = true;
    	first_is_valid = false;
    }
    catch(std::exception e)
    {
    	first_is_valid = false;
    }

    if(!first_is_valid) {
        std::cout << "Invalid configuration!" << std::endl;

        function->RejectStep();

        return result;
    }

    // iterations_file << k << " " << Fx << " " << r.lpNorm<Eigen::Infinity>() << std::endl;
    // std::cout << k << " " << Fx << " " << r.lpNorm<Eigen::Infinity>() << " " << J.norm() << std::endl;

    function->AcceptStep();

    MatrixXa U,C,V; //Woodbury
    Eigen::SparseMatrix<AScalar> A = J;
    
    VectorXa g = r;

    AScalar g_norm = g.lpNorm<Eigen::Infinity>();

    result.gradient = r.lpNorm<Eigen::Infinity>();
    result.gradient_vec = r;
    result.cost = Fx;
    //std::cout << A.rows() << " " << A.cols() << " " << g_norm <<  std::endl;

    if(g_norm <= options.global_stopping_criteria) {
        found = true;

        std::cout << "First iteration residual was " << g_norm << " global stopping criteria was " << options.global_stopping_criteria << std::endl;
    }

    AScalar mu = options.damping*MaxJtJDiagonalValue(A);

    #ifdef ALUNA_USE_PARDISO
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<AScalar> > l_solver_llt;
    //Eigen::PardisoLLT<Eigen::SparseMatrix<AScalar> > l_solver_llt;
    Eigen::PardisoLU<Eigen::SparseMatrix<AScalar> > l_solver_lu;
    //Eigen::SparseLU<Eigen::SparseMatrix<AScalar> > l_solver_lu;
    //Eigen::UmfPackLU<Eigen::SparseMatrix<AScalar> > l_solver_lu;
    #else
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<AScalar>, Eigen::Lower > l_solver_llt;
    //Eigen::UmfPackLU<Eigen::SparseMatrix<AScalar> > l_solver_llt;
    //Eigen::SimplicialLLT<Eigen::SparseMatrix<AScalar> > l_solver_llt;
    // Eigen::UmfPackLU<Eigen::SparseMatrix<AScalar>> l_solver_lu;
    //Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrix<AScalar>> l_solver_lu;
    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<AScalar>, Eigen::Upper > l_solver_lu;
    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<AScalar> > l_solver_lu;
    // Eigen::SuperLU<Eigen::SparseMatrix<AScalar> > l_solver_lu;
    Eigen::SparseLU<Eigen::SparseMatrix<AScalar> > l_solver_lu;
    #endif

    Eigen::SparseMatrix<AScalar> Hm = A+mu*I;
    Hm.makeCompressed();

    if(options.solver_type == DN_SOLVER_LLT)
        l_solver_llt.analyzePattern(Hm);
    else if(options.solver_type == DN_SOLVER_LU)
        l_solver_lu.analyzePattern(Hm);

    while(!found && k<options.max_iterations)
    {
        ++k;
        result.n_iterations = k;

        std::chrono::high_resolution_clock::time_point time = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point absolute = std::chrono::high_resolution_clock::now();

        VectorXa ng = -g;
        VectorXa h;

        if(options.solver_type == DN_SOLVER_LLT)
        {
            Hm = A+mu*I;
            Hm.makeCompressed();

            l_solver_llt.analyzePattern(Hm);
        	l_solver_llt.factorize(Hm);


        	if(l_solver_llt.info() == Eigen::NumericalIssue) {
	            mu = mu*nu;
	            nu = 2.0*nu;

                //std::cout << "NON_PSD" << std::endl;

                //std::cout << mu << std::endl;

	            continue;
	        }
	        else
	        {
                if(!options.sherman_morrison && !options.woodbury)
                {
	        	  h = l_solver_llt.solve(ng);
                }
                else if(options.sherman_morrison)
                {
                    VectorXa u, v;
                    function->ShermanMorrisonVectors(u, v);

                    VectorXa ra = l_solver_llt.solve(ng);
                    VectorXa ua = l_solver_llt.solve(u);

                    AScalar dem = 1.0 + v.transpose()*ua;

                    h = ra - (v.transpose()*ra)*ua/dem;
                }
                else
                {
                   function->WoodburyMatrices(U, C, V);

                    VectorXa ba = l_solver_llt.solve(ng);

                    MatrixXa Ua(U.rows(), U.cols());
                    for(int i=0; i<U.cols(); ++i)
                        Ua.col(i) = l_solver_llt.solve(U.col(i));

                    VectorXa Vab = V*ba;

                    MatrixXa mid = C.inverse() + V*Ua;
                    MatrixXa mid_i = mid.inverse();

                    VectorXa right = Ua*mid_i*Vab;

                    h = ba - right;

                    AScalar res1 = (Hm*ba-ng).norm();

                    AScalar res = (Hm*h+(U*(C*(V*h)))-ng).norm();

                    if(res>1e-5)
                    {
                        mu = mu*nu;
                        nu = 2.0*nu;

                        std::cout << "Failed to solve system with residual " << res << " " << res1 << std::endl;
                        continue;
                    }
                }
	        }
        }
        else if(options.solver_type == DN_SOLVER_LU)
        {
        	Hm = A+mu*I;
            Hm.makeCompressed();

            // std::ofstream test("test.txt");
            // test << Hm << std::endl;
            // test.close();
            // exit(0);

            // l_solver_lu.analyzePattern(Hm);
            // l_solver_lu.factorize(Hm);
            l_solver_lu.compute(Hm);

            // std::cout << Hm.rows() << " " << Hm.cols() << " " << ng.rows() << std::endl;

            if(l_solver_lu.info() == Eigen::NumericalIssue) {
                mu = mu*nu;
                nu = 2.0*nu;

                std::cout << "Numerical Issue!" << std::endl;

                continue;
            }
            else
            {
                // check inertia
                int np = 0;
                int nn = 0;
                int nz = 0;
                double EPS_ZERO = 1e-3;
                double EPS_ZERO_PSD = 1e-6;
                
                // for (int i = 0; i < l_solver_lu.vectorD().size(); i++)
                // {
                //     double di = l_solver_lu.vectorD()(i);
                //     if (di < -EPS_ZERO_PSD)
                //     {
                //         std::cout << "Negative " << di << std::endl;
                //         nn++;
                //     }
                //     else if (fabs(di) < EPS_ZERO_PSD)
                //         nz++;
                //     else
                //         np++;
                // }
                // if(nz>0 || nn>options.n_constraints)
                // {
                //     std::cout << "Non-Definite system... Inertia: np=" << np << " nz=" << nz << " nn=" << nn << std::endl;

                //     mu = mu*nu;
                //     nu = 2.0*nu;

                //     continue;
                // }
                
                if(!options.sherman_morrison && !options.woodbury)
                {
                    h = l_solver_lu.solve(ng);


                    if((Hm*h-ng).norm() > EPS_ZERO)
                    {
                        std::cout << "Couldn't solve system" << std::endl;
                        std::cout << "Err = " << (Hm*h-ng).norm() << std::endl;

                        mu = mu*nu;
                        nu = 2.0*nu;

                        continue;
                    }
                }
                else if(options.sherman_morrison)
                {
                    VectorXa u, v;
                    function->ShermanMorrisonVectors(u, v);

                    VectorXa ra = l_solver_lu.solve(ng);
                    VectorXa ua = l_solver_lu.solve(u);

                    AScalar dem = 1.0 + v.transpose()*ua;

                    h = ra - (v.transpose()*ra)*ua/dem;
                }
                else
                {
                    MatrixXa U, C, V;
                    function->WoodburyMatrices(U, C, V);

                    VectorXa ba = l_solver_lu.solve(ng);

                    MatrixXa Ua(U.rows(), U.cols());
                    for(int i=0; i<U.cols(); ++i)
                        Ua.col(i) = l_solver_lu.solve(U.col(i));

                    VectorXa Vab = V*ba;

                    MatrixXa mid = C.inverse() + V*Ua;
                    MatrixXa mid_i = mid.inverse();

                    VectorXa right = Ua*mid_i*Vab;

                    h = ba - right;

                    AScalar res1 = (Hm*ba-ng).norm();

                    AScalar res = (Hm*h+(U*(C*(V*h)))-ng).norm();

                    if(res>1e-6)
                    {
                        mu = mu*nu;
                        nu = 2.0*nu;

                        std::cout << "Failed to solve system with residual " << res << " " << res1 << std::endl;
                        continue;
                    }
                }
            }
        }

        //VectorXa h = l_solver.solve(-r);

        //VectorXa h = -r*0.00000001;

        //test
        /*VectorXa g_test = g/g.norm();
        VectorXa r_test = r/r.norm();
        VectorXa h_test = h/h.norm();
        std::cout << "puta " << h_test.dot(r_test) << std::endl;*/

        //Eigen::SparseMatrix<AScalar> reg = A+mu*I;
        //VectorXa ng = -g;
        //VectorXa h = viennacl::linalg::solve(reg, ng, viennacl::linalg::bicgstab_tag());
        //VectorXa h = -0.00001*r;

        auto decomposer_time = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - time ).count();

        AScalar h_norm = h.lpNorm<Eigen::Infinity>(); AScalar x_norm = parameters.lpNorm<Eigen::Infinity>();

        if(h_norm <= options.change_stopping_criteria)
        {
            std::cout << k << " iteration change was " << h_norm << " stopping criteria was " << options.change_stopping_criteria*(x_norm+options.change_stopping_criteria) << std::endl;
            --result.n_iterations;
        }

        if(h_norm <= options.change_stopping_criteria)
        //if(h_norm <= options->change_stopping_criteria)
            found = true;
        else
        {
            VectorXa x_new(parameters.rows());

            function->TakeStep(h, parameters, x_new);

            time = std::chrono::high_resolution_clock::now();

            Eigen::SparseMatrix<AScalar> J_new;
            VectorXa r_new(parameters.rows());

            function->PreProcess(x_new);

            AScalar Fx_new;
            bool valid = true;
            try{

                if(!options.simplified)
    			    std::tie(Fx_new, r_new, J_new) = function->Evaluate(x_new);
                else
                    Fx_new = function->ComputeEnergy(x_new);
            }
            catch(TerminateNowException e)
            {
            	found = true;
            	valid = false;
            }
    		catch(std::exception e)
    		{
    			valid = false;
    		}


            AScalar L_dem = -h.dot(g)-0.5*(h.dot(A*h));
            if(options.woodbury)
                L_dem -= 0.5*(h.transpose()*U)*C*(V*h);

             AScalar gamma = (Fx-Fx_new+1e-6)/(L_dem);

            if(gamma>0 && options.simplified)
            {
                 r_new = function->ComputeGradient(x_new);
                 J_new = function->ComputeHessian(x_new);
            }

            auto evaluation_time = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - time ).count();

            auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - absolute ).count();

            AScalar gamma_g = r_new.dot(r_new) - r.dot(r);

            if(options.use_log) {
                std::cout << std::setw(10) << k << std::setw(20) << r.dot(h)/(r.norm()*h.norm()) << std::setw(20) << h_norm
                          << std::setw(20) << Fx << std::setw(20);     
                outputlog << std::setw(10) << k << std::setw(20) << r.dot(h)/(r.norm()*h.norm()) << std::setw(20) << h_norm
                          << std::setw(20) << Fx << std::setw(20);// << std::endl;;    
                if(valid){
                    std::cout << Fx_new;
                    outputlog << Fx_new;
                }
                else{
                    std::cout << "N/A";
                    outputlog << "N/A";
                }

                std::cout << std::setw(20) << r.lpNorm<Eigen::Infinity>() << std::setw(20);
                outputlog << std::setw(20) << r.lpNorm<Eigen::Infinity>() << std::setw(20);

                if(valid){
                    std::cout << r_new.lpNorm<Eigen::Infinity>();
                    outputlog << r_new.lpNorm<Eigen::Infinity>();
                }
                else{
                    std::cout << "N/A";
                    outputlog << "N/A";
                }
            }

            if(options.use_log && options.benchmark) {
                std::stringstream ss;
                ss << "D=" << decomposer_time/1000.0 << "s E="
                   << evaluation_time/1000.0 << "s T=" << total_time/1000.0 << "s";
                std::cout << std::setw(36) << ss.str();

                outputlog << std::setw(36) << ss.str();
            }

            //AScalar L_dem = 0.5*(h.dot(mu*h-g));
            
            //std::cout << L_dem << std::endl;
            //L_dem = 1.0;

            // std::cout << "gamma " << (Fx-Fx_new+1e-5) << " " << L_dem << std::endl;

            //std::cout << std::endl << gamma << " " << L_dem << std::endl;

            //VectorXa g_new = J_new.matrix.transpose()*r_new;
            //AScalar gamma = g_norm-g_new.dot(g_new);

            // std::cout << gamma << std::endl;
            if((gamma > 0 && valid) || (options.accept_everything && gamma_g>0 && valid))
                //if(valid)
            {
                //iterations_file << k << " " << Fx_new << " " << r_new.lpNorm<Eigen::Infinity>() << std::endl;

                function->AcceptStep();
                ++result.n_iterations_accepted;

                if(options.use_log){
                    std::cout << std::setw(20) << "ACCEPTED";
                    outputlog << std::setw(20) << "ACCEPTED";
                }

                parameters = x_new;

                r = r_new;
                Fx = Fx_new;
                A=J_new;
                g=r_new;

                g_norm = g.lpNorm<Eigen::Infinity>();

                result.gradient = r.lpNorm<Eigen::Infinity>();
                result.gradient_vec = r;
                result.cost = Fx;

                if(g_norm <= options.global_stopping_criteria)
                    found = true;

                // mu = mu*std::max(1.0/30.0,1.0-abs(std::pow((2.0*gamma-1.0),3)));
                mu = mu*std::max(AScalar(1.0/3.0),1.0-pow((2.0*gamma-1.0),3));
                //mu = 1e-6;
                // if(options.accept_everything)
                //     mu = 0.0;
                //mu = 1e-6;
                //if(gamma < 0)
                //    mu = options->damping*MaxJtJDiagonalValue(A);

                nu = 2.0;
            }
            else
            {
                function->RejectStep();

                if(options.use_log) {
                    if (!valid){
                        std::cout << std::setw(20) << "EXTERN REJECTED";
                        outputlog << std::setw(20) << "EXTERN REJECTED";
                    }
                    else{
                        std::cout << std::setw(20) << "REJECTED";
                        outputlog << std::setw(20) << "REJECTED";
                    }
                }

                mu = mu*nu;
                nu = 2.0*nu;
            }

            if(!options.check_matrix){
                std::cout << std::endl;
                outputlog << std::endl;
            }
            else {
                if (J_new.cols() == J_new.rows()) {
                    Eigen::SimplicialLLT<Eigen::SparseMatrix<AScalar> > llt(J_new);
                    //Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<AScalar> > llt(J_new.matrix);

                    if (options.use_log) {
                        if (llt.info() == Eigen::NumericalIssue)
                            std::cout << std::setw(20) << "NON-PSD WARNING" << std::endl;
                        else
                            std::cout << std::endl;
                    }
                } else {
                    if (options.use_log)
                        std::cout << std::setw(20) << "NON-SQUARE WARNING" << std::endl;
                }
            }
        }
    }

    function->Finalize(parameters);
    outputlog.close();

    return result;
}	
