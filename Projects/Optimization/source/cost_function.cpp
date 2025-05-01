#include "../include/cost_function.h"

#include <iostream>

void CostFunction::TakeStep(const VectorXa& step, const VectorXa& prev_parameters, VectorXa& new_parameters)
{
    new_parameters = prev_parameters + step;
}

void CostFunction::TakePartialStep(AScalar multiplier)
{

}

void CostFunction::AcceptStep()
{

}

void CostFunction::RejectStep()
{

}

void CostFunction::Finalize(const VectorXa& parameters)
{

}

void CostFunction::PreProcess(VectorXa& parameters)
{

}

void CostFunction::ShermanMorrisonVectors(VectorXa& u, VectorXa& v)
{

}

void CostFunction::WoodburyMatrices(MatrixXa& U, MatrixXa& C, MatrixXa& V)
{

}

reduced_cost_evaluation CostFunction::EvaluateReduced(const VectorXa& parameters)
{
	return reduced_cost_evaluation();
}

AScalar CostFunction::ComputeEnergy(const VectorXa& parameters)
{
	return 0;
}

VectorXa CostFunction::ComputeGradient(const VectorXa& parameters)
{
	return VectorXa();
}

Eigen::SparseMatrix<AScalar> CostFunction::ComputeHessian(const VectorXa& parameters)
{
	return Eigen::SparseMatrix<AScalar>();
}

Eigen::SparseMatrix<AScalar> CostFunction::InexpensiveHessian()
{
	return Eigen::SparseMatrix<AScalar>();
}

Eigen::SparseMatrix<AScalar> CostFunction::ComputeJacobiPreconditioner()
{
	return Eigen::SparseMatrix<AScalar>();
}

void CostFunction::TestGradient(VectorXa& parameters)
{
	AScalar cost;
	VectorXa gradient;
	Eigen::SparseMatrix<AScalar> hessian;
	std::tie(cost, gradient, hessian) = Evaluate(parameters);
	AcceptStep();

	VectorXa dummy_g;
	Eigen::SparseMatrix<AScalar> dummy_h;

	AScalar h =1e-6;
	for(int i=0; i<parameters.rows(); ++i)
	{
		parameters[i]-=h;
		AScalar minus; std::tie(minus, dummy_g, dummy_h) = Evaluate(parameters);

		parameters[i]+=2.0*h;
		AScalar plus; std::tie(plus, dummy_g, dummy_h) = Evaluate(parameters);

		parameters[i]-=h;

		//if((gradient[i]-(plus-minus)/(2.0*h))>1e-2)
			{
				std::cout << i << " " << parameters[i]-h << "," << parameters[i] << "," << parameters[i]+h << " " << minus << "," << plus << " analytical: " << gradient[i] << " numerical: " << (plus-minus)/(2.0*h) << std::endl;
				//std::cout << "Gradient is WROOOOOOONG!!!!" << std::endl;
			}
	}
}

void CostFunction::TestGradient(VectorXa& parameters, int offset)
{
	AScalar cost;
	VectorXa gradient;
	Eigen::SparseMatrix<AScalar> hessian;
	std::tie(cost, gradient, hessian) = Evaluate(parameters);
	AcceptStep();

	VectorXa dummy_g;
	Eigen::SparseMatrix<AScalar> dummy_h;

	AScalar h = 1e-5;
	for(int i=offset; i<parameters.rows(); ++i)
	{
		parameters[i]-=h;
		AScalar minus; std::tie(minus, dummy_g, dummy_h) = Evaluate(parameters);

		parameters[i]+=2.0*h;
		AScalar plus; std::tie(plus, dummy_g, dummy_h) = Evaluate(parameters);

		parameters[i]-=h;

		//if(abs(gradient[i]-(plus-minus)/(2.0*h))>1e-5)
			{
				std::cout << i << ": " << parameters[i]-h << "," << parameters[i] << "," << parameters[i]+h << " " << minus << "," << plus << " --------------------------- " << gradient[i] << " " << (plus-minus)/(2.0*h) << std::endl;
			//	std::cout << "Gradient is WROOOOOOONG!!!!" << std::endl;
				//exit(0);
			}
	}
}

void CostFunction::TestHessian(VectorXa& parameters, int offset)
{
	// MatrixXa U, C, V;
	// WoodburyMatrices(U, C, V);

	AScalar cost;
	VectorXa gradient;
	Eigen::SparseMatrix<AScalar> hessian;
	std::tie(cost, gradient, hessian) = Evaluate(parameters);

	AScalar dummy_c;
	Eigen::SparseMatrix<AScalar> dummy_h;

	AScalar h =1e-4;
	for(int i=offset; i<parameters.rows(); ++i)
	{
		parameters[i]-=h;
		VectorXa minus; std::tie(dummy_c, minus, dummy_h) = Evaluate(parameters);

		parameters[i]+=2.0*h;
		VectorXa plus; std::tie(dummy_c, plus, dummy_h) = Evaluate(parameters);

		parameters[i]-=h;

		VectorXa analytical = hessian.col(i);//+U*C*V.col(i);
		VectorXa numerical = (plus-minus)/(2.0*h);

		for(int j=0; j<analytical.rows(); ++j)
			if(abs(numerical[j])>1e-3)
			{
				//if(abs(analytical[j]-numerical[j])>1e-2)
					std::cout << j << ": " << analytical[j] << " " << numerical[j] << std::endl;
			}
		std::cout << std::endl;
	}
}
