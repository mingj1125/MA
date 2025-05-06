#include "../include/enforce_matrix_constraints.h"

#include <iostream>

std::vector<Eigen::Triplet<AScalar>> SparseMatrixToTriplets(const Eigen::SparseMatrix<AScalar>& A)
{
	std::vector<Eigen::Triplet<AScalar> > triplets;

	for (int k=0; k < A.outerSize(); ++k)
        for (Eigen::SparseMatrix<AScalar>::InnerIterator it(A,k); it; ++it)
        	triplets.push_back(Eigen::Triplet<AScalar>(it.row(), it.col(), it.value()));

    return triplets;
}

Eigen::SparseMatrix<AScalar> EnforceSquareMatrixConstraints(Eigen::SparseMatrix<AScalar>& old, std::vector<int>& constraints, bool fill_ones)
{

	std::vector<Eigen::Triplet<AScalar>> triplets = SparseMatrixToTriplets(old);

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

Eigen::SparseMatrix<AScalar> ComputeAffineConstrainedHessian(const Eigen::SparseMatrix<AScalar>& H, const VectorXa& x, int cut_off)
{
	if(cut_off == 0)
		cut_off = H.rows()/3;

    std::vector<Eigen::Triplet<AScalar>> triplets = SparseMatrixToTriplets(H);

    Eigen::SparseMatrix<AScalar> Hc(H.rows()+6, H.cols()+6);

    int nx = x.rows();
    int m_nv = x.rows() / 3;

    int rows = H.rows();

    //Translations
    for (int i = 0; i < cut_off; i++)
    {
        triplets.push_back(Eigen::Triplet(rows + 0, 3 * i, -1.0));
        triplets.push_back(Eigen::Triplet(rows + 1, 3 * i + 1, -1.0));
        triplets.push_back(Eigen::Triplet(rows + 2, 3 * i + 2, -1.0));
        triplets.push_back(Eigen::Triplet(3 * i, rows  + 0, -1.0));
        triplets.push_back(Eigen::Triplet(3 * i + 1, rows  + 1, -1.0));
        triplets.push_back(Eigen::Triplet(3 * i + 2, rows  + 2, -1.0));
    }

    // Rotations
    for (int i = 0; i < cut_off; i++)
    {
        Vector3a xi = x.template segment<3>(i*3);
        triplets.push_back(Eigen::Triplet(rows + 3, 3 * i + 1, -xi(2)));
        triplets.push_back(Eigen::Triplet(rows + 3, 3 * i + 2, xi(1)));
        triplets.push_back(Eigen::Triplet(rows + 4, 3 * i + 0, xi(2)));
        triplets.push_back(Eigen::Triplet(rows + 4, 3 * i + 2, -xi(0)));
        triplets.push_back(Eigen::Triplet(rows + 5, 3 * i + 0, -xi(1)));
        triplets.push_back(Eigen::Triplet(rows + 5, 3 * i + 1, xi(0)));

        //symmetric part
        triplets.push_back(Eigen::Triplet(3 * i + 1, rows + 3, -xi(2)));
        triplets.push_back(Eigen::Triplet(3 * i + 2, rows + 3, xi(1)));
        triplets.push_back(Eigen::Triplet(3 * i + 0, rows + 4, xi(2)));
        triplets.push_back(Eigen::Triplet(3 * i + 2, rows + 4, -xi(0)));
        triplets.push_back(Eigen::Triplet(3 * i + 0, rows + 5, -xi(1)));
        triplets.push_back(Eigen::Triplet(3 * i + 1, rows + 5, xi(0)));

    }

    Hc.setFromTriplets(triplets.begin(), triplets.end());

    return Hc;
}

Eigen::SparseMatrix<AScalar> ComputeFakeAffineConstrainedHessian(const Eigen::SparseMatrix<AScalar>& H, const VectorXa& x)
{
    std::vector<Eigen::Triplet<AScalar>> triplets = SparseMatrixToTriplets(H);

    Eigen::SparseMatrix<AScalar> Hc(H.rows()+6, H.cols()+6);

    int nx = x.rows();
    int m_nv = x.rows() / 3;

    //Translations

    triplets.push_back(Eigen::Triplet(3 * m_nv + 0, 3 * m_nv + 0, 1.0));
    triplets.push_back(Eigen::Triplet(3 * m_nv + 1, 3 * m_nv + 1, 1.0));
    triplets.push_back(Eigen::Triplet(3 * m_nv + 2, 3 * m_nv + 2, 1.0));

    // Rotations
    
    triplets.push_back(Eigen::Triplet(3 * m_nv + 3, 3 * m_nv + 3, 1.0));
    triplets.push_back(Eigen::Triplet(3 * m_nv + 4, 3 * m_nv + 4, 1.0));
    triplets.push_back(Eigen::Triplet(3 * m_nv + 5, 3 * m_nv + 5, 1.0));

    Hc.setFromTriplets(triplets.begin(), triplets.end());

    return Hc;
}

Eigen::SparseMatrix<AScalar> EnforceRectangularMatrixConstraints(Eigen::SparseMatrix<AScalar>& old, std::vector<int>& constraints, RectangularMatrixConstraintType type)
{
	std::vector<Eigen::Triplet<AScalar>> triplets = SparseMatrixToTriplets(old);

	std::vector<Eigen::Triplet<AScalar>> new_triplets;

	if(type==CONSTRAIN_ROWS)
	{
		std::vector<bool> constrained(old.rows(), false);
		for(int i=0; i<constraints.size(); ++i)
			constrained[constraints[i]] = true;

		for(int i=0; i<triplets.size(); ++i)
		{
			if(!constrained[triplets[i].row()])
				new_triplets.push_back(triplets[i]);
		}
	}
	else if(type==CONSTRAIN_COLUMNS)
	{
		std::vector<bool> constrained(old.cols(), false);
		for(int i=0; i<constraints.size(); ++i)
			constrained[constraints[i]] = true;

		for(int i=0; i<triplets.size(); ++i)
		{
			if(!constrained[triplets[i].col()] )
				new_triplets.push_back(triplets[i]);
		}
	}

	Eigen::SparseMatrix<AScalar> new_matrix(old.rows(), old.cols());
	new_matrix.setFromTriplets(new_triplets.begin(), new_triplets.end());

	return new_matrix;
}