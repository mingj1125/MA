#ifndef PBCEVALUATION_H
#define PBCEVALUATION_H


#include <utility> 
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>
#include <iomanip>


#include "VecMatDef.h"

class PBCevaluation
{
public:
	using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
	using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorXi = Vector<int, Eigen::Dynamic>;
	using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
	using VtxList = std::vector<int>;
	using StiffnessMatrix = Eigen::SparseMatrix<T>;
	using Entry = Eigen::Triplet<T>;
	using TV = Vector<T, 3>;
	using TV2 = Vector<T, 2>;
	using TM2 = Matrix<T, 2, 2>;
	using TV3 = Vector<T, 3>;
	using IV = Vector<int, 3>;
	using IV2 = Vector<int, 2>;
	using TM = Matrix<T, 3, 3>;

public:

	// kernel
    double std = 100;
	int num_directions = 12;

	// visualization
    MatrixXi visual_faces;
    MatrixXT visual_undeformed;
	std::string tag_file;
    std::string mesh_file;
	std::string mesh_info;
	VectorXi face_tags;
	VectorXT nu_visualization;
    VectorXT E_visualization;
	VectorXT kernel_coloring_prob; // coloring with kernel weights
    VectorXT kernel_coloring_avg;
	std::vector<TV> sample;
    std::vector<TV> direction;

	// underlying unit
	std::vector<TM2> unit_stress_tensors;
	std::vector<TM2> unit_strain_tensors;
	MatrixXi unit_faces;
    MatrixXT unit_undeformed;

	TV2 transformation;

	void initializeFromDir(std::string mesh_info_dir, std::string exp = "exp1/");
	void initializeVisualizationMesh(std::string filename);
	void visualizeMaterialProperties();
	void visualizeKernelWeighting();
	void initializeMeshInformation(std::string undeformed_mesh, std::string deformed_mesh);
	void setProbingLineDirections(unsigned int num_directions);
	Matrix<T, 3, 3> findBestStressTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions);
    Matrix<T, 2, 2> findBestStrainTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions);
	Vector<T, 3> computeWeightedStress(const TV sample_loc, TV direction);
    T computeWeightedStrain(const TV sample_loc, TV direction);
	Vector<T, 3> triangleCenterofMass(Matrix<T, 3, 3> vertices);
	std::vector<Matrix<T, 2, 2>> findCorrespondenceInUnit(TV2 position);
	int pointInTriangle(const TV2 sample_loc);
	int pointInVisualTriangle(const TV2 sample_loc);
	Vector<T,3> pointPosInVisualTriangle(const TV2 sample_loc);
	std::vector<Matrix<T, 2, 2>> returnStressTensors(int A);
    std::vector<Matrix<T, 2, 2>> returnStrainTensors(int A);

	Matrix<T, 3, 3> getVisualFaceVtxUndeformed(int face)
    {
        Matrix<T, 3, 3> cellx;
        Vector<int, 3> nodal_indices = visual_faces.row(face);
        for (int i = 0; i < 3; i++)
        {
            cellx.row(i) = visual_undeformed.row(nodal_indices[i]);
        }
        return cellx;
    }

	Matrix<T, 3, 3> getUnitFaceVtxUndeformed(int face)
    {
        Matrix<T, 3, 3> cellx;
        Vector<int, 3> nodal_indices = unit_faces.row(face);
        for (int i = 0; i < 3; i++)
        {
            cellx.row(i) = unit_undeformed.row(nodal_indices[i]);
        }
        return cellx;
    }

	std::vector<Matrix<T, 2, 2>> read_matrices(const std::string& filename) {
		std::vector<Matrix<T, 2, 2>> matrices;
		std::ifstream in_file(filename);
		if (!in_file) {
			std::cerr << "Error opening file for reading: " << filename << std::endl;
			return matrices;
		}
	
		T a, b, c, d;
		while (in_file >> a >> b >> c >> d) {
			Matrix<T, 2, 2> mat;
			mat << a, b, c, d;
			matrices.push_back(mat);
		}
	
		return matrices;
	}

	std::vector<T> read_local_properties(const std::string& filename) {
		std::vector<T> properties;
		std::ifstream in_file(filename);
		if (!in_file) {
			std::cerr << "Error opening file for reading: " << filename << std::endl;
			return properties;
		}
	
		T a;
		while (in_file >> a) {
			properties.push_back(a);
		}
	
		return properties;
	}

	PBCevaluation() {} 
	~PBCevaluation() {} 
};


#endif
