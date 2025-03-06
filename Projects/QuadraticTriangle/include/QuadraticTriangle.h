#ifndef QUADRATIC_TRIANGLE_H
#define QUADRATIC_TRIANGLE_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>
#include <complex>
#include <iomanip>

#include "VecMatDef.h"
#include "Timer.h"
#include "Util.h"

class QuadraticTriangle
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using TV = Vector<T, 3>;
    using TV2 = Vector<T, 2>;
    using TM2 = Matrix<T, 2, 2>;
    using TV3 = Vector<T, 3>;
    using TM = Matrix<T, 3, 3>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using Triangle = Vector<int, 3>;

    using FaceVtx = Matrix<T, 3, 3>;
    using FaceIdx = Vector<int, 3>;

 
public:
    // material parameter setup1
    T density = 1.5e3;  //Cotton kg/m^3
    TV gravity = TV(0.0, -9.8, 0.0);    
    T E = 1e6;
    T nu = 0.45;
    T E_default = 1e6;
    float nu_default = 0.;
    float graded_k = 0.5;
    double std = 0.001;

    T lambda, mu;
    int mesh_nodes;

    T thickness = 0.003; // meter
    MatrixXi faces;
    std::vector<Triangle> triangles;

    VectorXT deformed, undeformed;
    VectorXT u;
    VectorXT external_force;

    int max_newton_iter = 500;
    bool use_Newton = true;
    bool add_gravity = false;
    bool use_consistent_mass_matrix = true;
    T newton_tol = 1e-9;
    bool verbose = false;

    std::vector<T> residual_norms;
    std::unordered_map<int, T> dirichlet_data;

    bool dynamics = false;
    T dt = 0.01;
    T simulation_duration = 10;
    StiffnessMatrix M;
    VectorXT mass_diagonal;
    VectorXT xn;
    VectorXT vn;

    // ============================= Stress Tensor Utilities ============================

    std::vector<TM> stress_tensors;
    std::vector<TM> cauchy_stress_tensors;

    // ============================= Strain Tensor Utilities ============================

    std::vector<TM> strain_tensors;
    std::vector<TM> defomation_gradients;

    // ============================= Kernel-weighted Cut Utilities ======================
    VectorXi cut_coloring; // coloring with cut categories
    VectorXT kernel_coloring_prob; // coloring with kernel weights
    VectorXT kernel_coloring_avg;
    std::vector<TV> sample;
    std::vector<TV> direction;

    // ============================= Heterogenuous material ==============================
    bool graded = false;
    bool tags = true;
    VectorXi face_tags;
    VectorXT nu_visualization;
    VectorXT E_visualization;

    // ============================= Stiffness tensor ====================================
    bool set_boundary_condition = true;
    

public:
    template <class OP>
    void iterateDirichletDoF(const OP& f) 
    {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    template<int dim = 2>
    void addForceEntry(VectorXT& residual, 
        const std::vector<int>& vtx_idx, 
        const VectorXT& gradent, int shift = 0)
    {
        for (int i = 0; i < vtx_idx.size(); i++)
            residual.template segment<dim>(vtx_idx[i] * dim + shift) += gradent.template segment<dim>(i * dim);
    }

    
    template<int dim_row=2, int dim_col=2>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const MatrixXT& hessian, 
        int shift_row = 0, int shift_col=0)
    {
        
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < dim_row; k++)
                    for (int l = 0; l < dim_col; l++)
                        triplets.emplace_back(
                                dof_i * dim_row + k + shift_row, 
                                dof_j * dim_col + l + shift_col, 
                                hessian(i * dim_row + k, j * dim_col + l)
                            );                
            }
        }
    }

    template<int dim_row=2, int dim_col=2>
    void addJacobianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx,
        const std::vector<int>& vtx_idx2, 
        const MatrixXT& jacobian, 
        int shift_row = 0, int shift_col=0)
    {
        
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx2.size(); j++)
            {
                int dof_j = vtx_idx2[j];
                for (int k = 0; k < dim_row; k++)
                    for (int l = 0; l < dim_col; l++)
                        triplets.emplace_back(
                                dof_i * dim_row + k + shift_row, 
                                dof_j * dim_col + l + shift_col, 
                                jacobian(i * dim_row + k, j * dim_col + l)
                            ); 
            }
        }
    }

    template<int dim_row=2, int dim_col=2>
    void addHessianMatrixEntry(
        MatrixXT& matrix_global,
        const std::vector<int>& vtx_idx, 
        const MatrixXT& hessian,
        int shift_row = 0, int shift_col=0)
    {
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < dim_row; k++)
                    for (int l = 0; l < dim_col; l++)
                    {
                        matrix_global(dof_i * dim_row + k + shift_row, 
                        dof_j * dim_col + l + shift_col) 
                            += hessian(i * dim_row + k, j * dim_col + l);
                    }
            }
        }
    }

    template<int dim_row=2, int dim_col=2>
    void addJacobianMatrixEntry(
        MatrixXT& matrix_global,
        const std::vector<int>& vtx_idx, 
        const std::vector<int>& vtx_idx2, 
        const MatrixXT& hessian,
        int shift_row = 0, int shift_col=0)
    {
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx2.size(); j++)
            {
                int dof_j = vtx_idx2[j];
                for (int k = 0; k < dim_row; k++)
                    for (int l = 0; l < dim_col; l++)
                    {
                        matrix_global(dof_i * dim_row + k + shift_row, 
                        dof_j * dim_col + l + shift_col) 
                            += hessian(i * dim_row + k, j * dim_col + l);
                    }
            }
        }
    }

    template<int dim0=2, int dim1=2>
    void addHessianBlock(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const MatrixXT& hessian_block,
        int shift_row = 0, int shift_col=0)
    {

        int dof_i = vtx_idx[0];
        int dof_j = vtx_idx[1];
        
        for (int k = 0; k < dim0; k++)
            for (int l = 0; l < dim1; l++)
            {
                triplets.emplace_back(dof_i * dim0 + k + shift_row, 
                    dof_j * dim1 + l + shift_col, 
                    hessian_block(k, l));
            }
    }

    template<int size>
    VectorXT computeHessianBlockEigenValues(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        return eigenSolver.eigenvalues();
    }

    std::vector<Entry> entriesFromSparseMatrix(const StiffnessMatrix &A) 
    {
        std::vector<Entry> triplets;
        for (int k = 0; k < A.outerSize(); ++k)
            for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
                triplets.emplace_back(it.row(), it.col(), it.value());
        return triplets;
    }


    // ====================== discrete shell ==========================
    void updateLameParameters()
    {
        lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        mu = E / 2.0 / (1.0 + nu);
    }

    Matrix<T, 6, 3> getFaceVtxDeformed(int face)
    {
        Matrix<T, 6, 3> cellx;
        Vector<int, 6> nodal_indices = faces.row(face);
        for (int i = 0; i < 6; i++)
        {
            cellx.row(i) = deformed.segment<3>(nodal_indices[i]*3);
        }
        return cellx;
    }

    Matrix<T, 6, 3> getFaceVtxUndeformed(int face)
    {
        Matrix<T, 6, 3> cellx;
        Vector<int, 6> nodal_indices = faces.row(face);
        for (int i = 0; i < 6; i++)
        {
            cellx.row(i) = undeformed.segment<3>(nodal_indices[i]*3);
        }
        return cellx;
    }

    template <typename OP>
    void iterateFaceSerial(const OP& f)
    {
        for (int i = 0; i < faces.rows(); i++)
            f(i);
    }

    template <typename OP>
    void iterateNodeSerial(const OP& f)
    {
        for (int i = 0; i < deformed.rows()/3; i++)
            f(i);
    }

    template <typename OP>
    void iterateFaceParallel(const OP& f)
    {
        tbb::parallel_for(0, int(faces.rows()), [&](int i)
        {
            f(i);
        });
    }

    template <typename OP>
    void iterateTriangleSerial(const OP& f)
    {
        for (int i = 0; i < triangles.size(); i++)
            f(triangles[i], i);
    }

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);
    
    // ================================================================

public:
    
    bool advanceOneStep(int step);
    bool advanceOneTimeStep();
    bool linearSolve(StiffnessMatrix& K, const VectorXT& residual, VectorXT& du);
    T computeTotalEnergy();
    T computeResidual(VectorXT& residual);
    void buildSystemMatrix(StiffnessMatrix& K);
    T lineSearchNewton(const VectorXT& residual);

    void initializeFromFile(const std::string& filename);
    int nFaces () { return faces.rows(); }
    
    void addShellEnergy(T& energy);
    void addShellForceEntry(VectorXT& residual);
    void addShellHessianEntries(std::vector<Entry>& entries);

    // virtual function for different time integration schemes
    //                      different ways of computing the mass matrix
    virtual void addInertialEnergy(T& energy);
    virtual void addInertialForceEntry(VectorXT& residual);
    virtual void addInertialHessianEntries(std::vector<Entry>& entries);
    virtual void updateDynamicStates();
    virtual void initializeDynamicStates();
    virtual void computeMassMatrix();
    virtual void computeConsistentMassMatrix(const FaceVtx& p, Matrix<T, 9, 9>& mass_mat);
    
    // stretching energy
    virtual void addShellInplaneEnergy(T& energy);
    virtual void addShellInplaneForceEntries(VectorXT& residual);
    virtual void addShellInplaneHessianEntries(std::vector<Entry>& entries);

    // graviational energy
    void addShellGravitionEnergy(T& energy);
    void addShellGravitionForceEntry(VectorXT& residual);
    void addShellGravitionHessianEntry(std::vector<Entry>& entries);

    void computeBoundingBox(TV& min_corner, TV& max_corner);

    // ============================= Stress Tensor Utilities ============================
    void computeStrainAndStressPerElement();
    Vector<T, 4> evaluatePerTriangleStress(const Matrix<T, 6, 3> vertices, const Matrix<T, 6, 3> undeformed_vertices, 
        const TV cut_point_coordinate, const TV direction_normal, const TV sample_loc, int face_idx);
    Vector<T, 2> evaluatePerTriangleStrain(const Matrix<T, 6, 3> vertices, const Matrix<T, 6, 3> undeformed_vertices, 
        const TV cut_point_coordinate, const TV direction, const TV sample_loc, int face_idx);

    // ============================= Kernel-weighted Cut Utilities ======================
    void setProbingLineDirections(unsigned int num_directions);
    Matrix<T, 3, 3> findBestStressTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions);
    Matrix<T, 2, 2> findBestStrainTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions);
    Matrix<T, 3, 3> findBestStressTensorviaAveraging(const TV sample_loc);
    Matrix<T, 3, 3> findBestStrainTensorviaAveraging(const TV sample_loc);
    Vector<T, 3> computeWeightedStress(const TV sample_loc, TV direction);
    T computeWeightedStrain(const TV sample_loc, TV direction);
    void visualizeCuts(const std::vector<TV> sample_points, const std::vector<TV> line_directions);
    void visualizeCut(const TV sample_point, const TV line_direction, unsigned int line_tag);
    bool lineCutTriangle(const TV x1, const TV x2, const TV x3, const TV sample_point, const TV line_direction, TV &cut_point_coordinate);
    Vector<T, 3> middlePointoflineCutTriangle(const TV x1, const TV x2, const TV x3, const TV cut_point_coordinate);
    T strainInCut(const int face_idx, const TV cut_point_coordinate);
    Vector<T, 2> solveLineIntersection(const TV sample_point, const TV line_direction, const TV v1, const TV v2);

    // ============================== Isotropic Stretch =================================
    void testIsotropicStretch();
    void testHorizontalDirectionStretch();
    void testVerticalDirectionStretch();
    void testSharedEdgeStress(int A, int B, int v1, int v2);
    void testStressTensors(int A, int B);

    // ============================== Native functionality ==============================
    std::vector<Matrix<T, 3, 3>> returnStressTensors(int A);
    std::vector<Matrix<T, 3, 3>> returnStrainTensors(int A);

    Vector<T, 3> triangleCenterofMass(FaceVtx vertices);
    int pointInTriangle(const TV sample_loc);
    std::vector<Vector<T, 3>> pointInDeformedTriangle();
    Vector<T, 3> pointInDeformedTriangle(const TV sample_loc);
    Vector<T, 2> findBarycentricCoord(const TV X, const Matrix<T,6,3> undeformed_vertices);

    // =============================== Quadratic Energy ==================================
    Matrix<T, 6, 1> get_shape_function(T beta_1, T beta_2);
    Vector<T, 18> compute2DQuadraticShellEnergyGradient(const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices, int face_idx);
    T compute2DQuadraticShellEnergy(const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices, int face_idx);	
    Matrix<T, 18, 18> compute2DQuadraticShellEnergyHessian(const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices, int face_idx);		
    Matrix<T, 2, 2> compute2DDeformationGradient(const Matrix<T,6,3> & vertices, const Matrix<T,6,3> & undeformed_vertices, const Vector<T, 2> beta);
    void setMaterialParameter(T& E, T& nu, T& local_lambda, T& local_mu, TV X, int face_idx);

public:
    QuadraticTriangle() 
    {
        updateLameParameters();
    }
    QuadraticTriangle(float nu_default_s, float graded_k_s, float std_s): nu_default(nu_default_s), graded_k(graded_k_s), std(std_s)
    {}
    ~QuadraticTriangle() {}
};

#endif