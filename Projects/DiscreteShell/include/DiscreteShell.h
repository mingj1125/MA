#ifndef DISCRETE_SHELL_H
#define DISCRETE_SHELL_H

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

class DiscreteShell
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using VtxList = std::vector<int>;
    using TV = Vector<T, 3>;
    using TV2 = Vector<T, 2>;
    using TM2 = Matrix<T, 2, 2>;
    using TV3 = Vector<T, 3>;
    using IV = Vector<int, 3>;
    using IV2 = Vector<int, 2>;
    using TM = Matrix<T, 3, 3>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using Face = Vector<int, 3>;
    using Triangle = Vector<int, 3>;
    using Edge = Vector<int, 2>;

    using Hinges = Matrix<int, Eigen::Dynamic, 4>;
    
    using FaceVtx = Matrix<T, 3, 3>;
    using FaceIdx = Vector<int, 3>;
    using HingeIdx = Vector<int, 4>;
    using HingeVtx = Matrix<T, 4, 3>;

 
public:
    // material parameter setup1
    T density = 1.5e3;  //Cotton kg/m^3
    TV gravity = TV(0.0, -9.8, 0.0);    
    T E = 1e6;
    T nu = 0.45;

    T lambda, mu;

    Hinges hinges;
    T thickness = 0.003; // meter
    VectorXi faces;
    std::vector<Triangle> triangles;

    VectorXT hinge_stiffness;

    VectorXT deformed, undeformed;
    VectorXT u;
    VectorXT external_force;

    bool run_diff_test = false;
    int max_newton_iter = 500;
    bool use_Newton = true;
    bool add_gravity = false;
    bool jump_out = true;
    bool verbose = false;
    bool use_consistent_mass_matrix = true;
    T newton_tol = 1e-6;

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
    std::vector<TM2> space_strain_tensors; // local 2D coodinate strain without transformation
    std::vector<TM2> optimization_homo_target_tensors; // optimization target

    // ============================= Test Window Utilities ==============================
    bool set_window = false;
    // nodal coordinate
    int window_x = 0;
    int window_y = 0;
    int window_length = 5;
    int window_height = 5;
    T total_window_area_undeformed = 0;

    std::vector<TM> homo_stress_tensors;
    std::vector<T> homo_stress_magnitudes;

    std::vector<TM> homo_strain_tensors;
    std::vector<T> homo_strain_magnitudes;

    VectorXT weights = VectorXT::Zero(4);
    VectorXT epsilons = VectorXT::Zero(4);

    // ============================= Kernel-weighted Cut Utilities ======================
    VectorXi cut_coloring; // coloring with cut categories
    VectorXT kernel_coloring_prob; // coloring with kernel weights
    VectorXT kernel_coloring_avg;
    std::vector<TV> sample;
    std::vector<TV> direction;

    // ============================= Heterogenuous material ==============================
    bool heterogenuous = true;
    VectorXT nu_visualization;
    VectorXT E_visualization;

    // ============================= Quadratic triangle ===================================
    bool quadratic = false;
    

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

    template<int dim = 2>
    void getSubVector(const VectorXT& _vector, 
        const std::vector<int>& vtx_idx, 
        VectorXT& sub_vec, int shift = 0)
    {
        sub_vec.resize(vtx_idx.size() * dim);
        sub_vec.setZero();
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            sub_vec.template segment<dim>(i * dim) = _vector.template segment<dim>(vtx_idx[i] * dim + shift);
        }
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

    HingeVtx getHingeVtxDeformed(const HingeIdx& hi)
    {
        HingeVtx cellx;
        for (int i = 0; i < 4; i++)
        {
            cellx.row(i) = deformed.segment<3>(hi[i]*3);
        }
        return cellx;
    }

    HingeVtx getHingeVtxUndeformed(const HingeIdx& hi)
    {
        HingeVtx cellx;
        for (int i = 0; i < 4; i++)
        {
            cellx.row(i) = undeformed.segment<3>(hi[i]*3);
        }
        return cellx;
    }

    FaceVtx getFaceVtxDeformed(int face)
    {
        FaceVtx cellx;
        FaceIdx nodal_indices = faces.segment<3>(face * 3);
        for (int i = 0; i < 3; i++)
        {
            cellx.row(i) = deformed.segment<3>(nodal_indices[i]*3);
        }
        return cellx;
    }

    FaceVtx getFaceVtxUndeformed(int face)
    {
        FaceVtx cellx;
        FaceIdx nodal_indices = faces.segment<3>(face * 3);
        for (int i = 0; i < 3; i++)
        {
            cellx.row(i) = undeformed.segment<3>(nodal_indices[i]*3);
        }
        return cellx;
    }

    template <typename OP>
    void iterateFaceSerial(const OP& f)
    {
        for (int i = 0; i < faces.rows()/3; i++)
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
        tbb::parallel_for(0, int(faces.rows()/3), [&](int i)
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

    template <typename OP>
    void iterateHingeSerial(const OP& f)
    {
        for (int i = 0; i < hinges.rows(); i++)
        {
            const Vector<int, 4> nodes = hinges.row(i);
            f(nodes, i);   
        }
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

    void computeLinearModes(MatrixXT& eigen_vectors, VectorXT& eigen_values);
    void initializeFromFile(const std::string& filename);
    void buildHingeStructure();
    int nFaces () { return faces.rows() / 3; }
    
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
    
    // stretching and bending energy
    virtual void addShellInplaneEnergy(T& energy);
    virtual void addShellBendingEnergy(T& energy);
    virtual void addShellInplaneForceEntries(VectorXT& residual);
    virtual void addShellBendingForceEntries(VectorXT& residual);
    virtual void addShellInplaneHessianEntries(std::vector<Entry>& entries);
    virtual void addShellBendingHessianEntries(std::vector<Entry>& entries);

    // graviational energy
    void addShellGravitionEnergy(T& energy);
    void addShellGravitionForceEntry(VectorXT& residual);
    void addShellGravitionHessianEntry(std::vector<Entry>& entries);

    void computeBoundingBox(TV& min_corner, TV& max_corner);

    void setHingeStiffness();

    // derivative tests
    void checkTotalGradient(bool perturb = false);
    void checkTotalGradientScale(bool perturb = false);
    void checkTotalHessian(bool perturb = false);
    void checkTotalHessianScale(bool perturb = false);

    // ============================= Stress Tensor Utilities ============================
    Matrix<T, 3, 2> computeDeformationGradientwrt2DXSpace(const TV X1, const TV X2, const TV X3, const TV x1, const TV x2, const TV x3);
    Matrix<T, 2, 3> computeBarycentricJacobian(const TV X1, const TV X2, const TV X3);
    Matrix<T, 2, 2> computeCauchyStrainwrt2dXSpace(Matrix<T, 3, 2> F);
    void computeStrainAndStressPerElement();
    void computeHomogenization();
    void computeEnergyComparison();

    // ============================= Test Window Utilities ==============================
    Matrix<T, 3, 3> computeHomogenisedStressTensorinWindow();
    T computeWindowAreaRatio();
    Matrix<T, 3, 3> computeHomogenisedStrainTensorinWindow();
    Matrix<T, 3, 3> computeHomogenisedTargetTensorinWindow();
    bool TriangleinsideWindow(int start_x, int start_y, int length, int height, int face_idx);
    // compute objective Obj = w1*(epsilon_xx - ep_t_xx)^2 + w2*(epsilon_xy - ep_t_xy)^2 + w3*(epsilon_yy - ep_t_yy)^2
    void addEnergyforDesiredTarget(T &energy);
    Vector<T, 9> computeGradientForDesiredTargetXX(const TV X1, const TV X2, const TV X3, const TV x1, const TV x2, const TV x3);
    Vector<T, 9> computeGradientForDesiredTargetXY(const TV X1, const TV X2, const TV X3, const TV x1, const TV x2, const TV x3);
    Vector<T, 9> computeGradientForDesiredTargetYY(const TV X1, const TV X2, const TV X3, const TV x1, const TV x2, const TV x3);
    Vector<T, 9> computeGradientForDesiredTargetYX(const TV X1, const TV X2, const TV X3, const TV x1, const TV x2, const TV x3);
    void addGradientForDesiredTarget(VectorXT& residual);
    T computeTestWindowForces(VectorXT& forces);

    // ============================= Boundary Utilities =================================
    void setEssentialBoundaryCondition(T displacement_x, T displacement_y);

    // ============================= Kernel-weighted Cut Utilities ======================
    void setProbingLineDirections(unsigned int num_directions);
    Matrix<T, 3, 3> findBestStressTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions);
    Matrix<T, 2, 2> findBestStrainTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions);
    Matrix<T, 3, 3> findBestStressTensorviaAveraging(const TV sample_loc);
    Matrix<T, 3, 3> findBestStrainTensorviaAveraging(const TV sample_loc);
    Vector<T, 3> computeWeightedStress(const TV sample_loc, TV direction);
    T computeWeightedStrain(const TV sample_loc, TV direction);
    Vector<T, 3> triangleCenterofMass(FaceVtx vertices);
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
    std::vector<Matrix<T, 3, 3>> returnStressTensors(int A);
    std::vector<Matrix<T, 3, 3>> returnStrainTensors(int A);
    void testSharedEdgeStress(int A, int B, int v1, int v2);
    void testStressTensors(int A, int B);
    T areaRatio(int A);
    int pointInTriangle(const TV sample_loc);
    std::vector<Vector<T, 3>> pointInDeformedTriangle();
    Vector<T, 3> pointInDeformedTriangle(const TV sample_loc);

    void setMaterialParameter(T& E, T& nu, int face_idx);

public:
    DiscreteShell() 
    {
        updateLameParameters();
    }
    ~DiscreteShell() {}
};

#endif