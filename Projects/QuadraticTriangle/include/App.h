#ifndef APP_H
#define APP_H

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

template<class Simulation>
class App
{
public:
    Simulation& simulation;
    int static_solve_step = 0;

    polyscope::SurfaceMesh* psMesh;
    polyscope::PointCloud* psCloud;
    polyscope::CurveNetwork* psLine;

    T t = 0;

    bool animate_modes = false;
    bool run_sim = false;
    bool show_geometry = false;
    int modes = 0;
    Eigen::MatrixXd eigen_vectors;
    Eigen::VectorXd eigen_values;

    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;

public:
    void initializeScene()
    {
        polyscope::options::autocenterStructures = true;
        polyscope::view::windowWidth = 3000;
        polyscope::view::windowHeight = 2000;
        polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
        polyscope::options::groundPlaneHeightFactor = 0.6; 
        polyscope::options::shadowDarkness = 0.4;

        // Initialize polyscope
        polyscope::init();
        vectorToIGLMatrix<T, 3>(simulation.deformed, meshV);
        meshF = simulation.faces.block(0,0,simulation.faces.rows(), 3);
        psMesh = polyscope::registerSurfaceMesh("surface mesh", meshV, meshF);
        psMesh->setSmoothShade(false);
        psMesh->setSurfaceColor(glm::vec3(0.255, 0.514, 0.996));
        psMesh->setEdgeWidth(1.0);
        std::vector<Eigen::Vector3d> points(simulation.sample.size());
        psCloud = polyscope::registerPointCloud("sample points", points);

        std::vector<Eigen::Vector3d> line (simulation.sample.size()*(simulation.direction.size()+1), Eigen::Vector3d::Zero());
        std::vector<std::array<size_t, 2>> edges = setEdge(simulation.sample.size(), simulation.direction.size());
        psLine = polyscope::registerCurveNetwork("sample direction", line, edges);

        polyscope::state::userCallback = [&](){ sceneCallback(); };
    }

    void sceneCallback()
    {
        if (ImGui::Button("RunSim")) 
        {
            run_sim = true;
        }
        if (ImGui::Button("Show Geometry")) 
        {
            show_geometry = true;
        }
        if (ImGui::Button("stop")) 
        {
            animate_modes = false;
            run_sim = false;
        }
        if (ImGui::Button("advanceOneStep")) 
        {
            simulation.advanceOneStep(static_solve_step++);
            vectorToIGLMatrix<T, 3>(simulation.deformed, meshV);
            psMesh->updateVertexPositions(meshV);
            // Eigen::MatrixXd meshV;
            // Eigen::MatrixXi meshF;
            // vectorToIGLMatrix<int, 3>(simulation.faces, meshF);
            // polyscope::registerSurfaceMesh("surface mesh", meshV, meshF);
        }
        if (ImGui::Checkbox("ConsistentMass", &simulation.use_consistent_mass_matrix)) 
        {
            if (!simulation.use_consistent_mass_matrix)
                simulation.computeMassMatrix();
        }
        if (ImGui::Checkbox("Dynamics", &simulation.dynamics)) 
        {
            if (simulation.dynamics)
                simulation.initializeDynamicStates();   
        }
        if (ImGui::Checkbox("Gravity", &simulation.add_gravity)) 
        {
            
        }
        if (ImGui::Checkbox("Verbose", &simulation.verbose)) 
        {
            
        }
        if (animate_modes && !run_sim)
        {
            t += 0.1;
            simulation.deformed = simulation.undeformed + simulation.u + eigen_vectors.col(modes) * std::sin(t);
            
            Eigen::MatrixXd meshV;
            vectorToIGLMatrix<T, 3>(simulation.deformed, meshV);
            psMesh->updateVertexPositions(meshV);
        }
        if (!animate_modes && run_sim)
        {
            bool finished = simulation.advanceOneStep(static_solve_step++);

            auto strain_xx = std::vector<T> (simulation.strain_tensors.size());
            auto strain_yy = std::vector<T> (simulation.strain_tensors.size());
            auto strain_xy = std::vector<T> (simulation.strain_tensors.size());
            for(int k = 0; k < simulation.strain_tensors.size(); ++k){
                auto t = simulation.returnStrainTensors(k).at(0);
                strain_xx[k] = t(0,0);
                strain_xy[k] = t(1,0);
                strain_yy[k] = t(1,1);
            }
            psMesh->addFaceScalarQuantity("strain xx magnitude", strain_xx);
            psMesh->addFaceScalarQuantity("strain xy magnitude", strain_xy);
            psMesh->addFaceScalarQuantity("strain yy magnitude", strain_yy);

            for(int k = 0; k < simulation.strain_tensors.size(); ++k){
                auto t = simulation.returnStrainTensors(k).at(1);
                strain_xx[k] = t(0,0);
                strain_xy[k] = t(1,0);
                strain_yy[k] = t(1,1);
            }

            psMesh->addFaceScalarQuantity("kernelised strain xx magnitude", strain_xx);
            psMesh->addFaceScalarQuantity("kernelised strain xy magnitude", strain_xy);
            psMesh->addFaceScalarQuantity("kernelised strain yy magnitude", strain_yy);

    
            psMesh->addFaceScalarQuantity("Poisson Ratio", simulation.nu_visualization);
            psMesh->addFaceScalarQuantity("Young's Modulo", simulation.E_visualization);

            auto stress_xx = std::vector<T> (simulation.stress_tensors.size());
            auto stress_yy = std::vector<T> (simulation.stress_tensors.size());
            auto stress_xy = std::vector<T> (simulation.stress_tensors.size());
            for(int k = 0; k < simulation.stress_tensors.size(); ++k){
                auto t = simulation.returnStressTensors(k).at(0);
                stress_xx[k] = t(0,0);
                stress_xy[k] = t(1,0);
                stress_yy[k] = t(1,1);
            }
            psMesh->addFaceScalarQuantity("stress xx magnitude", stress_xx);
            psMesh->addFaceScalarQuantity("stress xy magnitude", stress_xy);
            psMesh->addFaceScalarQuantity("stress yy magnitude", stress_yy);

            auto kernel_stress_xx = std::vector<T> (simulation.stress_tensors.size());
            auto kernel_stress_yy = std::vector<T> (simulation.stress_tensors.size());
            auto kernel_stress_xy = std::vector<T> (simulation.stress_tensors.size());
            for(int k = 0; k < simulation.stress_tensors.size(); ++k){
                auto t = simulation.returnStressTensors(k).at(1);
                kernel_stress_xx[k] = t(0,0);
                kernel_stress_xy[k] = t(1,0);
                kernel_stress_yy[k] = t(1,1);
            }

            psMesh->addFaceScalarQuantity("kernelised stress xx magnitude", kernel_stress_xx);
            psMesh->addFaceScalarQuantity("kernelised stress xy magnitude", kernel_stress_xy);
            psMesh->addFaceScalarQuantity("kernelised stress yy magnitude", kernel_stress_yy);
            psMesh->addFaceScalarQuantity("gaussian kernel weight via probing", simulation.kernel_coloring_prob);
            psMesh->addFaceScalarQuantity("gaussian kernel weight via averaging", simulation.kernel_coloring_avg);
            // set some options
            psCloud->setPointRadius(0.004);
            psCloud->updatePointPositions(simulation.pointInDeformedTriangle());

            std::vector<Eigen::Vector3d> line = updateLinePosition(simulation);
            psLine->updateNodePositions(line);
            psLine->setRadius(0.0022);
 
            psMesh->addFaceScalarQuantity("cut", simulation.cut_coloring);

            Eigen::MatrixXd meshV;
            vectorToIGLMatrix<T, 3>(simulation.deformed, meshV);
            psMesh->updateVertexPositions(meshV);
            Eigen::MatrixXd ext_force;
            vectorToIGLMatrix<T, 3>(simulation.external_force, ext_force);
            psMesh->addVertexVectorQuantity("external forces", ext_force);

            if (finished)
                run_sim = false;
        }
        if (show_geometry)
        {   
            // simulation.testHorizontalDirectionStretch();
            // simulation.testIsotropicStretch();
            simulation.testVerticalDirectionStretch();
            auto strain_xx_k = std::vector<T> (simulation.strain_tensors.size());
            auto strain_yy_k = std::vector<T> (simulation.strain_tensors.size());
            auto strain_xy_k = std::vector<T> (simulation.strain_tensors.size());
            for(int k = 0; k < simulation.strain_tensors.size(); ++k){
                auto t = simulation.returnStrainTensors(k).at(1);
                strain_xx_k[k] = t(0,0);
                strain_xy_k[k] = t(1,0);
                strain_yy_k[k] = t(1,1);
            }

            auto strain_xx = std::vector<T> (simulation.strain_tensors.size());
            auto strain_yy = std::vector<T> (simulation.strain_tensors.size());
            auto strain_xy = std::vector<T> (simulation.strain_tensors.size());
            auto eigenvec1 = std::vector<Eigen::Vector3d> (simulation.strain_tensors.size());
            auto eigenvec2 = std::vector<Eigen::Vector3d> (simulation.strain_tensors.size());
            size_t idx = 0;
            for (auto strain_tensor: simulation.strain_tensors) {
                Eigen::EigenSolver<Eigen::MatrixXd> es(strain_tensor);
                strain_xx[idx] = strain_tensor(0,0);
                strain_xy[idx] = strain_tensor(0,1);
                strain_yy[idx] = strain_tensor(1,1);
                for (int j = 0; j < 3; ++j){
                    if( es.eigenvalues()(0).real() > es.eigenvalues()(1).real()){
                        eigenvec1[idx](j) = es.eigenvectors().col(0)(j).real() * es.eigenvalues()(0).real();
                        eigenvec2[idx](j) = es.eigenvectors().col(1)(j).real() * es.eigenvalues()(1).real();
                    } else {
                        eigenvec2[idx](j) = es.eigenvectors().col(0)(j).real() * es.eigenvalues()(0).real();
                        eigenvec1[idx](j) = es.eigenvectors().col(1)(j).real() * es.eigenvalues()(1).real();
                    }
                }
                ++idx;
            }
            psMesh->addFaceVectorQuantity("strain eigenvec1", eigenvec1);
            psMesh->addFaceVectorQuantity("strain eigenvec2", eigenvec2);
            psMesh->addFaceScalarQuantity("strain xx magnitude", strain_xx);
            psMesh->addFaceScalarQuantity("strain xy magnitude", strain_xy);
            psMesh->addFaceScalarQuantity("strain yy magnitude", strain_yy);

            auto stress_xx = std::vector<T> (simulation.stress_tensors.size());
            auto stress_yy = std::vector<T> (simulation.stress_tensors.size());
            auto stress_xy = std::vector<T> (simulation.stress_tensors.size());
            eigenvec1 = std::vector<Eigen::Vector3d> (simulation.stress_tensors.size());
            eigenvec2 = std::vector<Eigen::Vector3d> (simulation.stress_tensors.size());
            idx = 0;
            for (auto stress_tensor: simulation.stress_tensors) {
                Eigen::EigenSolver<Eigen::MatrixXd> es(stress_tensor);
                stress_xx[idx] = stress_tensor(0,0);
                stress_xy[idx] = stress_tensor(0,1);
                stress_yy[idx] = stress_tensor(1,1);
                for (int j = 0; j < 2; ++j){
                    if( es.eigenvalues()(0).real() > es.eigenvalues()(1).real()){
                        eigenvec1[idx](j) = es.eigenvectors().col(0)(j).real() * es.eigenvalues()(0).real();
                        eigenvec2[idx](j) = es.eigenvectors().col(1)(j).real() * es.eigenvalues()(1).real();
                    } else {
                        eigenvec2[idx](j) = es.eigenvectors().col(0)(j).real() * es.eigenvalues()(0).real();
                        eigenvec1[idx](j) = es.eigenvectors().col(1)(j).real() * es.eigenvalues()(1).real();
                    }
                }
                ++idx;
            }
            psMesh->addFaceVectorQuantity("stress eigenvec1", eigenvec1);
            psMesh->addFaceVectorQuantity("stress eigenvec2", eigenvec2);
            psMesh->addFaceScalarQuantity("gaussian kernel weight via probing", simulation.kernel_coloring_prob);
            psMesh->addFaceScalarQuantity("gaussian kernel weight via averaging", simulation.kernel_coloring_avg);
            psMesh->addFaceScalarQuantity("stress xx magnitude", stress_xx);
            psMesh->addFaceScalarQuantity("stress xy magnitude", stress_xy);
            psMesh->addFaceScalarQuantity("stress yy magnitude", stress_yy);

            psMesh->addFaceScalarQuantity("kernelised strain xx magnitude", strain_xx_k);
            psMesh->addFaceScalarQuantity("kernelised strain xy magnitude", strain_xy_k);
            psMesh->addFaceScalarQuantity("kernelised strain yy magnitude", strain_yy_k);

            psMesh->addFaceScalarQuantity("Poisson Ratio", simulation.nu_visualization);
            psMesh->addFaceScalarQuantity("Young's Modulo", simulation.E_visualization);

            auto kernel_stress_xx = std::vector<T> (simulation.stress_tensors.size());
            auto kernel_stress_yy = std::vector<T> (simulation.stress_tensors.size());
            auto kernel_stress_xy = std::vector<T> (simulation.stress_tensors.size());
            for(int k = 0; k < simulation.stress_tensors.size(); ++k){
                auto t = simulation.returnStressTensors(k).at(1);
                kernel_stress_xx[k] = t(0,0);
                kernel_stress_xy[k] = t(1,0);
                kernel_stress_yy[k] = t(1,1);
            }

            psMesh->addFaceScalarQuantity("kernelised stress xx magnitude", kernel_stress_xx);
            psMesh->addFaceScalarQuantity("kernelised stress xy magnitude", kernel_stress_xy);
            psMesh->addFaceScalarQuantity("kernelised stress yy magnitude", kernel_stress_yy);
            // set some options
            psCloud->setPointRadius(0.004);
            psCloud->updatePointPositions(simulation.pointInDeformedTriangle());

            std::vector<Eigen::Vector3d> line = updateLinePosition(simulation);
            psLine->updateNodePositions(line);
            psLine->setRadius(0.0022);
 
            psMesh->addFaceScalarQuantity("cut", simulation.cut_coloring);

            Eigen::MatrixXd meshV;
            vectorToIGLMatrix<T, 3>(simulation.deformed, meshV);
            psMesh->updateVertexPositions(meshV);
            Eigen::MatrixXd ext_force;
            vectorToIGLMatrix<T, 3>(simulation.external_force, ext_force);
            psMesh->addVertexVectorQuantity("external forces", ext_force);
            show_geometry = false;
        }
    }

    void run()
    {
        polyscope::show();
    }

    std::vector<std::array<size_t, 2>> setEdge(size_t samples, size_t directions){
        std::vector<std::array<size_t, 2>> edges;
        for(size_t i = 0; i < samples; ++i){
            for(size_t j = 0; j < directions; ++j){
                edges.push_back({i*(directions+1), i*(directions+1)+j+1});
            }
        }
        return edges;
    }

    std::vector<Eigen::Vector3d> updateLinePosition(Simulation& sim){
        std::vector<Eigen::Vector3d> line;
        auto sample_pos = sim.pointInDeformedTriangle();
        for(size_t i = 0; i < sim.sample.size(); ++i){
            line.push_back(sample_pos[i]);
            int tri = sim.pointInTriangle(sim.sample[i]);
            // auto F_2D_inv = sim.defomation_gradients[tri].lu().solve(Matrix<T, 2, 2>::Identity());
            Matrix<T, 3, 3> F_inv = Matrix<T, 3, 3>::Zero();
            auto F = F_inv;
            F.block(0,0,2,2) = sim.defomation_gradients[tri];
            // F_inv.block(0,0,2,2) = F_2D_inv;
            for(size_t j = 0; j < sim.direction.size(); ++j){
                // line.push_back(sample_pos[i] + (F_inv.transpose()*sim.direction[j]).normalized()*0.05);
                line.push_back(sample_pos[i] + (F*sim.direction[j]).normalized()*0.05);
            }
        }
        return line;
    }


public:
    App(Simulation& sim) : simulation(sim) {}
    ~App() {}
};

#endif