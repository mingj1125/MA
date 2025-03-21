#ifndef APP_H
#define APP_H

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"
#include "../include/StiffnessTensor.h"

template<class Simulation>
class App
{
public:
    Simulation& simulation;
    int static_solve_step = 0;

    polyscope::SurfaceMesh* psMesh;
    polyscope::PointCloud* psCloud;
    polyscope::CurveNetwork* psLine;


    Eigen::MatrixXd eigen_vectors;
    Eigen::VectorXd eigen_values;

    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;

public:
    void initializeScene()
    {
        // polyscope::options::autocenterStructures = true;
        polyscope::view::windowWidth = 3000;
        polyscope::view::windowHeight = 2000;
        polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
        polyscope::options::groundPlaneHeightFactor = 0.6; 
        polyscope::options::shadowDarkness = 0.4;

        // Initialize polyscope
        polyscope::init();
        meshV = simulation.visual_undeformed;
        meshF = simulation.visual_faces;
        psMesh = polyscope::registerSurfaceMesh("surface mesh", meshV, meshF);
        psMesh->setSmoothShade(false);
        psMesh->setSurfaceColor(glm::vec3(0.255, 0.514, 0.996));
        psMesh->setEdgeWidth(1.0);
        std::vector<Eigen::Vector3d> points(simulation.sample);
        psCloud = polyscope::registerPointCloud("sample points", points);

        std::vector<Eigen::Vector3d> line (simulation.sample.size()*(simulation.direction.size()+1), Eigen::Vector3d::Zero());
        std::vector<std::array<size_t, 2>> edges = setEdge(simulation.sample.size(), simulation.direction.size());
        line = updateLinePosition(simulation);
        psLine = polyscope::registerCurveNetwork("sample direction", line, edges);
        psLine->setRadius(0.001);

        psMesh->addFaceScalarQuantity("Face Tag", simulation.face_tags);
        psMesh->addFaceScalarQuantity("Local E", simulation.E_visualization);
        psMesh->addFaceScalarQuantity("Local nu", simulation.nu_visualization);

        polyscope::state::userCallback = [&](){ sceneCallback(); };
    }

    void sceneCallback()
    {
        if(ImGui::Button("Stiffness")){
            std::cout << "C approximation for kernel size: " << simulation.std << std::endl;
            std::vector<Matrix<T, 3, 3>> sts = findStiffnessTensor(simulation.mesh_file, simulation);
            // std::vector<T> E(sts.size());
            // std::vector<T> nus(sts.size());
            
            // for(int st = 0; st < sts.size(); ++st){
            //     T C_12 = sts[st](1,0); T C_11 = 0.5*(sts[st](0,0)+sts[st](1,1));
            //     T nu = C_12/C_11/(1+C_12/C_11);
            //     nus[st] = nu;
            //     E[st] = ((C_11-C_12)/2+sts[st](2,2)) * (1+nu);
            //     E[st] = sts[st](0,0);
            //     if(st == 1036) std::cout << "1036: "<< sts[st] << std::endl;
            //     if(st == 701) std::cout << "701: "<< sts[st] << std::endl;
            // }
            // psMesh->addFaceScalarQuantity("Kernelised stiffness(E)", E);
            // psMesh->addFaceScalarQuantity("Kernelised nu", nus);
            std::vector<T> C_11(sts.size());
            std::vector<T> C_12(sts.size());
            std::vector<T> C_13(sts.size());
            std::vector<T> C_22(sts.size());
            std::vector<T> C_23(sts.size());
            std::vector<T> C_33(sts.size());
            for(int st = 0; st < sts.size(); ++st){
                C_11[st] = sts[st](0,0);
                C_12[st] = sts[st](0,1);
                C_13[st] = sts[st](0,2);
                C_22[st] = sts[st](1,1);
                C_23[st] = sts[st](1,2);
                C_33[st] = sts[st](2,2);
                // if(st == 1036) std::cout << "1036: "<< sts[st] << std::endl;
                // if(st == 701) std::cout << "701: "<< sts[st] << std::endl;
            }
            psMesh->addFaceScalarQuantity("C_11", C_11);
            psMesh->addFaceScalarQuantity("C_12", C_12);
            psMesh->addFaceScalarQuantity("C_13", C_13);
            psMesh->addFaceScalarQuantity("C_22", C_22);
            psMesh->addFaceScalarQuantity("C_23", C_23);
            psMesh->addFaceScalarQuantity("C_33", C_33);
        }
        ImGui::InputDouble("Kernel size", &simulation.std, 0.00001, 0.01, "%.5f");
        if(ImGui::Button("Update kernelization")){
        //     simulation.kernel_coloring_prob.setZero();
            auto strain_xx = std::vector<T> (simulation.visual_faces.rows());
            auto strain_yy = std::vector<T> (simulation.visual_faces.rows());
            auto strain_xy = std::vector<T> (simulation.visual_faces.rows());
            for(int k = 0; k < simulation.visual_faces.rows(); ++k){
                auto t = simulation.returnStrainTensors(k).at(0);
                strain_xx[k] = t(0,0);
                strain_xy[k] = t(1,0);
                strain_yy[k] = t(1,1);
            }

            psMesh->addFaceScalarQuantity("strain xx magnitude", strain_xx);
            psMesh->addFaceScalarQuantity("strain xy magnitude", strain_xy);
            psMesh->addFaceScalarQuantity("strain yy magnitude", strain_yy);

            for(int k = 0; k < simulation.visual_faces.rows(); ++k){
                auto t = simulation.returnStrainTensors(k).at(1);
                strain_xx[k] = t(0,0);
                strain_xy[k] = t(1,0);
                strain_yy[k] = t(1,1);
            }

            psMesh->addFaceScalarQuantity("kernelised strain xx magnitude", strain_xx);
            psMesh->addFaceScalarQuantity("kernelised strain xy magnitude", strain_xy);
            psMesh->addFaceScalarQuantity("kernelised strain yy magnitude", strain_yy);

            auto kernel_stress_xx = std::vector<T> (simulation.visual_faces.rows());
            auto kernel_stress_yy = std::vector<T> (simulation.visual_faces.rows());
            auto kernel_stress_xy = std::vector<T> (simulation.visual_faces.rows());
            for(int k = 0; k < simulation.visual_faces.rows(); ++k){
                auto t = simulation.returnStressTensors(k).at(0);
                kernel_stress_xx[k] = t(0,0);
                kernel_stress_xy[k] = t(1,0);
                kernel_stress_yy[k] = t(1,1);
            }

            psMesh->addFaceScalarQuantity("stress xx magnitude", kernel_stress_xx);
            psMesh->addFaceScalarQuantity("stress xy magnitude", kernel_stress_xy);
            psMesh->addFaceScalarQuantity("stress yy magnitude", kernel_stress_yy);

            for(int k = 0; k < simulation.visual_faces.rows(); ++k){
                auto t = simulation.returnStressTensors(k).at(1);
                kernel_stress_xx[k] = t(0,0);
                kernel_stress_xy[k] = t(1,0);
                kernel_stress_yy[k] = t(1,1);
            }

            psMesh->addFaceScalarQuantity("kernelised stress xx magnitude", kernel_stress_xx);
            psMesh->addFaceScalarQuantity("kernelised stress xy magnitude", kernel_stress_xy);
            psMesh->addFaceScalarQuantity("kernelised stress yy magnitude", kernel_stress_yy);
            
        }
        if(ImGui::Button("Update kernel visualization")){
            std::cout << "Kernel size: " << simulation.std << std::endl;
            simulation.visualizeKernelWeighting();
            psMesh->addFaceScalarQuantity("gaussian kernel weight via probing", simulation.kernel_coloring_prob);
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
        for(size_t i = 0; i < sim.sample.size(); ++i){
            Eigen::Vector3d sample_i = sim.sample[i];
            auto sample_pos = sim.pointPosInVisualTriangle(sample_i.segment<2>(0));
            line.push_back(sample_pos);
            for(size_t j = 0; j < sim.direction.size(); ++j){
                line.push_back(sample_pos + (sim.direction[j]).normalized()*100);
            }
        }
        return line;
    }


public:
    App(Simulation& sim) : simulation(sim) {}
    ~App() {}
};

#endif