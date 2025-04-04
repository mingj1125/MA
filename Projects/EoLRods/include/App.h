#ifndef APP_H
#define APP_H

#include "polyscope/polyscope.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "glm/ext.hpp"
#include <map>

// for processing mesh to rod network
#include <igl/readOBJ.h>

#include "EoLRodSim.h"

class App
{
public:
    using TV = Vector<T, 3>;
    using TM = Eigen::Matrix<T, 3, 3>;
    EoLRodSim& simulation;
    int static_solve_step = 0;

    polyscope::CurveNetwork* rod_network;
    polyscope::PointCloud* rod_vertices;
    polyscope::PointCloud* stress_probes;
    polyscope::PointCloud* nodal_stress_probes;
    polyscope::CurveNetwork* window;
    std::vector<TV> sample_loc;
    unsigned sample_density = 20;
    int to_boundary_width = 5;

    T t = 0;

    bool animate_modes = false;
    bool run_sim = false;
    int modes = 0;
    Eigen::MatrixXd eigen_vectors;
    Eigen::VectorXd eigen_values;

    Eigen::MatrixXd meshV;
    Eigen::MatrixXd meshV_deformed;
    Eigen::MatrixXi meshF;
    bool use_mesh = false;

public:
    void initializeScene();

    void initializeScene(std::string mesh_file);

    void sceneCallback();

    void run()
    {
        polyscope::show();
    }

    Vector<T, 3> pointInDeformedTriangle(const TV sample_loc);

    void updateCurrentVertex();

public:
    App(EoLRodSim& sim) : simulation(sim) {}
    ~App() {}
};

#endif