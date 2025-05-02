#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "polyscope/polyscope.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "Scene.h"

#include "glm/ext.hpp"
#include <map>

// for processing mesh to rod network
#include <igl/readOBJ.h>


class Visualization
{
private:

    polyscope::CurveNetwork* rod_network;
    polyscope::PointCloud* rod_vertices;
    polyscope::PointCloud* probes;
    std::vector<Vector3a> sample_loc;
    unsigned sample_density = 20;
    int to_boundary_width = 5;

    Eigen::MatrixXd meshV;
    Eigen::MatrixXd meshV_deformed;
    Eigen::MatrixXi meshF;

    Scene* scene;
    bool optimized = false;
    int stretch_type = 1;
    float C_test_point[2] = {0.5, 0.5};

public:
    Visualization(Scene* scene_i): scene(scene_i){}
    void initializeScene(bool network);
    void sceneCallback();

    void run()
    {
        polyscope::show();
    }
    void updateCurrentVertex();
    Vector3a pointInDeformedTriangle(const Vector3a sample_loc);
};



#endif