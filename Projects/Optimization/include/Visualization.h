#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "polyscope/polyscope.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include "Scene.h"

#include "glm/ext.hpp"
#include <map>

// for processing mesh to rod network
#include <igl/readOBJ.h>


class Visualization
{
private:

    bool network_visual;

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
    bool tag = false;
    bool show_window = false;
    int groups = 0;
    int gradient_descent = 1;
    int stretch_type = 1;
    float C_test_point[2] = {0.5, 0.5};
    float length = 0.1;
    float kernel_std = 0.08;
    std::vector<Vector6a> Cs;
    std::vector<Vector6a> window_Cs;

    polyscope::SurfaceMesh* psMesh;

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
    Eigen::VectorXi readTag(const std::string tag_file); 
    VectorXa setParameterFromTags(Eigen::VectorXi tags);
    std::vector<Vector3a> getSampleLocations(){return sample_loc;};
    std::vector<Vector6a> getCs(){return Cs;};
    std::vector<Vector6a> getWindowCs(){return window_Cs;};
    std::vector<Vector4a> getWindowCorners(){
        std::vector<Vector4a> corners(sample_loc.size(), Vector4a::Zero());
        for(int i = 0; i < sample_loc.size(); ++i){
            Vector2a s = {sample_loc[i](0), sample_loc[i](1)};
            Vector2a max_corner = s + length*Vector2a({1,1});
            Vector2a min_corner = s - length*Vector2a({1,1});
            corners[i] = Vector4a({max_corner(0), max_corner(1), min_corner(0), min_corner(1)});
        }
        return corners;
    }
};



#endif