#include "../include/Scene.h"

void Scene::buildSceneFromMesh(const std::string& filename){
    mesh_file = filename;
    sim.initializeScene(filename);
}

int Scene::parameter_dof(){
    return parameters.rows();
}

Matrix3a Scene::returnApproxStressInCurrentSimulation(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){
    return sim.findBestStressTensorviaProbing(sample_loc, line_directions);
}

Matrix3a Scene::returnApproxStrainInCurrentSimulation(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){
    return sim.findBestStrainTensorviaProbing(sample_loc, line_directions);
}