#include "../include/Scene.h"

void Scene::buildSceneFromMesh(const std::string& filename){
    mesh_file = filename;
    sim.initializeScene(filename);
    parameters = sim.get_initial_parameter();
}

int Scene::parameter_dof(){
    return parameters.rows();
}

int Scene::x_dof(){
    return sim.get_deformed_nodes().rows();
}

Matrix3a Scene::returnApproxStressInCurrentSimulation(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){
    return sim.findBestStressTensorviaProbing(sample_loc, line_directions);
}

Matrix3a Scene::returnApproxStrainInCurrentSimulation(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){
    return sim.findBestStrainTensorviaProbing(sample_loc, line_directions);
}

struct stress_strain_relationship{

    stress_strain_relationship(int r, int c, int d){
        n.resize(3, c);
        t.resize(3, c);
        t_diff = std::vector<MatrixXa>(r, MatrixXa(3,c));
        n_diff_x = std::vector<MatrixXa>(d, MatrixXa(3,c));
        t_diff_x = std::vector<MatrixXa> (d, MatrixXa(3,c));
    }
    MatrixXa n;
    MatrixXa t;
    std::vector<MatrixXa> t_diff;
    std::vector<MatrixXa> n_diff_x;
    std::vector<MatrixXa> t_diff_x;

};

void Scene::findBestCTensorviaProbing(std::vector<Vector3a> sample_locs, 
    const std::vector<Vector3a> line_directions, bool opt){
        
    int num_test = 3;
    std::vector<stress_strain_relationship> samples(sample_locs.size(), stress_strain_relationship(parameters.rows(), num_test, sim.get_deformed_nodes().rows()));
    sample_Cs_info = std::vector<C_info>(sample_locs.size(), C_info(sim.get_deformed_nodes().rows(), parameter_dof())); 
    for(int i = 1; i <= num_test; ++i){
        sim.ApplyBoundaryStretch(i);
        if(opt){
            sim.setOptimizationParameter(parameters);
        }
        sim.Simulate(false);

        for(int l = 0; l < sample_locs.size(); ++l){
            Matrix3a E = sim.findBestStrainTensorviaProbing(sample_locs[l], line_directions);
            Matrix3a S = sim.findBestStressTensorviaProbing(sample_locs[l], line_directions);
            samples[l].t.col(i-1) = Vector3a({S(0,0), S(1,1), 2*S(1,0)});
            samples[l].n.col(i-1) = Vector3a({E(0,0), E(1,1), 2*E(1,0)});
            for(int j = 0; j < (samples[l].t_diff).size(); ++j){
                samples[l].t_diff[j].col(i-1) = sim.getStressGradientWrtParameter().col(j);
            }
            for(int j = 0; j < samples[l].n_diff_x.size(); ++j){
               samples[l].t_diff_x[j].col(i-1) = sim.getStressGradientWrtx().col(j);
               samples[l].n_diff_x[j].col(i-1) = sim.getStrainGradientWrtx().col(j);
            }
        }

    }

    for(int l = 0; l < sample_locs.size(); ++l){
        Matrix3a fitted_tensor; fitted_tensor.setZero();
        MatrixXa A = MatrixXa::Zero(3*num_test,6);
        VectorXa b(3*num_test);
        std::vector<VectorXa> b_diff(samples[l].t_diff.size(), VectorXa(3*num_test));
        std::vector<VectorXa> b_diff_x(samples[l].t_diff_x.size(), VectorXa(3*num_test));
        std::vector<MatrixXa> A_diff_x(samples[l].n_diff_x.size(), MatrixXa::Zero(3*num_test,6));
        for(int i = 0; i < num_test; ++i){
            MatrixXa A_block = MatrixXa::Zero(3,6);
            Vector3a normal = samples[l].n.col(i);
            A_block << normal(0), normal(1), normal(2), 0, 0, 0,
                    0, normal(0), 0, normal(1), normal(2), 0,
                    0, 0, normal(0), 0, normal(1), normal(2);
            A.block(i*3, 0, 3, 6) = A_block;
            b.segment(i*3, 3) = samples[l].t.col(i);
            for(int j = 0; j < samples[l].t_diff.size(); ++j){
                b_diff[j].segment(i*3, 3) = samples[l].t_diff[j].col(i);
            }
            for(int j = 0; j < samples[l].n_diff_x.size(); ++j){
                b_diff_x[j].segment(i*3, 3) = samples[l].t_diff_x[j].col(i);
                MatrixXa A_block_diff_x = MatrixXa::Zero(3,6);
                Vector3a normal_x = samples[l].n_diff_x[j].col(i);
                A_block_diff_x << normal_x(0), normal_x(1), normal_x(2), 0, 0, 0,
                        0, normal_x(0), 0, normal_x(1), normal_x(2), 0,
                        0, 0, normal_x(0), 0, normal_x(1), normal_x(2);
                A_diff_x[j].block(i*3, 0, 3, 6) = A_block_diff_x;
            }
        }
        VectorXa x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
        sample_Cs_info[l].C_entry = x;
        for(int i = 0; i < samples[l].t_diff.size(); ++i){
            sample_Cs_info[l].C_diff_p[i] =  (A.transpose()*A).ldlt().solve(A.transpose()*b_diff[i]);
        }
        for(int i = 0; i < samples[l].t_diff_x.size(); ++i){
            sample_Cs_info[l].C_diff_x[i] =  (A.transpose()*A).ldlt().solve(A_diff_x[i].transpose()*b+A.transpose()*b_diff_x[i]-(A_diff_x[i].transpose()*A+A.transpose()*A_diff_x[i])*sample_Cs_info[l].C_entry);
        }  
    }
    for(int l = 0; l < sample_locs.size(); ++l)
        std::cout << "C: " << sample_Cs_info[l].C_entry.transpose() << std::endl;
}