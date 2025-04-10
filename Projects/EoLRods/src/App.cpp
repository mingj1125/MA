#include "../include/App.h"

void App::sceneCallback()
{
    if (ImGui::Button("RunSim")) 
        {
            run_sim = true;
        }
        if (ImGui::Button("stop")) 
        {
            run_sim = false;
        }
        if (ImGui::Button("Show optimized radii")) 
        {
            int global_cnt = 0;
            std::vector<double> radii;
            std::vector<double> mesh_radii;
            std::string filename = "../../../Projects/EoLRods/optimization_output/"+mesh_name+"_radii.dat";
            std::ifstream in_file(filename);
            if (!in_file) {
                std::cerr << "Error opening file for reading: " << filename << std::endl;
            }
        
            T a;
            while (in_file >> a) {
                mesh_radii.push_back(a);
            }
            for(auto& rod : simulation.Rods)
            {   
                for (int i = 0; i < rod->indices.size()-1; i++)
                {
                    radii.push_back(mesh_radii[rod->rod_id]);
                }
                global_cnt += rod->indices.size();
            }
            rod_network->addEdgeScalarQuantity("rod radii", radii);

        }
        if(ImGui::Checkbox("Optimized", &optimized)){static_solve_step =0;}
        if (run_sim)
        {   
            if(optimized){
                std::vector<double> rods_radii;
                std::string filename = "../../../Projects/EoLRods/optimization_output/"+mesh_name+"_radii.dat";
                std::ifstream in_file(filename);
                if (!in_file) {
                    std::cerr << "Error opening file for reading: " << filename << std::endl;
                }
            
                T a;
                while (in_file >> a) {
                    rods_radii.push_back(a);
                }
                for(auto rod: simulation.Rods){
                    rod->a = rods_radii[rod->rod_id];
                    rod->b = rods_radii[rod->rod_id];
                    rod->initCoeffs();
                }
            } else {
                for(auto rod: simulation.Rods){
                    rod->a = 3e-4;
                    rod->b = 3e-4;
                    rod->initCoeffs();
                }
            }
            bool finished = simulation.advanceOneStep(static_solve_step++);
            std::vector<glm::vec3> nodes;
            for(auto& rod : simulation.Rods)
            {
                for (int idx : rod->indices)
                {
                    TV node_i;
                    rod->x(idx, node_i);
                    nodes.emplace_back(node_i[0], node_i[1], node_i[2]);
                }
            }
            rod_network->updateNodePositions(nodes);
        
            if(finished){
                std::vector<double> stress_xx(sample_loc.size());
                std::vector<double> stress_xy(sample_loc.size());
                std::vector<double> stress_yy(sample_loc.size());
                std::vector<double> strain_xx(sample_loc.size());
                std::vector<double> strain_xy(sample_loc.size());
                std::vector<double> strain_yy(sample_loc.size());
                std::vector<TV> directions;
                int num_directions = 15;
                for(int i = 0; i < num_directions; ++i) {double angle = i*2*M_PI/num_directions; 
                directions.push_back(TV{std::cos(angle), std::sin(angle), 0});}
                int idx = 0;
                TM average = TM::Zero();
                for(auto sample_location: sample_loc){
                    // if(sample_location.norm() > 0) 
                    {
                        auto tensor = simulation.findBestStressTensorviaProbing(sample_location, directions);
                        auto F = simulation.computeWeightedDeformationGradient(sample_location, directions);
                        // auto cauchy_stress = F*tensor*F.transpose()/F.determinant();
                        auto cauchy_stress = tensor;
                        // if(idx % (sample_density-1-2*to_boundary_width) == 1) average += cauchy_stress;
                        average += cauchy_stress;
                        // std::cout << "sample: \n" << sample_location.transpose() << "\n stress \n" << cauchy_stress << std::endl;
                        auto green_strain = 0.5*(F.transpose()*F-TM::Identity());
                        stress_xx.at(idx) = cauchy_stress(0,0);
                        stress_xy.at(idx) = cauchy_stress(1,0);
                        stress_yy.at(idx) = cauchy_stress(1,1);
                        strain_xx.at(idx) = green_strain(0,0);
                        strain_xy.at(idx) = green_strain(1,0);
                        strain_yy.at(idx) = green_strain(1,1);
                        if(idx == 133) {
                            // for(auto g: simulation.stress_gradients_wrt_rod_thickness){
                            //     std::cout << g.transpose() << std::endl;
                            // }
                            TV top_right = sample_location+TV{0.2, 0.2, 0}*simulation.unit;
                            TV bottom_left = sample_location-TV{0.2, 0.2, 0}*simulation.unit;
                            std::vector<TV> window_corner = std::vector<TV>(4, TV::Zero());
                            window_corner[0] = bottom_left;
                            window_corner[1] = TV(bottom_left(0), top_right(1), 0); // top left
                            // std::cout << TV(bottom_left(0), top_right(1), 0).transpose() << std::endl;
                            window_corner[2] = top_right;
                            window_corner[3] = TV(top_right(0), bottom_left(1), 0); // bottom right
                            if(use_mesh){
                                for(int i = 0; i < window_corner.size(); ++i){
                                    window_corner.at(i) = pointInDeformedTriangle(window_corner.at(i));
                                }
                                window->updateNodePositions(window_corner);
                            }
                            std::cout << "Window for " << idx << " : \n" << simulation.computeWindowHomogenization(top_right, bottom_left) << std::endl;
                        }
                    }
                    if(idx % 50 == 0) std::cout << "stress evaluation progress: " << idx+1 << "/" << sample_loc.size() << std::endl;
                    ++idx;
                }
                // std::cout << "Average stress tensor along y: \n" << average/(sample_density-1-2*to_boundary_width) << std::endl;
                // std::cout << "Average stress tensor: \n" << average/idx << std::endl;
                stress_probes->addScalarQuantity("stress xx", stress_xx);
                stress_probes->addScalarQuantity("stress xy", stress_xy);
                stress_probes->addScalarQuantity("stress yy", stress_yy);
                stress_probes->addScalarQuantity("strain xx", strain_xx);
                stress_probes->addScalarQuantity("strain xy", strain_xy);
                stress_probes->addScalarQuantity("strain yy", strain_yy);

                // std::vector<double> nodal_stress_xx(simulation.n_nodes);
                // std::vector<double> nodal_stress_xy(simulation.n_nodes);
                // std::vector<double> nodal_stress_yy(simulation.n_nodes);
                // for(int i = 0; i < simulation.n_nodes; ++i){
                //     auto tensor = simulation.computeNodeStress(i);
                //     nodal_stress_xx.at(i) = tensor(0,0);
                //     nodal_stress_xy.at(i) = tensor(1,0);
                //     nodal_stress_yy.at(i) = tensor(1,1);
                //     // if(i == 45) {
                //     //     auto tensor = simulation.computeNodeStress(i);
                //     //     std::cout << "nodal stress local: \n" << tensor << std::endl;
                //     //     TV node_i;
                //     //     for(auto& rod : simulation.Rods)
                //     //     {
                //     //         for (int idx : rod->indices)
                //     //         {   
                //     //             if(idx == i)
                //     //             rod->X(idx, node_i);

                //     //         }
                //     //     }
                //     //     tensor = simulation.findBestStressTensorviaProbing(node_i, directions);
                //     //     std::cout << "nodal S approx: \n" << tensor << std::endl;
                //     //     // auto F = simulation.computeWeightedDeformationGradient(node_i, directions);
                //     //     // std::cout << "nodal F: \n" << F << std::endl;
                //     //     // auto cauchy_stress = F*tensor*F.transpose()/F.determinant();
                //     //     // std::cout << "nodal stress approx: \n" << cauchy_stress << std::endl;
                //     // }
                //     // if(i == 45) {
                //     //     TV sample_location;
                //     //     for(auto& rod : simulation.Rods)
                //     //     {
                //     //         for (int idx : rod->indices)
                //     //         {   
                //     //             if(idx == i)
                //     //             rod->X(idx, sample_location);

                //     //         }
                //     //     }
                //     //     TV top_right = sample_location+TV{0.06, 0.06, 0}*simulation.unit;
                //     //     TV bottom_left = sample_location-TV{0.06, 0.06, 0}*simulation.unit;
                //     //     auto window_corner = std::vector<TV>(4, TV::Zero());
                //     //     window_corner[0] = bottom_left;
                //     //     window_corner[1] = TV(bottom_left(0), top_right(1), 0); // top left
                //     //     window_corner[2] = top_right;
                //     //     window_corner[3] = TV(top_right(0), bottom_left(1), 0); // bottom right
                //     //     if(use_mesh){
                //     //         for(int i = 0; i < window_corner.size(); ++i){
                //     //             window_corner.at(i) = pointInDeformedTriangle(window_corner.at(i));
                //     //         }
                //     //         window->updateNodePositions(window_corner);
                //     //     }
                //     //     std::cout << "Window: \n" << simulation.computeWindowHomogenization(top_right, bottom_left) << std::endl;
                //     // }
                // }
                // nodal_stress_probes->addScalarQuantity("nodal stress xx", nodal_stress_xx);
                // nodal_stress_probes->addScalarQuantity("nodal stress xy", nodal_stress_xy);
                // nodal_stress_probes->addScalarQuantity("nodal stress yy", nodal_stress_yy);
                
                if(use_mesh){
                    std::vector<TV> points(sample_loc.size(), TV::Zero());
                    updateCurrentVertex();
                    for(int i = 0; i < sample_loc.size(); ++i){
                        points.at(i) = pointInDeformedTriangle(sample_loc.at(i));
                    }
                    stress_probes->updatePointPositions(points);
                } else {
                    TV top_right, bottom_left;
                    simulation.computeBoundingBox(bottom_left, top_right);
                    TV start = bottom_left + (top_right-bottom_left)/sample_density;
                    std::vector<TV> points((sample_density-1-2*to_boundary_width)*(sample_density-1-2*to_boundary_width), TV::Zero());
                    for(int i = to_boundary_width; i < sample_density-1-to_boundary_width; i++){
                        for(int j = to_boundary_width; j < sample_density-1-to_boundary_width; j++){
                            int p_i = i-to_boundary_width;
                            int p_j = j-to_boundary_width;
                            points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j) = start;
                            points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j)(0) += (top_right-bottom_left)(0)/sample_density*j;
                            points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j)(1) += (top_right-bottom_left)(1)/sample_density*i;
                        }
                    }
                    stress_probes->updatePointPositions(points);
                }
                // nodal_stress_probes->updatePointPositions(meshV_deformed);
                
                run_sim = false;
            }
        }
}

void App::initializeScene()
{
    // polyscope::options::autocenterStructures = true;
    polyscope::view::windowWidth = 3000;
    polyscope::view::windowHeight = 2000;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
    polyscope::options::groundPlaneHeightFactor = 0.2; 
    polyscope::options::shadowDarkness = 0.4;

    // Initialize polyscope
    polyscope::init();
    // std::cout << simulation.Rods.size() << std::endl;
    std::vector<glm::vec3> nodes;
    std::vector<std::array<size_t, 2>> edges;
    int global_cnt = 0;
    for(auto& rod : simulation.Rods)
    {
        for (int idx : rod->indices)
        {
            TV node_i;
            rod->x(idx, node_i);
            nodes.push_back(glm::vec3(node_i[0], node_i[1], node_i[2]));
        }
        for (int i = 0; i < rod->indices.size()-1; i++)
        {
            edges.push_back({size_t(global_cnt + i), size_t(global_cnt + i + 1)});
        }
        global_cnt += rod->indices.size();
    }
    rod_network = polyscope::registerCurveNetwork("rod network", nodes, edges);
    rod_vertices = polyscope::registerPointCloud("nodes", nodes);
    rod_vertices->setPointRadius(0.007);
    // psMesh = polyscope::registerSurfaceMesh("surface mesh", simulation.surface_vertices, simulation.surface_indices);
    // psMesh->setSmoothShade(false);

    TV top_right, bottom_left;
    simulation.computeUndeformedBoundingBox(bottom_left, top_right);
    sample_density = 25;
    to_boundary_width = 7;
    TV start = bottom_left + (top_right-bottom_left)/sample_density;
    std::vector<TV> points((sample_density-1-2*to_boundary_width)*(sample_density-1-2*to_boundary_width), TV::Zero());
    for(int i = to_boundary_width; i < sample_density-1-to_boundary_width; i++){
        for(int j = to_boundary_width; j < sample_density-1-to_boundary_width; j++){
            int p_i = i-to_boundary_width;
            int p_j = j-to_boundary_width;
            points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j) = start;
            points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j)(0) += (top_right-bottom_left)(0)/sample_density*j;
            points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j)(1) += (top_right-bottom_left)(1)/sample_density*i;
        }
    }
    stress_probes = polyscope::registerPointCloud("sample locations", points);
    sample_loc = points;     

    stress_probes->setPointRadius(0.008);
    rod_network->setColor(glm::vec3(0.255, 0.514, 0.996));
    // psMesh->setEdgeWidth(1.0);

    polyscope::state::userCallback = [&](){ sceneCallback(); };
}

void App::initializeScene(std::string mesh){

     // polyscope::options::autocenterStructures = true;
     polyscope::view::windowWidth = 3000;
     polyscope::view::windowHeight = 2000;
     polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
     polyscope::options::groundPlaneHeightFactor = 0.2; 
     polyscope::options::shadowDarkness = 0.4;

     // Initialize polyscope
     polyscope::init();
     // std::cout << simulation.Rods.size() << std::endl;
     std::vector<glm::vec3> nodes;
     std::vector<TV> nodal_pos(simulation.n_nodes);
     std::vector<std::array<size_t, 2>> edges;
     int global_cnt = 0;
     for(auto& rod : simulation.Rods)
     {
         for (int idx : rod->indices)
         {
             TV node_i;
             rod->x(idx, node_i);
             nodes.push_back(glm::vec3(node_i[0], node_i[1], node_i[2]));
             nodal_pos[idx] = node_i;
         }
         for (int i = 0; i < rod->indices.size()-1; i++)
         {
            edges.push_back({size_t(global_cnt + i), size_t(global_cnt + i + 1)});
         }
         global_cnt += rod->indices.size();
     }
     rod_network = polyscope::registerCurveNetwork("rod network", nodes, edges);
     rod_vertices = polyscope::registerPointCloud("nodes", nodes);
     rod_vertices->setPointRadius(0.007);
    //  nodal_stress_probes = polyscope::registerPointCloud("node locations", nodal_pos);
     TV top_right, bottom_left;
     simulation.computeUndeformedBoundingBox(bottom_left, top_right);
     sample_density = 25;
     to_boundary_width = 4;
     TV start = bottom_left + (top_right-bottom_left)/sample_density;
     std::vector<TV> points((sample_density-1-2*to_boundary_width)*(sample_density-1-2*to_boundary_width), TV::Zero());
     for(int i = to_boundary_width; i < sample_density-1-to_boundary_width; i++){
         for(int j = to_boundary_width; j < sample_density-1-to_boundary_width; j++){
             int p_i = i-to_boundary_width;
             int p_j = j-to_boundary_width;
             points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j) = start;
             points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j)(0) += (top_right-bottom_left)(0)/sample_density*j;
             points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j)(1) += (top_right-bottom_left)(1)/sample_density*i;
         }
     }
     stress_probes = polyscope::registerPointCloud("sample locations", points);
     sample_loc = points;     

     stress_probes->setPointRadius(0.008);
    //  nodal_stress_probes->setPointRadius(0.008);
     rod_network->setColor(glm::vec3(0.255, 0.514, 0.996));
     rod_network->setRadius(0.00335);

     std::vector<std::array<size_t, 2>> window_edge(4);
     for(int e = 0; e < 4; ++e){
         window_edge.push_back({e, (e+1)%4});
     }
     window = polyscope::registerCurveNetwork("window", std::vector<TV>(4, TV::Zero()), window_edge);
     
     use_mesh = true;
     igl::readOBJ("../../../Projects/EoLRods/data/"+mesh+".obj", meshV, meshF);
     TV min_corner = meshV.colwise().minCoeff();
     TV max_corner = meshV.colwise().maxCoeff();
     T length = max_corner(0)-min_corner(0);
     meshV /= length;
     meshV_deformed = meshV;
     mesh_name = mesh;

     polyscope::state::userCallback = [&](){ sceneCallback(); };
}

Vector<T, 3> App::pointInDeformedTriangle(const TV sample_loc){

    for (int i = 0; i < meshF.rows(); i++){
        TM undeformed_vertices;
        Vector<int, 3> nodal_indices = meshF.row(i);
        for (int j = 0; j < 3; j++)
        {
            undeformed_vertices.row(j) = meshV.row(nodal_indices[j]);
        }

        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);
        Matrix<T, 2, 2> X; X.col(0) = (X1-X0).segment(0,2); X.col(1) = (X2-X0).segment(0,2); 
        T denom = X.determinant();
        X.col(0) = (X1-sample_loc).segment(0,2); X.col(1) = (X2-sample_loc).segment(0,2); 
        T alpha = X.determinant()/denom;
        X.col(0) = (X1-X0).segment(0,2); X.col(1) = (sample_loc-X0).segment(0,2); 
        T beta = X.determinant()/denom;
        T gamma = 1-alpha-beta;

        if (alpha >= 0 && beta >= 0 && gamma >= 0) {
            TM vertices;
            for (int j = 0; j < 3; j++)
            {
                vertices.row(j) = meshV_deformed.row(nodal_indices[j]);
            }
            TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
            return alpha*x0 + gamma*x1 + beta*x2;  
        }
    }

    return TV::Zero();
}

void App::updateCurrentVertex(){

    for(auto& rod : simulation.Rods)
    {
        for (int idx : rod->indices)
        {
            TV node_i;
            rod->x(idx, node_i);
            meshV_deformed.row(idx) = node_i;
        }
    }    
}
