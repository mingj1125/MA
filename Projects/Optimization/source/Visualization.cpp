#include "../include/Visualization.h"

void Visualization::initializeScene(bool network){

    // polyscope::options::autocenterStructures = true;
    polyscope::view::windowWidth = 3000;
    polyscope::view::windowHeight = 2000;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
    polyscope::options::groundPlaneHeightFactor = 0.2; 
    polyscope::options::shadowDarkness = 0.4;

    // Initialize polyscope
    polyscope::init();

    igl::readOBJ(scene->mesh_file, meshV, meshF);
    Vector3a min_corner = meshV.colwise().minCoeff();
    Vector3a max_corner = meshV.colwise().maxCoeff();
    AScalar length = max_corner(0)-min_corner(0);
    meshV.rowwise() -= min_corner.transpose();
    meshV /= length;
    meshV_deformed = meshV;

    if(network){
        
        network_visual = true;
        VectorXa X = scene->get_undeformed_nodes();
        std::vector<glm::vec3> nodes(X.rows()/3);
        for(int i = 0; i < nodes.size(); ++i) nodes[i] = glm::vec3({X(i*3, 0), X(i*3+1, 0), X(i*3+2, 0)});
        std::vector<std::array<size_t, 2>> edges = scene->get_edges();

        rod_network = polyscope::registerCurveNetwork("rod network", nodes, edges);
        rod_vertices = polyscope::registerPointCloud("nodes", nodes);
        rod_vertices->setPointRadius(0.007);

        rod_network->setColor(glm::vec3(0.255, 0.514, 0.996));
        rod_network->setRadius(0.00335);

    }
    else {

        network_visual = false;

        psMesh = polyscope::registerSurfaceMesh("surface mesh", meshV, meshF);
        psMesh->setSurfaceColor(glm::vec3(0.255, 0.514, 0.996));
        psMesh->setEdgeWidth(1.0);
    }
    Vector3a top_right({1,1,0});
    Vector3a bottom_left({0,0,0});
    sample_density = 25;
    to_boundary_width = 4;
    Vector3a start = bottom_left + (top_right-bottom_left)/sample_density;
    std::vector<Vector3a> points((sample_density-1-2*to_boundary_width)*(sample_density-1-2*to_boundary_width), Vector3a::Zero());
    for(int i = to_boundary_width; i < sample_density-1-to_boundary_width; i++){
        for(int j = to_boundary_width; j < sample_density-1-to_boundary_width; j++){
            int p_i = i-to_boundary_width;
            int p_j = j-to_boundary_width;
            points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j) = start;
            points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j)(0) += (top_right-bottom_left)(0)/sample_density*j;
            points.at(p_i*(sample_density-1-2*to_boundary_width)+p_j)(1) += (top_right-bottom_left)(1)/sample_density*i;
        }
    }
    probes = polyscope::registerPointCloud("sample locations", points);
    sample_loc = points;     

    probes->setPointRadius(0.007);

    polyscope::state::userCallback = [&](){ sceneCallback(); };
}

void Visualization::sceneCallback(){
    if (ImGui::Button("Simulation Result")){
        VectorXa x = scene->get_deformed_nodes();
        std::vector<glm::vec3> nodes(x.rows()/3);
        for(int i = 0; i < nodes.size(); ++i) nodes[i] = glm::vec3({x(i*3, 0), x(i*3+1, 0), x(i*3+2, 0)});
        if(network_visual){
            rod_network -> updateNodePositions(nodes);
            rod_vertices -> updatePointPositions(nodes);
        } else {
            psMesh -> updateVertexPositions(nodes);
        }

        std::vector<Vector3a> points(sample_loc.size(), Vector3a::Zero());
        updateCurrentVertex();
        for(int i = 0; i < sample_loc.size(); ++i){
            points.at(i) = pointInDeformedTriangle(sample_loc.at(i));
        }
        probes->updatePointPositions(points);
    }
    if (ImGui::Button("Initial State")){
        VectorXa X = scene->get_undeformed_nodes();
        std::vector<glm::vec3> nodes(X.rows()/3);
        for(int i = 0; i < nodes.size(); ++i) nodes[i] = glm::vec3({X(i*3, 0), X(i*3+1, 0), X(i*3+2, 0)});
        if(network_visual){
            rod_network -> updateNodePositions(nodes);
            rod_vertices -> updatePointPositions(nodes);
        } else {
            psMesh -> updateVertexPositions(nodes);
        }

        std::vector<Vector3a> points(sample_loc.size(), Vector3a::Zero());
        meshV_deformed = meshV;
        for(int i = 0; i < sample_loc.size(); ++i){
            points.at(i) = pointInDeformedTriangle(sample_loc.at(i));
        }
        probes->updatePointPositions(points);
    }
    if (ImGui::Button("Visualize Stress")){
        std::vector<AScalar> stress_xx(sample_loc.size());
        std::vector<AScalar> stress_xy(sample_loc.size());
        std::vector<AScalar> stress_yy(sample_loc.size());

        std::vector<Vector3a> directions;
        for(int i = 0; i < scene->num_directions; ++i) {
            AScalar angle = i*2*M_PI/scene->num_directions; 
            directions.push_back(Vector3a{std::cos(angle), std::sin(angle), 0});
        }
        int idx = 0;
        for(auto sample_location: sample_loc){
            auto stress = scene->returnApproxStressInCurrentSimulation(sample_location, directions);
            stress_xx.at(idx) = stress(0,0);
            stress_xy.at(idx) = stress(1,0);
            stress_yy.at(idx) = stress(1,1);
            ++idx;
        }

        probes->addScalarQuantity("stress xx", stress_xx);
        probes->addScalarQuantity("stress xy", stress_xy);
        probes->addScalarQuantity("stress yy", stress_yy);
    }
    if (ImGui::Button("Visualize Strain")){
        std::vector<AScalar> strain_xx(sample_loc.size());
        std::vector<AScalar> strain_xy(sample_loc.size());
        std::vector<AScalar> strain_yy(sample_loc.size());

        std::vector<Vector3a> directions;
        for(int i = 0; i < scene->num_directions; ++i) {
            AScalar angle = i*2*M_PI/scene->num_directions; 
            directions.push_back(Vector3a{std::cos(angle), std::sin(angle), 0});
        }
        int idx = 0;
        for(auto sample_location: sample_loc){
            auto strain = scene->returnApproxStrainInCurrentSimulation(sample_location, directions);
            strain_xx.at(idx) = strain(0,0);
            strain_xy.at(idx) = strain(1,0);
            strain_yy.at(idx) = strain(1,1);
            ++idx;
        }

        probes->addScalarQuantity("strain xx", strain_xx);
        probes->addScalarQuantity("strain xy", strain_xy);
        probes->addScalarQuantity("strain yy", strain_yy);
    }
    // if(ImGui::InputInt("Stretch type", &stretch_type)){}
    if (ImGui::Checkbox("Optimized", &optimized) || ImGui::InputInt("GD/SGN/FD", &gradient_descent) || ImGui::InputInt("Stretch type", &stretch_type) || ImGui::Checkbox("Predefined mesh group", &tag)) 
    {
        if(optimized && !tag){
            VectorXa params_from_file(scene->parameter_dof());
            std::string filename;
            if(network_visual){
                if(gradient_descent == 1) filename= "../../../Projects/Optimization/optimization_output/"+scene->mesh_name+"_gd_params.dat";
                else if(gradient_descent == 2) filename= "../../../Projects/Optimization/optimization_output/"+scene->mesh_name+"_sgn_params.dat";
                else if(gradient_descent == 3) filename= "../../../Projects/Optimization/optimization_output/"+scene->mesh_name+"_fd_params.dat";
            } else {
                if(gradient_descent == 1) filename= "../../../Projects/Optimization/optimization_output/shell_"+scene->mesh_name+"_gd_params.dat";
                else if(gradient_descent == 2) filename= "../../../Projects/Optimization/optimization_output/shell_"+scene->mesh_name+"_sgn_params.dat";
                else if(gradient_descent == 3) filename= "../../../Projects/Optimization/optimization_output/shell_"+scene->mesh_name+"_fd_params.dat";
            }
            std::ifstream in_file(filename);
            if (!in_file) {
                std::cerr << "Error opening file for reading: " << filename << std::endl;
            }
        
            AScalar a; int cnt = 0;
            while (in_file >> a) {
                params_from_file(cnt) = a;
                ++cnt;
            }
            scene->parameters = params_from_file;

            if(network_visual) rod_network->addEdgeScalarQuantity("rod radii", params_from_file);
            else psMesh->addFaceScalarQuantity("Young's modulus", params_from_file);
            scene->simulateWithParameter(params_from_file, stretch_type);
        } else if(tag && !optimized && !network_visual){
            Eigen::VectorXi tag_from_file(scene->parameter_dof());
            std::string filename = "../../../Projects/Optimization/data/"+scene->mesh_name+"_tags.dat";
            tag_from_file = readTag(filename);
            groups = tag_from_file.maxCoeff();
            scene->parameters = setParameterFromTags(tag_from_file);

            scene->simulateWithParameter(scene->parameters, stretch_type);
            psMesh->addFaceScalarQuantity("Young's modulus", scene->parameters);
        }  else {
            scene->parameters = scene->get_initial_params();
            scene->simulateWithParameter(scene->get_initial_params(), stretch_type);
            if(network_visual){
                rod_network->addEdgeScalarQuantity("rod radii", scene->parameters);
            } else {
                psMesh->addFaceScalarQuantity("Young's modulus", scene->parameters);
            }
        }

    } 
    if(ImGui::InputFloat2("Point for C: ", C_test_point)){}
    if(ImGui::Button("Calculate C")){
        Vector3a s = {C_test_point[0], C_test_point[1], 0.0};
        std::vector<Vector3a> directions;
        for(int i = 0; i < scene->num_directions; ++i) {
            AScalar angle = i*2*M_PI/scene->num_directions; 
            directions.push_back(Vector3a{std::cos(angle), std::sin(angle), 0});
        }
        scene->findBestCTensorviaProbing({s}, directions, true);
    }
    
}

// for sample points visualization in deformed configuration

void Visualization::updateCurrentVertex(){

    VectorXa x = scene->get_deformed_nodes();
    for(int i = 0; i < x.rows()/3; ++i){
        meshV_deformed.row(i) = x.segment(i*3, 3).transpose();
    }
}

Vector3a Visualization::pointInDeformedTriangle(const Vector3a sample_loc){

    for (int i = 0; i < meshF.rows(); i++){
        Matrix3a undeformed_vertices;
        Eigen::Vector<int, 3> nodal_indices = meshF.row(i);
        for (int j = 0; j < 3; j++)
        {
            undeformed_vertices.row(j) = meshV.row(nodal_indices[j]);
        }

        Vector3a X0 = undeformed_vertices.row(0); 
        Vector3a X1 = undeformed_vertices.row(1); 
        Vector3a X2 = undeformed_vertices.row(2);
        Matrix2a X; X.col(0) = (X1-X0).segment(0,2); X.col(1) = (X2-X0).segment(0,2); 
        AScalar denom = X.determinant();
        X.col(0) = (X1-sample_loc).segment(0,2); 
        X.col(1) = (X2-sample_loc).segment(0,2); 
        AScalar alpha = X.determinant()/denom;
        X.col(0) = (X1-X0).segment(0,2); X.col(1) = (sample_loc-X0).segment(0,2); 
        AScalar beta = X.determinant()/denom;
        AScalar gamma = 1-alpha-beta;

        if (alpha >= 0 && beta >= 0 && gamma >= 0) {
            Matrix3a vertices;
            for (int j = 0; j < 3; j++)
            {
                vertices.row(j) = meshV_deformed.row(nodal_indices[j]);
            }
            Vector3a x0 = vertices.row(0); 
            Vector3a x1 = vertices.row(1); 
            Vector3a x2 = vertices.row(2);
            return alpha*x0 + gamma*x1 + beta*x2;  
        }
    }

    return Vector3a::Zero();
}

Eigen::VectorXi Visualization::readTag(const std::string tag_file){
    std::ifstream file(tag_file); // Open the file
    if (!file.is_open()) {
        std::cerr << "Error: Could not open tag file!" << std::endl;
    }

    std::string l;
    std::vector<int> ftags;
    Eigen::VectorXi face_tags = Eigen::VectorXi(scene->parameter_dof());
    int count = 0;

    while (std::getline(file, l)) {
        try {
            int tag = std::stoi(l); // Convert string to integer
            face_tags(count) = tag; ++count;
        } catch (const std::exception& e) {
            std::cerr << "Error converting line to integer: " << l << std::endl;
        }
    }
    return face_tags;
}

VectorXa Visualization::setParameterFromTags(Eigen::VectorXi tags){
    VectorXa E = scene->get_initial_params();
    if(scene->mesh_name == "sun_mesh_line_clean"){
        AScalar factor = 10;
        for(int i = 0; i < E.rows(); ++i){
            if(tags[i] == 0)  E(i) /= factor*factor;
            else if(tags[i] % 2 == 0) {
                E(i) /= factor;
            }
        }
    }
    return E;
}