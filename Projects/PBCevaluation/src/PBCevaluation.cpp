#include "../include/PBCevaluation.h"
#include <igl/readOBJ.h>

void PBCevaluation::initializeVisualizationMesh(std::string filename){
    MatrixXT V; MatrixXi F;
    igl::readOBJ(filename, V, F);
    face_tags = VectorXi(F.rows()); face_tags.setZero();
    mesh_file = filename;

    std::ifstream file(tag_file); // Open the file
    if (!file.is_open()) {
        std::cerr << "Error: Could not open tag file!" << std::endl;
    }

    std::string l;
    std::vector<int> ftags;
    face_tags = VectorXi(F.rows());
    int count = 0;

    while (std::getline(file, l)) {
        try {
            int tag = std::stoi(l); // Convert string to integer
            face_tags(count) = tag; ++count;
        } catch (const std::exception& e) {
            std::cerr << "Error converting line to integer: " << l << std::endl;
        }
    }

    visual_undeformed = V; 
    visual_faces = F;

    kernel_coloring_prob = VectorXT::Zero(F.rows());
    kernel_coloring_avg = VectorXT::Zero(F.rows());
    sample = std::vector<TV>(2);
    sample[1] = triangleCenterofMass(getVisualFaceVtxUndeformed(1716));
    setProbingLineDirections(15);

    nu_visualization = VectorXT::Zero(F.rows());
    E_visualization = VectorXT::Zero(F.rows());
    visualizeMaterialProperties();
}

void PBCevaluation::visualizeMaterialProperties(){
    
    for(int i = 0; i < nu_visualization.size(); ++i){
        T thickness = 0.003;
        T E = 1e6*thickness, nu = 0.0;
        T factor = 10;
        if(face_tags[i] % 2 == 0) {
            E /= factor;
        }
        E_visualization[i] = E;
        nu_visualization[i] = nu;
    }
}

void PBCevaluation::visualizeKernelWeighting(){
    
    kernel_coloring_prob.setZero();
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel = [variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / (2 * M_PI *std::sqrt(variance_matrix.determinant()));
    };

    for(int j = 0; j < direction.size(); ++j){
        T step = 5;
        int n = 4*std/step; // Discretization points for cut segment
        for(int i = -n; i <= n; ++i){
            TV point = sample[1] + i*direction[j]*step;
            kernel_coloring_prob[pointInVisualTriangle(point.segment<2>(0))] = 
                std::max(kernel_coloring_prob[pointInVisualTriangle(point.segment<2>(0))], 
                        gaussian_kernel(sample[1], point));
        }
    }
}

void PBCevaluation::initializeMeshInformation(std::string undeformed_mesh, std::string mesh_info){
    igl::readOBJ(undeformed_mesh, unit_undeformed, unit_faces);
    TV min_corner = unit_undeformed.colwise().minCoeff();
    TV max_corner = unit_undeformed.colwise().maxCoeff();
    transformation = (max_corner-min_corner).segment<2>(0);
    unit_stress_tensors = read_matrices(mesh_info+"pbc_stresses.dat");
    unit_strain_tensors = read_matrices(mesh_info+"pbc_strains.dat");
}

std::vector<Matrix<T, 2, 2>> PBCevaluation::findCorrespondenceInUnit(TV2 position){

    while((position-transformation)(0) > 0.){
        position(0) -= transformation(0);
    }
    while((position)(0) < 0.){
        position(0) += transformation(0);
    }
    while((position-transformation)(1) > 0.){
        position(1) -= transformation(1);
    }
    while((position)(1) < 0.){
        position(1) += transformation(1);
    }
    int unit_tri = pointInTriangle(position);
    if(unit_tri == -1) {
        std::cout << "Cannot find unit correspondence for " << position.transpose() << std::endl;
        return {Matrix<T, 2, 2>::Zero(), Matrix<T, 2, 2>::Zero()};
    }
    return {unit_stress_tensors[unit_tri], unit_strain_tensors[unit_tri]};
}

Matrix<T, 3, 3> PBCevaluation::findBestStressTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions){
    int c = line_directions.size();
    MatrixXT n(3, c);
    MatrixXT t(3, c);
    for(int i = 0; i < c; ++i){
        TV direction = line_directions.at(i);
        TV2 direction_normal_2D; direction_normal_2D << direction(1), -direction(0);
        TV direction_normal; direction_normal.segment(0,2) = direction_normal_2D;
        t.col(i) = computeWeightedStress(sample_loc, direction);
        n.col(i) = direction_normal;
    }

    bool fit_symmetric_constrained = true;
    TM fitted_symmetric_tensor;
    if(!fit_symmetric_constrained){
        MatrixXT A = n.transpose();
        MatrixXT b = t.transpose();
        TM x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
        fitted_symmetric_tensor = x.transpose();
    } else {
        MatrixXT A = MatrixXT::Zero(3*c,6);
        VectorXT b(3*c);
        for(int i = 0; i < c; ++i){
            MatrixXT A_block = MatrixXT::Zero(3,6);
            TV normal = n.col(i);
            A_block << normal(0), normal(1), 0, normal(2), 0, 0,
                    0, normal(0), normal(1), 0, normal(2), 0,
                    0, 0, 0, normal(0), normal(1), normal(2);
            A.block(i*3, 0, 3, 6) = A_block;
            b.segment(i*3, 3) = t.col(i);
        }
        VectorXT x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
        fitted_symmetric_tensor << x(0), x(1), x(3), 
                                    x(1), x(2), x(4),
                                    x(3), x(4), x(5);
    }

    return fitted_symmetric_tensor;
}

Matrix<T, 2, 2> PBCevaluation::findBestStrainTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions){
    int c = line_directions.size();
    MatrixXT n(3, c);
    VectorXT t(c);
    for(int i = 0; i < c; ++i){
        TV direction = line_directions.at(i);
        t(i) = computeWeightedStrain(sample_loc, direction);
        n.col(i) = direction;
    }

    TM2 fitted_symmetric_tensor;
    MatrixXT A = MatrixXT::Zero(c,3);
    for(int i = 0; i < c; ++i){
        MatrixXT A_block = MatrixXT::Zero(1,3);
        TV2 normal = n.col(i).segment(0,2);
        A_block << normal(0)*normal(0), 2*normal(1)*normal(0), normal(1)*normal(1);
        A.row(i) = A_block;
        }
        VectorXT x = (A.transpose()*A).ldlt().solve(A.transpose()*t);
        fitted_symmetric_tensor << x(0), x(1), 
                                    x(1), x(2);

    return fitted_symmetric_tensor;
}

Vector<T, 3> PBCevaluation::computeWeightedStress(const TV sample_loc, TV direction){
    T sum = 0.;
    TV stress = TV::Zero();
    TV direction_normal; direction_normal << direction(1), -direction(0), 0;
    direction_normal = direction_normal.normalized(); 
        
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel = [variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / (2 * M_PI *std::sqrt(variance_matrix.determinant()));
    };

    TV weighted_traction = TV::Zero();
    T weights = 0;
    T step = 5;
    int n = 4*std/step; // Discretization points for cut segment

    for(int i = -n; i <= n; ++i){
        TV point = sample_loc + i*direction*step;
        TM2 S = findCorrespondenceInUnit(point.segment<2>(0))[0];
        if(S.squaredNorm() > 0){
            weighted_traction.segment<2>(0) += S * direction_normal.segment<2>(0) * gaussian_kernel(sample_loc, point);
            weights += gaussian_kernel(sample_loc, point);
        }
    }
    Vector<T, 4> res; res.segment<3>(0) = weighted_traction; res(3) = weights;
    res *= step;
    stress = res.segment<3>(0);
    sum = res(3);

    if (sum <= 0.) std::cout << "Sum is 0 for direction " << direction.transpose() << std::endl; 

    return stress/sum;
}

T PBCevaluation::computeWeightedStrain(const TV sample_loc, TV direction){
    T sum = 0.;
    T strain = 0.;
        
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel = [variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / (2 * M_PI *std::sqrt(variance_matrix.determinant()));
    };

    T weighted_strain = 0;
    T weights = 0;
    T step = 5;
    int n = 4*std/step; // Discretization points for cut segment
    for(int i = -n; i <= n; ++i){
        TV point = sample_loc + i*direction*step;
        TM2 GS = findCorrespondenceInUnit(point.segment<2>(0))[1];
        if(GS.squaredNorm() > 0)
        {T strain = ((direction.segment<2>(0)).transpose()) * (GS * direction.segment<2>(0));
        weighted_strain += strain * gaussian_kernel(sample_loc, point);
        weights += gaussian_kernel(sample_loc, point);}
        // if(pointInVisualTriangle(sample[1].segment<2>(0)) == pointInVisualTriangle(sample_loc.segment<2>(0))){
        //     kernel_coloring_prob[pointInVisualTriangle(point.segment<2>(0))] = 
        //         std::max(kernel_coloring_prob[pointInVisualTriangle(point.segment<2>(0))], 
        //                  gaussian_kernel(sample_loc, point));
        // }
    }
    Vector<T, 2> res; res(0) = weighted_strain; res(1) = weights;
    res *= step;
    strain = res(0);
    sum = res(1);

    if (sum <= 0.) std::cout << "Sum is 0 for direction " << direction.transpose() << std::endl; 

    return strain/sum;
}

int PBCevaluation::pointInTriangle(const TV2 sample_loc){

    for (int i = 0; i < unit_faces.rows(); i++){
        Matrix<T, 3, 2> undeformed_vertices = getUnitFaceVtxUndeformed(i).block(0,0,3,2);

        TV2 X0 = undeformed_vertices.row(0); TV2 X1 = undeformed_vertices.row(1); TV2 X2 = undeformed_vertices.row(2);
        TM2 X; X.col(0) = (X1-X0); X.col(1) = (X2-X0); 
        T denom = X.determinant();
        X.col(0) = (X1-sample_loc); X.col(1) = (X2-sample_loc); 
        T alpha = X.determinant()/denom;
        X.col(0) = (X1-X0); X.col(1) = (sample_loc-X0); 
        T beta = X.determinant()/denom;
        T gamma = 1-alpha-beta;

        if (alpha >= 0 && beta >= 0 && gamma >= 0) {
            return i;  // Return the index of the containing triangle
        }
    }

    return -1;
}

int PBCevaluation::pointInVisualTriangle(const TV2 sample_loc){

    for (int i = 0; i < visual_faces.rows(); i++){
        Matrix<T, 3, 2> undeformed_vertices = getVisualFaceVtxUndeformed(i).block(0,0,3,2);

        TV2 X0 = undeformed_vertices.row(0); TV2 X1 = undeformed_vertices.row(1); TV2 X2 = undeformed_vertices.row(2);
        TM2 X; X.col(0) = (X1-X0); X.col(1) = (X2-X0); 
        T denom = X.determinant();
        X.col(0) = (X1-sample_loc); X.col(1) = (X2-sample_loc); 
        T alpha = X.determinant()/denom;
        X.col(0) = (X1-X0); X.col(1) = (sample_loc-X0); 
        T beta = X.determinant()/denom;
        T gamma = 1-alpha-beta;

        if (alpha >= 0 && beta >= 0 && gamma >= 0) {
            return i;  // Return the index of the containing triangle
        }
    }

    return -1;
}

Vector<T,3> PBCevaluation::pointPosInVisualTriangle(const TV2 sample_loc){
    for (int i = 0; i < visual_faces.rows(); i++){
        Matrix<T, 3, 2> undeformed_vertices = getVisualFaceVtxUndeformed(i).block(0,0,3,2);

        TV2 X0 = undeformed_vertices.row(0); TV2 X1 = undeformed_vertices.row(1); TV2 X2 = undeformed_vertices.row(2);
        TM2 X; X.col(0) = (X1-X0); X.col(1) = (X2-X0); 
        T denom = X.determinant();
        X.col(0) = (X1-sample_loc); X.col(1) = (X2-sample_loc); 
        T alpha = X.determinant()/denom;
        X.col(0) = (X1-X0); X.col(1) = (sample_loc-X0); 
        T beta = X.determinant()/denom;
        T gamma = 1-alpha-beta;

        if (alpha >= 0 && beta >= 0 && gamma >= 0) {
            TV pos; pos.setZero(); pos.segment<2>(0) = alpha*X0 + gamma*X1 + beta*X2;  
            return pos;  
        }
    }
}

std::vector<Matrix<T, 2, 2>> PBCevaluation::returnStressTensors(int A){
    Matrix<T, 3, 3> undeformed_vertices = getVisualFaceVtxUndeformed(A);
    TV CoM = triangleCenterofMass(undeformed_vertices);
    TM2 stress = findCorrespondenceInUnit(CoM.segment<2>(0))[0];
    TM2 k_stress = findBestStressTensorviaProbing(CoM, direction).block(0,0,2,2);
    return {stress, k_stress};
}

std::vector<Matrix<T, 2, 2>> PBCevaluation::returnStrainTensors(int A){
    Matrix<T, 3, 3> undeformed_vertices = getVisualFaceVtxUndeformed(A);
    TV CoM = triangleCenterofMass(undeformed_vertices);
    TM2 strain = findCorrespondenceInUnit(CoM.segment<2>(0))[1];
    TM2 k_strain = findBestStrainTensorviaProbing(CoM, direction);
    return {strain, k_strain};
}

void PBCevaluation::setProbingLineDirections(unsigned int num_directions){
    direction = std::vector<TV>(num_directions);
    T angle = M_PI / num_directions;
    T offset = M_PI / 7;
    for(int i = 0; i < num_directions; ++i){
        direction[i] << std::cos(angle*i+offset), std::sin(angle*i+offset), 0;
    }
}

Vector<T, 3> PBCevaluation::triangleCenterofMass(Matrix<T, 3, 3> vertices){
    TV CoM; CoM << vertices.col(0).mean(), vertices.col(1).mean(), vertices.col(2).mean(); 
    return CoM;
}