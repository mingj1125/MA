#include "../include/LinearShell.h"
#include "../include/LinearShell_EvalDiff.h"
#include <iostream>

Matrix3a LinearShell::findBestStressTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){

    int c = line_directions.size();
    std::vector<MatrixXa> gradient_t(faces.rows(), MatrixXa(3,c));
    std::vector<MatrixXa> gradient_t_wrt_x(deformed_states.rows(), MatrixXa(3,c));
    MatrixXa n(3, c);
    MatrixXa t(3, c);
    for(int i = 0; i < c; ++i){
        Vector3a direction = line_directions.at(i);
        Vector3a direction_normal; direction_normal = direction.cross(Vector3a{0,0,1});
        direction_normal = direction_normal.normalized(); 

        std::vector<Vector3a> gradient_t_di(faces.rows(), Vector3a::Zero());
        std::vector<Vector3a> gradient_t_wrt_x_di(deformed_states.rows(), Vector3a::Zero());

        t.col(i) = computeWeightedStress(sample_loc, direction, gradient_t_di, gradient_t_wrt_x_di);
        n.col(i) = direction_normal;
        // std::cout << "direction : " << direction.transpose() << "\n traction: " << t.col(i).transpose() << std::endl;
        for(int j = 0; j < gradient_t.size(); ++j){
            gradient_t[j].col(i) = gradient_t_di[j];
        }
        for(int j = 0; j < gradient_t_wrt_x.size(); ++j){
            gradient_t_wrt_x[j].col(i) = gradient_t_wrt_x_di[j];
            // if(gradient_t_wrt_x_di[j].norm() > 0) std::cout << "x " << j << " dtdx : " << gradient_t_wrt_x_di[j].transpose() << std::endl;
        }
    }

    Matrix3a fitted_tensor; fitted_tensor.setZero();
    MatrixXa A = MatrixXa::Zero(3*c,6);
    VectorXa b(3*c);
    std::vector<VectorXa> b_diff(faces.rows(), VectorXa(3*c));
    std::vector<VectorXa> b_diff_wrt_x(deformed_states.rows(), VectorXa(3*c));
    for(int i = 0; i < c; ++i){
        MatrixXa A_block = MatrixXa::Zero(3,6);
        Vector3a normal = n.col(i);
        A_block << normal(0), normal(1), normal(2), 0, 0, 0,
                0, normal(0), 0, normal(1), normal(2), 0,
                0, 0, normal(0), 0, normal(1), normal(2);
        A.block(i*3, 0, 3, 6) = A_block;
        b.segment(i*3, 3) = t.col(i);
        for(int j = 0; j < faces.rows(); ++j){
            b_diff[j].segment(i*3, 3) = gradient_t[j].col(i);
        }
        for(int j = 0; j < deformed_states.rows(); ++j){
            b_diff_wrt_x[j].segment(i*3, 3) = gradient_t_wrt_x[j].col(i);
        }
    }
    VectorXa x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    // AScalar epsilon = 1e-3;
    // x = (x.array().abs() < epsilon).select(0.0, x);
    // std::cout << "stress : " << x.transpose() << std::endl;
    eval_info_of_sample.stress_gradients_wrt_parameter.resize(3, faces.rows());
    eval_info_of_sample.stress_gradients_wrt_x.resize(3, deformed_states.rows());
    for(int i = 0; i < faces.rows(); i++){
        VectorXa x_temp = (A.transpose()*A).ldlt().solve(A.transpose()*b_diff[i]);
        eval_info_of_sample.stress_gradients_wrt_parameter.col(i) = Vector3a({x_temp(0), x_temp(3), x_temp(1)});
    }
    for(int i = 0; i < deformed_states.rows(); i++){
        VectorXa x_temp = (A.transpose()*A).ldlt().solve(A.transpose()*b_diff_wrt_x[i]);
        // AScalar epsilon = 1e-1;
        // x_temp = (x_temp.array().abs() < epsilon).select(0.0, x_temp);
        eval_info_of_sample.stress_gradients_wrt_x.col(i) = Vector3a({x_temp(0), x_temp(3), x_temp(1)});
    }
    fitted_tensor << x(0), x(1), x(2), 
                    x(1), x(3), x(4),
                    x(2), x(4), x(5);

    eval_info_of_sample.stress_tensor = fitted_tensor; 

    return fitted_tensor;
}

/* Vector3a LinearShell::computeWeightedStress(const Vector3a sample_loc, const Vector3a direction, 
    std::vector<Vector3a>& gradients_wrt_E, std::vector<Vector3a>& gradients_wrt_nodes){

        auto gaussian_kernel = [=](AScalar distance){
            return std::exp(-0.5*distance*distance/(kernel_std*kernel_std)) / (kernel_std * std::sqrt(2 * M_PI));
        };

        AScalar sum = 0.;
        Vector3a stress = Vector3a::Zero();
        Vector3a direction_normal; direction_normal << direction(1), -direction(0), 0;
        direction_normal = direction_normal.normalized(); 

        Vector3a weighted_traction = Vector3a::Zero();
        AScalar weights = 0;
        AScalar step = std::min(1e-6, kernel_std);
        int n = 4*kernel_std/step; // Discretization points for cut segment

        for(int i = -n; i <= n; ++i){
            Vector3a point = sample_loc + i*direction*step;
            if(point(0) > 1.-1e-5 || point(0) < 0.+1e-5 || point(1) > 1.-1e-5 || point(1) < 0.+1e-5) continue;
            AScalar dist = (point - sample_loc).norm();
            int t = pointInTriangle(point.segment<2>(0));
            // if point is on the edge make a small offset
            int off = 1;
            while(t == -1 && off < 15) {
                point = sample_loc + i*direction*step*(1+step/1000.0*off);
                t = pointInTriangle(point.segment<2>(0));
                dist = (point - sample_loc).norm();
                ++off;
            }
            if(t == -1) std::cout << "Cannot find corresponding triangle for location: " << (sample_loc + i*direction*step).transpose() << std::endl;
            Matrix2a S = stress_tensors_each_element[t];
            weighted_traction.segment<2>(0) += S * direction_normal.segment<2>(0) * gaussian_kernel(dist);
            gradients_wrt_E[t].segment<2>(0) += S * direction_normal.segment<2>(0) * gaussian_kernel(dist) / youngsmodulus_each_element[t];
            weights += gaussian_kernel(dist);

            std::vector<Matrix2a> dSdxs = SGradientWrtx(t);
            Eigen::Vector3i indices = faces.row(t);
            for(int j = 0; j < dSdxs.size()/3; ++j){
                for(int k = 0; k < 2; ++k){
                    gradients_wrt_nodes[indices(j)*3+k].segment<2>(0) += dSdxs[3*j+k] * direction_normal.segment<2>(0) * gaussian_kernel(dist);
                }
            }
        }
        Vector4a res; res.segment<3>(0) = weighted_traction; res(3) = weights;
        res *= step;

        stress = res.segment<3>(0);
        sum = res(3);

        for(int i = 0; i < faces.rows(); i++){
            gradients_wrt_E[i] *= step;
            gradients_wrt_E[i] /= sum;
        }
        for(int i = 0; i < deformed_states.rows(); i++){
            gradients_wrt_nodes[i] *= step;
            gradients_wrt_nodes[i] /= sum;
        }

        if (sum <= 0.) std::cout << "Sum is 0 for direction " << direction.transpose() << std::endl; 

    return stress/sum;
}
*/

Vector3a LinearShell::computeWeightedStress(const Vector3a sample_loc, const Vector3a direction, 
    std::vector<Vector3a>& gradients_wrt_E, std::vector<Vector3a>& gradients_wrt_nodes){
        
    auto gaussian_kernel = [=](AScalar distance){
        return std::exp(-0.5*distance*distance/(kernel_std*kernel_std)) / (kernel_std * std::sqrt(2 * M_PI));
    };

    AScalar sum = 0.;
    Vector3a stress = Vector3a::Zero();
    Vector3a direction_normal; direction_normal << direction(1), -direction(0), 0;
    direction_normal = direction_normal.normalized(); 

    for(int i = 0; i < faces.rows(); ++i){
        Eigen::Vector3i indices = faces.row(i);
        Vector3a X0 = rest_states.segment(indices(0)*3, 3);
        Vector3a X1 = rest_states.segment(indices(1)*3, 3);
        Vector3a X2 = rest_states.segment(indices(2)*3, 3);
        Vector3a cut_point_coordinate;
        if(lineCutTriangle(X0, X1, X2, sample_loc, direction, cut_point_coordinate)){
            Vector3a weighted_traction = Vector3a::Zero();
            AScalar weights = 0;
            Vector3a gradients_wrt_E_tr; gradients_wrt_E_tr.setZero();
            std::vector<Vector3a> gradients_wrt_nodes_tr(deformed_states.rows(), Vector3a::Zero());
            std::vector<Vector3a> boundary_points;
            if(cut_point_coordinate(0) >= 0.-1e-6 && cut_point_coordinate(0) <= 1.+1e-8) {boundary_points.push_back(X0 + cut_point_coordinate(0)*(X1-X0));}
            if(cut_point_coordinate(1) >= 0.-1e-6 && cut_point_coordinate(1) <= 1.+1e-8) {boundary_points.push_back(X1 + cut_point_coordinate(1)*(X2-X1));}
            if(cut_point_coordinate(2) >= 0.-1e-6 && cut_point_coordinate(2) <= 1.+1e-8) {boundary_points.push_back(X2 + cut_point_coordinate(2)*(X0-X2));}
            int n = std::max((int)((boundary_points.at(1) - boundary_points.at(0)).norm()/(kernel_std/6)), 5);
            for(int j = 0; j <= n; ++j){
                Vector3a point = (boundary_points.at(1) - boundary_points.at(0))*j/n + boundary_points.at(0);
                AScalar dist = (sample_loc - point).norm();
                if(dist > 4*kernel_std) continue;
                Matrix2a S = stress_tensors_each_element[i];
                weighted_traction.segment<2>(0) += S * direction_normal.segment<2>(0) * gaussian_kernel(dist);
                gradients_wrt_E_tr.segment<2>(0) += S * direction_normal.segment<2>(0) * gaussian_kernel(dist) / youngsmodulus_each_element[i];
                weights += gaussian_kernel(dist);

                std::vector<Matrix2a> dSdxs = SGradientWrtx(i);
                for(int l = 0; l < dSdxs.size()/3; ++l){
                    for(int k = 0; k < 2; ++k){
                        gradients_wrt_nodes_tr[indices(l)*3+k].segment<2>(0) += dSdxs[3*l+k] * direction_normal.segment<2>(0) * gaussian_kernel(dist);
                    }
                }
            }
            weighted_traction *= (boundary_points.at(1) - boundary_points.at(0)).norm() /(2*n);
            weights *= (boundary_points.at(1) - boundary_points.at(0)).norm() /(2*n);
            gradients_wrt_E[i] += gradients_wrt_E_tr * (boundary_points.at(1) - boundary_points.at(0)).norm() /(2*n);
            for(int j = 0; j < deformed_states.rows(); ++j){
                gradients_wrt_nodes[j] += gradients_wrt_nodes_tr[j] * (boundary_points.at(1) - boundary_points.at(0)).norm() /(2*n);
            }

            stress += weighted_traction;
            sum += weights;
        }
    }
    for(int i = 0; i < faces.rows(); i++){
        gradients_wrt_E[i] /= sum;
    }
    for(int i = 0; i < deformed_states.rows(); i++){
        gradients_wrt_nodes[i] /= sum;
    }
    return stress / sum;
}

void LinearShell::computeStressAndStraininTriangles(){

    for (int i = 0; i < faces.rows(); i++){

        AScalar E = youngsmodulus_each_element(i);
        AScalar lambda = E * nu /((1+nu)*(1-2*nu));
        AScalar mu = E / (2*(1+nu));

        Vector9a q;
        Vector6a p;
        Eigen::Vector3i indices = faces.row(i);
        for(int j = 0; j < 3; ++j){
            q.segment(j*3,3) = deformed_states.segment(indices(j)*3, 3);
            p.segment(j*2,2) = rest_states.segment(indices(j)*3, 2);
        }
        Eigen::Matrix<AScalar, 3, 2> Q; 
        Q.col(0) = q.segment(3, 3) - q.segment(0, 3); 
        Q.col(1) = q.segment(6, 3) - q.segment(0, 3);
        Matrix2a P;
        P.col(0) = p.segment(2, 2) - p.segment(0, 2);
        P.col(1) = p.segment(4, 2) - p.segment(0, 2);

        Eigen::Matrix<AScalar, 3, 2> F = Q * P.lu().solve(Matrix2a::Identity());
        Matrix2a green_strain = 0.5 * (F.transpose()*F - Matrix2a::Identity());
        // if(std::abs(green_strain(0,0)) < 1e-6) green_strain(0,0) = 0;
        // if(std::abs(green_strain(1,0)) < 1e-6) green_strain(1,0) = 0;
        // if(std::abs(green_strain(0,1)) < 1e-6) green_strain(0,1) = 0;
        // if(std::abs(green_strain(1,1)) < 1e-6) green_strain(1,1) = 0;
        strain_tensors_each_element[i] = green_strain;
        Matrix2a S = (lambda*green_strain.trace()*Matrix2a::Identity() + 2*mu*green_strain);
        stress_tensors_each_element[i] = S;
    }    
}

int LinearShell::pointInTriangle(const Vector2a sample_loc){

    for (int i = 0; i < faces.rows(); i++){
        Eigen::Vector3i indices = faces.row(i);
        Vector2a X0 = rest_states.segment(indices(0)*3, 2);
        Vector2a X1 = rest_states.segment(indices(1)*3, 2);
        Vector2a X2 = rest_states.segment(indices(2)*3, 2);
        Matrix2a X; X.col(0) = (X1-X0); X.col(1) = (X2-X0); 
        AScalar denom = X.determinant();
        X.col(0) = (X1-sample_loc); X.col(1) = (X2-sample_loc); 
        AScalar alpha = X.determinant()/denom;
        X.col(0) = (X1-X0); X.col(1) = (sample_loc-X0); 
        AScalar beta = X.determinant()/denom;
        AScalar gamma = 1-alpha-beta;

        if (alpha >= 0 && beta >= 0 && gamma >= 0) {
            return i;  // Return the index of the containing triangle
        }
    }

    return -1;
}

std::vector<Matrix2a> LinearShell::SGradientWrtx(int face_id){

    Eigen::Vector3i indices = faces.row(face_id);
    Vector9a q;
    Vector6a p;
    for(int j = 0; j < 3; ++j){
        q.segment(j*3,3) = deformed_states.segment(indices(j)*3, 3);
        p.segment(j*2,2) = rest_states.segment(indices(j)*3, 2);
    }

    AScalar E = youngsmodulus_each_element(face_id);
    AScalar lambda = E * nu /((1+nu)*(1-2*nu));
    AScalar mu = E / (2*(1+nu));

    Eigen::Matrix<AScalar, 4, 9> diff_S = dSdx(q, p, lambda, mu);
    std::vector<Matrix2a> res(9);
    for(int j = 0; j < 9; ++j){
        Matrix2a dSdx_i;
        dSdx_i << diff_S(0, j), diff_S(2, j), diff_S(1, j), diff_S(3, j);
        res[j] = dSdx_i;
    }
    
    return res;
}

Matrix3a LinearShell::findBestStrainTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){
    
    int c = line_directions.size();
    MatrixXa n(3, c);
    VectorXa t(c);

    std::vector<VectorXa> gradient_t(faces.rows(), VectorXa(c));
    std::vector<VectorXa> gradient_t_wrt_x(deformed_states.rows(), VectorXa(c));

    for(int i = 0; i < c; ++i){
        Vector3a direction = line_directions.at(i);
        std::vector<AScalar> gradient_t_wrt_x_di(deformed_states.rows(), 0);

        t(i) = computeWeightedStrain(sample_loc, direction, gradient_t_wrt_x_di);
        n.col(i) = direction;
        // std::cout << "direction : " << direction.transpose() << "\n strain: " << t(i) << std::endl;
        
        for(int j = 0; j < gradient_t_wrt_x.size(); ++j){
            gradient_t_wrt_x[j](i) = gradient_t_wrt_x_di[j];
            // if(std::abs(gradient_t_wrt_x_di[j] )> 0) std::cout << "x " << j << " dedx : " << gradient_t_wrt_x_di[j] << std::endl;
        }
    }

    Matrix3a fitted_symmetric_tensor;
    MatrixXa A = MatrixXa::Zero(c,3);

    for(int i = 0; i < c; ++i){
        MatrixXa A_block = MatrixXa::Zero(1,3);
        Vector2a normal = n.col(i).segment(0,2);
        A_block << normal(0)*normal(0), 2*normal(1)*normal(0), normal(1)*normal(1);
        A.row(i) = A_block;
    }
    VectorXa x = (A.transpose()*A).ldlt().solve(A.transpose()*t);
    // AScalar epsilon = 1e-6;
    // x = (x.array().abs() < epsilon).select(0.0, x);
    // std::cout << "strain : " << x.transpose() << std::endl;
    eval_info_of_sample.strain_gradients_wrt_x.resize(3, deformed_states.rows());
    for(int i = 0; i < deformed_states.rows(); i++){
        VectorXa x_temp = (A.transpose()*A).ldlt().solve(A.transpose()*gradient_t_wrt_x[i]);
        // AScalar epsilon = 1e-3;
        // x_temp = (x_temp.array().abs() < epsilon).select(0.0, x_temp);
        eval_info_of_sample.strain_gradients_wrt_x.col(i) = Vector3a({x_temp(0), x_temp(2), 2*x_temp(1)});
    }
    fitted_symmetric_tensor << x(0), x(1), 0,
                                x(1), x(2), 0,
                                0, 0, 1;
    eval_info_of_sample.strain_tensor = fitted_symmetric_tensor;                        


    return fitted_symmetric_tensor;
}

/*AScalar LinearShell::computeWeightedStrain(const Vector3a sample_loc, Vector3a direction,
    std::vector<AScalar>& gradients_wrt_nodes){

    AScalar sum = 0.;
    AScalar strain = 0.;
        
    auto gaussian_kernel = [=](AScalar distance){
        return std::exp(-0.5*distance*distance/(kernel_std*kernel_std)) / (kernel_std * std::sqrt(2 * M_PI));
    };

    AScalar weighted_strain = 0;
    AScalar weights = 0;
    AScalar step = std::min(1e-3, kernel_std);
    int n = 4*kernel_std/step; // Discretization points for cut segment
    for(int i = -n; i <= n; ++i){
        Vector3a point = sample_loc + i*direction*step;
        if(point(0) > 1.-1e-5 || point(0) < 0.+1e-5 || point(1) > 1.-1e-5 || point(1) < 0.+1e-5) continue;
        AScalar dist = (point - sample_loc).norm();
        int t = pointInTriangle(point.segment<2>(0));
        // if point is on the edge make a small offset
        int off = 1;
        while(t == -1 && off < 15) {
            point = sample_loc + i*direction*step*(1-step/1000.0*off);
            t = pointInTriangle(point.segment<2>(0));
            dist = (point - sample_loc).norm();
            ++off;
        }
        if(t == -1) std::cout << "Cannot find corresponding triangle for location: " << (sample_loc + i*direction*step).transpose() << std::endl;
        Matrix2a GS = strain_tensors_each_element[t];
        AScalar strain = ((direction.segment<2>(0)).transpose()) * (GS * direction.segment<2>(0));
        weighted_strain += strain * gaussian_kernel(dist);
        weights += gaussian_kernel(dist);

        std::vector<Matrix2a> dGSdxs = GSGradientWrtx(t);
        Eigen::Vector3i indices = faces.row(t);
        for(int j = 0; j < dGSdxs.size()/3; ++j){
            for(int k = 0; k < 3; ++k){
                AScalar strain_dx = ((direction.segment<2>(0)).transpose()) * (dGSdxs[3*j+k] * direction.segment<2>(0));
                gradients_wrt_nodes[indices(j)*3+k] += strain_dx * gaussian_kernel(dist);
            }
        }
    }
    Vector2a res; res(0) = weighted_strain; res(1) = weights;
    res *= step;
    strain = res(0);
    sum = res(1);
    for(int i = 0; i < deformed_states.rows(); i++){
        gradients_wrt_nodes[i] *= step;
        gradients_wrt_nodes[i] /= sum;
    }

    // std::cout << "Weight sum: " << sum << std::endl;
    if (sum <= 0.) std::cout << "Sum is 0 for direction " << direction.transpose() << std::endl; 

    return strain/sum;
}
*/

AScalar LinearShell::computeWeightedStrain(const Vector3a sample_loc, Vector3a direction,
    std::vector<AScalar>& gradients_wrt_nodes){

    AScalar sum = 0.;
    AScalar strain = 0.;
        
    auto gaussian_kernel = [=](AScalar distance){
        return std::exp(-0.5*distance*distance/(kernel_std*kernel_std)) / (kernel_std * std::sqrt(2 * M_PI));
    };

    for(int i = 0; i < faces.rows(); ++i){
        Eigen::Vector3i indices = faces.row(i);
        Vector3a X0 = rest_states.segment(indices(0)*3, 3);
        Vector3a X1 = rest_states.segment(indices(1)*3, 3);
        Vector3a X2 = rest_states.segment(indices(2)*3, 3);
        Vector3a cut_point_coordinate;
        if(lineCutTriangle(X0, X1, X2, sample_loc, direction, cut_point_coordinate)){
            AScalar weighted_strain = 0;
            AScalar weights = 0;
            std::vector<AScalar> gradients_wrt_nodes_tr(deformed_states.rows(), 0.0);
            std::vector<Vector3a> boundary_points;
            if(cut_point_coordinate(0) >= 0.-1e-6 && cut_point_coordinate(0) <= 1.+1e-8) {boundary_points.push_back(X0 + cut_point_coordinate(0)*(X1-X0));}
            if(cut_point_coordinate(1) >= 0.-1e-6 && cut_point_coordinate(1) <= 1.+1e-8) {boundary_points.push_back(X1 + cut_point_coordinate(1)*(X2-X1));}
            if(cut_point_coordinate(2) >= 0.-1e-6 && cut_point_coordinate(2) <= 1.+1e-8) {boundary_points.push_back(X2 + cut_point_coordinate(2)*(X0-X2));}
            int n = std::max((int)((boundary_points.at(1) - boundary_points.at(0)).norm()/(kernel_std/6)), 5);
            for(int j = 0; j <= n; ++j){
                Vector3a point = (boundary_points.at(1) - boundary_points.at(0))*j/n + boundary_points.at(0);
                AScalar dist = (sample_loc - point).norm();
                if(dist > 4*kernel_std) continue;
                Matrix2a GS = strain_tensors_each_element[i];
                AScalar strain = ((direction.segment<2>(0)).transpose()) * (GS * direction.segment<2>(0));
                weighted_strain += strain * gaussian_kernel(dist);
                weights += gaussian_kernel(dist);

                std::vector<Matrix2a> dGSdxs = GSGradientWrtx(i);
                for(int l = 0; l < dGSdxs.size()/3; ++l){
                    for(int k = 0; k < 2; ++k){
                        AScalar strain_dx = ((direction.segment<2>(0)).transpose()) * (dGSdxs[3*l+k] * direction.segment<2>(0));
                        gradients_wrt_nodes_tr[indices(l)*3+k] += strain_dx * gaussian_kernel(dist);
                    }
                }
            }
            weighted_strain *= (boundary_points.at(1) - boundary_points.at(0)).norm() /(2*n);
            weights *= (boundary_points.at(1) - boundary_points.at(0)).norm() /(2*n);
            for(int j = 0; j < deformed_states.rows(); ++j){
                gradients_wrt_nodes[j] += gradients_wrt_nodes_tr[j] * (boundary_points.at(1) - boundary_points.at(0)).norm() /(2*n);
            }

            strain += weighted_strain;
            sum += weights;
        }
    }
    for(int i = 0; i < deformed_states.rows(); i++){
        gradients_wrt_nodes[i] /= sum;
    }    
    // std::cout << "Weight sum: " << sum << std::endl;
    if (sum <= 0.) std::cout << "Sum is 0 for direction " << direction.transpose() << std::endl; 

    return strain/sum;
}

std::vector<Matrix2a> LinearShell::GSGradientWrtx(int face_id){

    Eigen::Vector3i indices = faces.row(face_id);
    Vector9a q;
    Vector6a p;
    for(int j = 0; j < 3; ++j){
        q.segment(j*3,3) = deformed_states.segment(indices(j)*3, 3);
        p.segment(j*2,2) = rest_states.segment(indices(j)*3, 2);
    }

    Eigen::Matrix<AScalar, 4, 9> diff_E = dEdx(q, p);
    std::vector<Matrix2a> res(9);
    for(int j = 0; j < 9; ++j){
        Matrix2a dEdx_i;
        dEdx_i << diff_E(0, j), diff_E(2, j), diff_E(1, j), diff_E(3, j);
        res[j] = dEdx_i;
    }
    
    return res;
}

// return if a line cut through a triangle and return the cut points' barycentric coordinate
bool LinearShell::lineCutTriangle(const Vector3a x1, const Vector3a x2, const Vector3a x3, const Vector3a sample_point, const Vector3a line_direction, Vector3a &cut_point_coordinate){
    
    int count = 0;
    cut_point_coordinate << -1, -1, -1;
    std::vector<Vector3a> points(3, Vector3a::Zero());

    Vector2a r1 = solveLineIntersection(sample_point, line_direction, x3, x2);
    Vector2a r2 = solveLineIntersection(sample_point, line_direction, x2, x1);
    Vector2a r3 = solveLineIntersection(sample_point, line_direction, x1, x3);
    if(r1(0) >= 0.-1e-6 && r1(0) <= 1.+1e-8) {++count; cut_point_coordinate(1) = r1(0);points[1] = x2 + r1(0)*(x3-x2);}
    if(r2(0) >= 0.-1e-6 && r2(0) <= 1.+1e-8) {++count; cut_point_coordinate(0) = r2(0);points[0] = x1 + r2(0)*(x2-x1);}
    if(r3(0) >= 0.-1e-6 && r3(0) <= 1.+1e-8) {++count; cut_point_coordinate(2) = r3(0);points[2] = x3 + r3(0)*(x1-x3);}
    
    // check if intersections are at the corners
    for(int i = 0; i < 3; ++i){
        for(int j = i+1; j < 3; ++j){
            if(cut_point_coordinate(i) <= -1 || cut_point_coordinate(j) <= -1) continue;
            if((points[i]-points[j]).norm() <= 1e-8) {
                --count; cut_point_coordinate(j) = -1;
            }
        }
    }
    

    if(count > 1) return true;
    return false;

}

Vector2a LinearShell::solveLineIntersection(const Vector3a sample_point, const Vector3a line_direction, const Vector3a v1, const Vector3a v2){
    Vector3a e = v1 - v2;
    Vector3a b; b << sample_point-v2;
    Eigen::Matrix<AScalar, 3, 2> A; A.col(0) = e; A.col(1) = -line_direction;
    Vector2a r = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    if(std::abs(e.normalized().transpose()*line_direction.normalized())-1 >= 0.-1e-8) r(0) = -1;

    return r;
}
