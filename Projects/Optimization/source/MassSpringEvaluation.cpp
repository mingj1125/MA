#include "../include/MassSpring.h"
#include "../include/MassSpring_StressDiff.h"
#include <iostream>

Vector2a solveLineIntersection(const Vector3a sample_point, const Vector3a line_direction, const Vector3a v1, const Vector3a v2){
    Vector3a e = v1 - v2;
    Vector3a b; b << sample_point-v2;
    Eigen::Matrix<AScalar, 3, 2> A; A.col(0) = e; A.col(1) = -line_direction;
    Vector2a r = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    if(std::abs(e.normalized().transpose()*line_direction.normalized())-1 >= 0.-1e-8) r(0) = -1;

    return r;
}

Matrix3a MassSpring::Spring::computeSecondPiolaStress(Vector3a xi, Vector3a xj, Vector3a Xi, Vector3a Xj){
    AScalar E = YoungsModulus;
    AScalar l = (xi-xj).norm(); AScalar L0 = (Xi-Xj).norm();
    Vector3a N1 = rest_tangent;
    Vector3a n1 = (xj - xi).normalized();
    Vector3a N2 = Vector3a({0,0,1});
    Vector3a n2 = Vector3a({0,0,1});
    Vector3a N3 = N1.cross(N2).normalized();
    Vector3a n3 = n1.cross(n2).normalized();
    AScalar lambda1 = l/L0;
    Matrix3a F = lambda1 * n1 * N1.transpose() + n2 * N2.transpose() + n3 * N3.transpose();
    AScalar mu = 0.5*E;
    AScalar lambda = 0;
    Matrix3a GreenStrain = 0.5*(F.transpose()*F-Matrix3a::Identity());
    Matrix3a S = lambda * GreenStrain.trace() * Matrix3a::Identity() + 2*mu*GreenStrain;
    return S;
}

Matrix3a MassSpring::findBestStressTensorviaProbing(const Vector3a sample_loc, const std::vector<Vector3a> line_directions){

    int c = line_directions.size();
    std::vector<MatrixXa> gradient_t(springs.size(), MatrixXa(3,c));
    std::vector<MatrixXa> gradient_t_wrt_x(deformed_states.rows(), MatrixXa(3,c));
    MatrixXa n(3, c);
    MatrixXa t(3, c);
    for(int i = 0; i < c; ++i){
        Vector3a direction = line_directions.at(i);
        Vector3a direction_normal; direction_normal = direction.cross(Vector3a{0,0,1});
        direction_normal = direction_normal.normalized(); 

        std::vector<Vector3a> gradient_t_di(springs.size(), Vector3a::Zero());
        std::vector<Vector3a> gradient_t_wrt_x_di(deformed_states.rows(), Vector3a::Zero());

        t.col(i) = computeWeightedStress(sample_loc, direction, gradient_t_di, gradient_t_wrt_x_di, true);
        n.col(i) = direction_normal;
        for(int j = 0; j < gradient_t.size(); ++j){
            gradient_t[j].col(i) = gradient_t_di[j];
        }
        for(int j = 0; j < gradient_t_wrt_x.size(); ++j){
            gradient_t_wrt_x[j].col(i) = gradient_t_wrt_x_di[j];
        }
    }

    Matrix3a fitted_tensor; fitted_tensor.setZero();
    MatrixXa A = MatrixXa::Zero(3*c,6);
    VectorXa b(3*c);
    std::vector<VectorXa> b_diff(springs.size(), VectorXa(3*c));
    std::vector<VectorXa> b_diff_wrt_x(deformed_states.rows(), VectorXa(3*c));
    for(int i = 0; i < c; ++i){
        MatrixXa A_block = MatrixXa::Zero(3,6);
        Vector3a normal = n.col(i);
        A_block << normal(0), normal(1), normal(2), 0, 0, 0,
                0, normal(0), 0, normal(1), normal(2), 0,
                0, 0, normal(0), 0, normal(1), normal(2);
        A.block(i*3, 0, 3, 6) = A_block;
        b.segment(i*3, 3) = t.col(i);
        for(int j = 0; j < springs.size(); ++j){
            b_diff[j].segment(i*3, 3) = gradient_t[j].col(i);
        }
        for(int j = 0; j < deformed_states.rows(); ++j){
            b_diff_wrt_x[j].segment(i*3, 3) = gradient_t_wrt_x[j].col(i);
        }
    }
    VectorXa x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    eval_info_of_sample.stress_gradients_wrt_spring_thickness.resize(3, springs.size());
    eval_info_of_sample.stress_gradients_wrt_x.resize(3, deformed_states.rows());
    for(int i = 0; i < springs.size(); i++){
        VectorXa x = (A.transpose()*A).ldlt().solve(A.transpose()*b_diff[i]);
        eval_info_of_sample.stress_gradients_wrt_spring_thickness.col(i) = Vector3a({x(0), x(3), 2*x(1)});
    }
    for(int i = 0; i < deformed_states.rows(); i++){
        VectorXa x = (A.transpose()*A).ldlt().solve(A.transpose()*b_diff_wrt_x[i]);
        eval_info_of_sample.stress_gradients_wrt_x.col(i) = Vector3a({x(0), x(3), 2*x(1)});
    }
    fitted_tensor << x(0), x(1), x(2), 
                    x(1), x(3), x(4),
                    x(2), x(4), x(5);

    return fitted_tensor;
}

Vector3a MassSpring::computeWeightedStress(const Vector3a sample_loc, const Vector3a direction, 
    std::vector<Vector3a>& gradients_wrt_parameter, std::vector<Vector3a>& gradients_wrt_nodes, bool diff){

    Vector3a stress = Vector3a::Zero();
    bool cut = false;
    for(auto spring: springs){
        AScalar cut_point_barys;
        if(lineCutRodinSegment(spring, sample_loc, direction, cut_point_barys)){
            
            cut = true;
            Vector3a xi, xj, Xi, Xj;
            int node_i = spring->p1;
            int node_j = spring->p2;
            xi = deformed_states.segment(node_i*3, 3);
            xj = deformed_states.segment(node_j*3, 3);
            Xi = rest_states.segment(node_i*3, 3);
            Xj = rest_states.segment(node_j*3, 3);
            Vector3a cut_point = Xj + cut_point_barys*(Xi-Xj);
            AScalar distance = (cut_point - sample_loc).norm();
            Vector3a normal = direction.cross(Vector3a{0,0,1}).normalized();
            // if cut goes through the node position, only consider the springs on the correct half plane
            if(cut_point_barys <= 0.){
                auto sp_d = (Xi-cut_point).normalized();
                if(normal.dot(sp_d) < 0.) {continue;}
            } else if(cut_point_barys >= 1.){
                auto sp_d = (Xj-cut_point).normalized();
                if(normal.dot(sp_d) < 0.) {continue;}
            }

            Vector3a stress_cross_section; 
            // stress over cross section area
            stress_cross_section = integrateOverCrossSection(spring, normal, distance, gradients_wrt_parameter[spring->spring_id], gradients_wrt_nodes, diff);
            stress += stress_cross_section;

        }
    }
    
    if(!cut){ std::cout << "No cut found in the direction " << direction.transpose() << "!\n";return stress;}
    AScalar weight_sum = integrateKernelOverDomain(sample_loc, direction);
    for(int i = 0; i < springs.size(); i++){
        gradients_wrt_parameter[i] /= weight_sum;
    }
    for(int i = 0; i < deformed_states.rows(); i++){
        gradients_wrt_nodes[i] /= weight_sum;
    }
    // std::cout << weight_sum << std::endl;
    return stress/weight_sum;
}

bool MassSpring::lineCutRodinSegment(Spring* spring, Vector3a sample_loc, Vector3a direction, AScalar& cut_point_barys){

    unsigned count = 0;
    cut_point_barys = 0;
    Vector3a Xi, Xj;
    int node_i = spring->p1;
    int node_j = spring->p2;
    Xi = rest_states.segment(node_i*3, 3);
    Xj = rest_states.segment(node_j*3, 3);

    Vector3a segment_direction = (Xj-Xi).normalized();
    auto r = solveLineIntersection(sample_loc, direction, Xi, Xj);
    if(r(0) > 0.-1e-6 && r(0) < 1.+1e-6){
        cut_point_barys=r(0);
        ++count;
    }

    if(count > 0) return true;
    return false;
}

// Function to integrate over a single ellipse
Vector3a MassSpring::integrateOverCrossSection(Spring* spring, const Vector3a normal, const AScalar center_line_distance_to_sample, 
    Vector3a& gradient_wrt_thickness, std::vector<Vector3a>& gradient_wrt_nodes, bool diff){

    Vector3a integral = {0.0, 0.0, 0.0};
    Eigen::Vector<AScalar, 18> integral_diff_traction; integral_diff_traction.setZero();
    int n = 10; // Discretization points for x

    auto gaussian_kernel = [=](AScalar distance){
        return std::exp(-0.5*distance*distance/(kernel_std*kernel_std)) / (kernel_std * std::sqrt(2 * M_PI)); /// width;
    };

    AScalar weights = 0;
    Vector3a xi, xj, Xi, Xj;
    int node_i = spring->p1;
    int node_j = spring->p2;
    xi = deformed_states.segment(node_i*3, 3);
    xj = deformed_states.segment(node_j*3, 3);
    Xi = rest_states.segment(node_i*3, 3);
    Xj = rest_states.segment(node_j*3, 3);

    AScalar b = spring->width;
    for (int i = 0; i <= n; ++i) {
        AScalar x = -b + 2.0 * b * i / n;
        Matrix3a S = spring->computeSecondPiolaStress(xi, xj, Xi, Xj);
        Vector3a f_val = S * normal;
        AScalar kernel = gaussian_kernel(center_line_distance_to_sample+x);

        integral += kernel * f_val;
        if(diff) gradient_wrt_thickness += kernel * f_val*(-(center_line_distance_to_sample+x))/kernel_std/kernel_std*x/b;
        weights += kernel;

        if(diff){
        Eigen::Matrix<AScalar, 18, 1> diff_traction = SGradientWrtx(spring, xi, xj, Xi, Xj) *normal;
        integral_diff_traction += kernel * diff_traction;
        }
    }

    integral *= 2.0 * b / n;
    weights *= 2.0 * b / n;
    integral_diff_traction *= 2.0 * b / n;
    gradient_wrt_thickness *= 2.0 * b / n;
    gradient_wrt_thickness += integral/b;
    // std::cout << "Weights in spring: " << weights << std::endl;

    int offset_i = spring->p1;
    int offset_j = spring->p2;
    for(int i = 0; i < 3; ++i){
        gradient_wrt_nodes[offset_i+i] = integral_diff_traction.segment<3>(i*3);
        gradient_wrt_nodes[offset_j+i] = integral_diff_traction.segment<3>(i*3+9);
    }

    return integral;
}

Eigen::Matrix<AScalar, 18, 3> MassSpring::SGradientWrtx(Spring* spring, Vector3a xi, Vector3a xj, Vector3a Xi, Vector3a Xj){

    Vector6a X; X.segment<3>(0) = Xi; X.segment<3>(3) = Xj; 
    Vector6a x; x.segment<3>(0) = xi; x.segment<3>(3) = xj; 
    Vector3a N1 = spring->rest_tangent;
    Eigen::Matrix<AScalar, 9, 6> diff_S = dSdx(spring->YoungsModulus, X, x, N1);

    Eigen::Matrix<AScalar, 3, 9> diff_S_i = diff_S.block(0,0,9,3).transpose();
    Eigen::Matrix<AScalar, 3, 9> diff_S_j = diff_S.block(0,3,9,3).transpose();

    Eigen::Matrix<AScalar, 18, 3> res;
    for(int i = 0; i < 3; ++i){
        res.block(i*3, 0, 3, 3) = diff_S_i.block(0, i*3, 3, 3);
        res.block(i*3+9, 0, 3, 3) = diff_S_j.block(0, i*3, 3, 3);
    }

    return res;
}

AScalar MassSpring::integrateKernelOverDomain(const Vector3a sample_loc, const Vector3a line_direction){

    Vector3a bottom_left({0,0,0});
    Vector3a top_right({1,1,0});
    std::vector<AScalar> intersections(4);
    intersections[0] = (bottom_left[0] - sample_loc[0])/line_direction[0];
    intersections[1] = (top_right[0] - sample_loc[0])/line_direction[0];
    intersections[2] = (bottom_left[1] - sample_loc[1])/line_direction[1];
    intersections[3] = (top_right[1] - sample_loc[1])/line_direction[1];

    AScalar min = -1e10, max = 1e10;
    for(AScalar intersection: intersections){
        if(intersection > 0) max = std::min(max, intersection);
        else min = std::max(min, intersection);
    }

    // if there is only one spring
    if(std::abs(bottom_left[1]-top_right[1]) < 1e-8) {
        min = -springs[0]->width;
        max = springs[0]->width;
    }

    auto gaussian_kernel = [=](AScalar distance){
        return std::exp(-0.5*distance*distance/(kernel_std*kernel_std)) / (kernel_std * std::sqrt(2 * M_PI));
    };

    AScalar b = springs[0]->width;
    int density = 4*kernel_std/b;
    int n = 10;
    AScalar weights = 0;
    for(int s = -density; s <= density; ++s){
        AScalar center_line_distance_to_sample = 2*b*s;
        if(center_line_distance_to_sample > max+1e-7 || center_line_distance_to_sample < min-1e-7) continue;
        for (int i = 0; i <= n; ++i) {
            AScalar x = -b + 2.0 * b * i / n;
            weights += gaussian_kernel(center_line_distance_to_sample+x);
        }          
    }
    weights *= 2.0 * b / n;  

    return weights;
}