#include "../include/LinearShell.h"
#include <iostream>

Eigen::Vector2i test_point_in_window(Vector2a point, Vector2a max_corner, Vector2a min_corner){
    Eigen::Vector2i result;
    result.setZero();
    // check if point is in the window
    if(point(0) > max_corner(0)){
        result(0) = 1;
    }
    else if(point(0) < min_corner(0)){
        result(0) = -1;
    }
    if(point(1) > max_corner(1)){
        result(1) = 1;
    }
    else if(point(1) < min_corner(1)){
        result(1) = -1;
    }
    return result;
}

Vector2a find_line_intersection(Vector2a p1, Vector2a p2, Vector2a p3, Vector2a p4){
    // line 1: p1 + t*(p2-p1)
    // line 2: p3 + s*(p4-p3)
    // t = (p3-p1).cross(p4-p3) / (p2-p1).cross(p4-p3)
    // s = (p3-p1).cross(p2-p1) / (p4-p3).cross(p2-p1)

    Vector2a d1 = p2 - p1;
    Vector2a d2 = p4 - p3;
    Vector2a d3 = p3 - p1;

    Matrix2a cross_d1_d2;
    cross_d1_d2.col(0) = d1;
    cross_d1_d2.col(1) = -d2;
    Vector2a result = cross_d1_d2.inverse() * d3;
    // check if t and s are in [0, 1]   
    if(result(0) < 0 || result(0) > 1 || result(1) < 0 || result(1) > 1){
        Vector2a minus_one; minus_one.setConstant(-1);
        return minus_one;
    }
    // return the intersection point
    Vector2a intersection = p1 + result(0) * d1;
    return intersection;
}

AScalar LinearShell::computeTriangleAreaInWindow(const Vector2a max_corner, const Vector2a min_corner, int face_id){

    std::vector<Vector2a> polygon;
    for(int i = 0; i < 3; ++i){
        Vector2a v0 = rest_states.segment<2>(faces(face_id, i)*3);
        Vector2a v1 = rest_states.segment<2>(faces(face_id, (i+1)%3)*3);
        Eigen::Vector2i v0_in_window = test_point_in_window(v0, max_corner, min_corner);
        Eigen::Vector2i v1_in_window = test_point_in_window(v1, max_corner, min_corner);

        if(v0_in_window.norm() < 1e-9){
            polygon.push_back(v0);
        }
        Vector2a intersection_1 = find_line_intersection(v0, v1, max_corner, Vector2a{max_corner(0), min_corner(1)});
        Vector2a intersection_2 = find_line_intersection(v0, v1, max_corner, Vector2a{min_corner(0), max_corner(1)});
        Vector2a intersection_3 = find_line_intersection(v0, v1, min_corner, Vector2a{min_corner(0), max_corner(1)});
        Vector2a intersection_4 = find_line_intersection(v0, v1, min_corner, Vector2a{max_corner(0), min_corner(1)});
        std::vector<Vector2a> polygon_candidates;
        if(intersection_1(0) > 0 && intersection_1(1) > 0){
            polygon_candidates.push_back(intersection_1);
        }
        if(intersection_2(0) > 0 && intersection_2(1) > 0){
            polygon_candidates.push_back(intersection_2);
        }
        if(intersection_3(0) > 0 && intersection_3(1) > 0){
            polygon_candidates.push_back(intersection_3);
        }
        if(intersection_4(0) > 0 && intersection_4(1) > 0){
            polygon_candidates.push_back(intersection_4);
        }
        std::sort(polygon_candidates.begin(), polygon_candidates.end(), [v0](const Vector2a& a, const Vector2a& b){
            return (a-v0).norm() < (b-v0).norm();
        });
        for(int j = 0; j < polygon_candidates.size(); ++j){
            polygon.push_back(polygon_candidates[j]);
        }
        if(v1_in_window.norm() < 1e-9){
            polygon.push_back(v1);
        }
    }

    // test if the corner points are in the polygon
    Vector2a c1 = min_corner;
    Vector2a c2 = Vector2a{min_corner(0), max_corner(1)};
    Vector2a c3 = max_corner;
    Vector2a c4 = Vector2a{max_corner(0), min_corner(1)};

    if(pointInTriangle(c1, face_id)){
        polygon.push_back(c1);
    }
    if(pointInTriangle(c2, face_id)){
        polygon.push_back(c2);
    }
    if(pointInTriangle(c3, face_id)){
        polygon.push_back(c3);  
    }
    if(pointInTriangle(c4, face_id)){
        polygon.push_back(c4);
    }
    // Sort polygon vertices into clockwise order
    Vector2a centroid = Vector2a::Zero();
    for (const auto& vertex : polygon) {
        centroid += vertex;
    }
    centroid /= polygon.size();

    std::sort(polygon.begin(), polygon.end(), [&centroid](const Vector2a& a, const Vector2a& b) {
        AScalar angle_a = std::atan2(a(1) - centroid(1), a(0) - centroid(0));
        AScalar angle_b = std::atan2(b(1) - centroid(1), b(0) - centroid(0));
        return angle_a < angle_b;
    });


    if(polygon.size() <= 2){
        return 0;
    }
    else{
        AScalar area = 0;
        for(int i = 0; i < polygon.size(); ++i){
            Vector2a v0 = polygon[i];
            Vector2a v1 = polygon[(i+1)%polygon.size()];
            area += (v0(0)*v1(1) - v0(1)*v1(0))/2.;
            // std::cout << "area: " << std::abs(area) << std::endl;
        }
        return std::abs(area);
    }
}

AScalar LinearShell::computeWindowArea(const Vector2a max_corner, const Vector2a min_corner){
    AScalar area = (max_corner(0) - min_corner(0)) * (max_corner(1) - min_corner(1));
    return area;
}

Matrix3a LinearShell::findStressTensorinWindow(const Vector2a max_corner, const Vector2a min_corner){
    Matrix2a stress_tensor = Matrix2a::Zero();
    AScalar total_area = 0;

    eval_info_of_sample.stress_gradients_wrt_parameter.resize(3, faces.rows());
    eval_info_of_sample.stress_gradients_wrt_x.resize(3, deformed_states.rows());
    eval_info_of_sample.stress_gradients_wrt_parameter.setZero();
    eval_info_of_sample.stress_gradients_wrt_x.setZero();

    for(int i = 0; i < faces.rows(); ++i){
        AScalar area = computeTriangleAreaInWindow(max_corner, min_corner, i);
        // if(area > 1e-9){
            stress_tensor += stress_tensors_each_element[i] * area;
            total_area += area;
        // }
        eval_info_of_sample.stress_gradients_wrt_parameter.col(i) = Vector3a({stress_tensors_each_element[i](0,0), stress_tensors_each_element[i](1,1), stress_tensors_each_element[i](1,0)})/youngsmodulus_each_element[i] * area;
        Eigen::Vector3i indices = faces.row(i);
        std::vector<Matrix2a> dSdxs = SGradientWrtx(i);
        for(int l = 0; l < dSdxs.size()/3; ++l){
            for(int k = 0; k < 2; ++k){
                Matrix2a dSdx = dSdxs[3*l+k] * area;
                eval_info_of_sample.stress_gradients_wrt_x.col(indices(l)*3+k) += Vector3a({dSdx(0,0), dSdx(1,1), dSdx(1,0)});
            }
        }

    }
    for(int i = 0; i < faces.rows(); i++){
        eval_info_of_sample.stress_gradients_wrt_parameter.col(i) /= total_area;
    }
    for(int i = 0; i < deformed_states.rows(); i++){
        eval_info_of_sample.stress_gradients_wrt_x.col(i) /= total_area;
    }

    stress_tensor /= total_area;
    Matrix3a stress_tensor_3d;
    stress_tensor_3d << stress_tensor(0,0), stress_tensor(0,1), 0,
                        stress_tensor(1,0), stress_tensor(1,1), 0,
                        0, 0, 0;
    eval_info_of_sample.stress_tensor = stress_tensor_3d;

    return stress_tensor_3d;
}

Matrix3a LinearShell::findStrainTensorinWindow(const Vector2a max_corner, const Vector2a min_corner){
    Matrix2a strain_tensor = Matrix2a::Zero();
    AScalar total_area = 0;

    eval_info_of_sample.strain_gradients_wrt_x.resize(3, deformed_states.rows());
    eval_info_of_sample.strain_gradients_wrt_x.setZero();

    for(int i = 0; i < faces.rows(); ++i){
        AScalar area = computeTriangleAreaInWindow(max_corner, min_corner, i);
        // if(area > 1e-9){
            strain_tensor += strain_tensors_each_element[i] * area;
            total_area += area;
        // }
        Eigen::Vector3i indices = faces.row(i);
        std::vector<Matrix2a> dGSdxs = GSGradientWrtx(i);
        for(int l = 0; l < dGSdxs.size()/3; ++l){
            for(int k = 0; k < 2; ++k){
                Matrix2a dGSdx = dGSdxs[3*l+k] * area;
                eval_info_of_sample.strain_gradients_wrt_x.col(indices(l)*3+k) += Vector3a({dGSdx(0,0), dGSdx(1,1), 2*dGSdx(1,0)});
            }
        }
    }
    strain_tensor /= total_area;
    for(int i = 0; i < deformed_states.rows(); i++){
        eval_info_of_sample.strain_gradients_wrt_x.col(i) /= total_area;
    }
    // std::cout << "strain tensor: " << strain_tensor << std::endl;
    Matrix3a strain_tensor_3d;
    strain_tensor_3d << strain_tensor(0,0), strain_tensor(0,1), 0,
                        strain_tensor(1,0), strain_tensor(1,1), 0,
                        0, 0, 0;
    eval_info_of_sample.strain_tensor = strain_tensor_3d;
    return strain_tensor_3d;
}

bool LinearShell::pointInTriangle(const Vector2a sample_loc, int face_id){
    Eigen::Vector3i indices = faces.row(face_id);
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
        return true;
    }
    return false;
}