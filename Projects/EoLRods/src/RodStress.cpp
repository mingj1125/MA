#include "../include/EoLRodSim.h"
#include <Eigen/Geometry>
#include <boost/math/quadrature/gauss_kronrod.hpp>

// we assume rod has a circular cross-section
bool EoLRodSim::lineCutRodinSegment(Rod* rod, TV sample_loc, TV direction, std::vector<int>& cut_segments, std::vector<T>& cut_point_barys){

    unsigned count = 0;
    cut_point_barys = std::vector<T>(0);
    rod->iterateSegmentsWithOffset([&](int node_i, int node_j, Offset offset_i, Offset offset_j, int rod_idx){
        TV Xi, Xj;
        rod->X(node_i, Xi); rod->X(node_j, Xj);

        TV segment_direction = (Xj-Xi).normalized();
        auto r = solveLineIntersection(sample_loc, direction, Xi, Xj);
        if(r(0) > 0.-1e-6 && r(0) < 1.+1e-6){
                cut_segments.push_back(rod_idx); // record cut segemnt index e_i
                cut_point_barys.push_back(r(0));
                ++count;
        }

    });

    if(count > 0) return true;
    return false;
}

Vector<T,2> EoLRodSim::solveLineIntersection(const TV sample_point, const TV line_direction, const TV v1, const TV v2){
    TV e = v1 - v2;
    TV b; b << sample_point-v2;
    Matrix<T, 3, 2> A; A.col(0) = e; A.col(1) = -line_direction;
    TV2 r = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    if(std::abs(e.normalized().transpose()*line_direction.normalized())-1 >= 0.-1e-8) r(0) = -1;

    return r;
}

struct Pair{
    T signed_distance;
    Vector<T, 3> point_location_undeformed;
    Vector<T, 3> point_location_deformed;
    Matrix<T, 6, 3> gradient_wrt_xij;
    Vector<int, 6> offset_maps_xij;

    Pair(T d, Vector<T, 3> loc, Vector<T, 3> loc_deformed, Matrix<T, 6, 3> grads, Vector<int, 6> offset): 
        signed_distance(d), point_location_undeformed(loc), 
        point_location_deformed(loc_deformed), gradient_wrt_xij(grads), offset_maps_xij(offset){}
    bool operator<(const Pair& other) const {
        return signed_distance < other.signed_distance;
    }
};

Matrix<T, 3, 3> EoLRodSim::computeWeightedDeformationGradient(const TV sample_loc, const std::vector<TV> line_directions){
    
    std::vector<TV> dx;
    std::vector<TV> dX;
    std::vector<MatrixXT> diff_b_dx(deformed_states.rows(), MatrixXT(3, line_directions.size())); // size = # directions
    T std = 0.1*unit;
    auto gaussian_kernel = [std](T distance){
        return std::exp(-0.5*distance*distance/(std*std)) / (std * std::sqrt(2 * M_PI));
    };

    int counter_dir = 0;
    for(auto direction : line_directions){
        std::vector<TV> dx_gradients_wrt_x(deformed_states.rows(),TV::Zero());
        bool cut = false;
        std::priority_queue<Pair> intersections;
        T weight_sum = 0;

        for(auto& rod: Rods){
            std::vector<int> cut_segments(0);
            std::vector<T> cut_point_barys;
            if(lineCutRodinSegment(rod, sample_loc, direction, cut_segments, cut_point_barys)){
                cut = true;

                for(int i = 0; i < cut_segments.size(); ++i){
                    TV xi, xj, Xi, Xj;
                    int rod_idx = cut_segments[i];
                    auto node_i = rod->indices[rod_idx];
                    auto node_j = rod->indices[rod_idx+1];
                    rod->X(node_i, Xi); rod->X(node_j, Xj);
                    rod->x(node_i, xi); rod->x(node_j, xj);
                    TV cut_point = Xj + cut_point_barys[i]*(Xi-Xj);
                    TV cut_point_deformed = xj + cut_point_barys[i]*(xi-xj);
                    Vector<int,6> offsets; 
                    offsets.segment<3>(0) = rod->offset_map[node_i].segment<3>(0); 
                    offsets.segment<3>(3) = rod->offset_map[node_j].segment<3>(0);
                    Matrix<T, 6, 3> gradients;
                    gradients.block(0, 0, 3, 3) = MatrixXT::Identity(3,3)*(cut_point_barys[i]);
                    gradients.block(3, 0, 3, 3) = MatrixXT::Identity(3,3)*(1-cut_point_barys[i]);

                    T distance = (cut_point[0] - sample_loc[0])/direction[0];
                    if(abs(direction[0]) < 1e-9) distance =  (cut_point[1] - sample_loc[1])/direction[1];
                    // std::cout << "Distance: " << distance << std::endl;

                    intersections.push(Pair(distance, cut_point, cut_point_deformed, gradients, offsets));
                }
            }
        }
        TV dx_direction = TV::Zero();
        TV dX_direction = TV::Zero();
        while(intersections.size() > 0){
            Pair p1 = intersections.top();
            intersections.pop();
            Pair p2 = intersections.top();
            T distance = 0.5*(p1.signed_distance+p2.signed_distance);
            TV dx_seg = p1.point_location_deformed - p2.point_location_deformed;
            TV dX_seg = p1.point_location_undeformed - p2.point_location_undeformed;

            dx_direction += dx_seg*gaussian_kernel(distance);
            dX_direction += dX_seg*gaussian_kernel(distance);
            weight_sum += gaussian_kernel(distance);

            Vector<int, 6> offset_p1 = p1.offset_maps_xij;
            Vector<int, 6> offset_p2 = p2.offset_maps_xij;
            for(int i = 0; i < 6; ++i){
                dx_gradients_wrt_x[offset_p1(i)] += p1.gradient_wrt_xij.row(i)*gaussian_kernel(distance);
                dx_gradients_wrt_x[offset_p2(i)] -= p2.gradient_wrt_xij.row(i)*gaussian_kernel(distance);
            }
        }
        if(!cut){ 
            std::cout << "No cut found in the direction " << direction.transpose() << "!\n";
            dx.push_back(dx_direction);
            dX.push_back(dX_direction);
        } else {
            dx.push_back(dx_direction/weight_sum);
            dX.push_back(dX_direction/weight_sum);
            for(int i = 0; i < dx_gradients_wrt_x.size(); ++i){
                diff_b_dx[i].col(counter_dir) = dx_gradients_wrt_x[i]/weight_sum;
            }
        }
        ++counter_dir;
    }

    MatrixXT A_dX(3, dX.size()); MatrixXT b_dx(3, dx.size());
    if(dx.size() != dX.size()) std::cout << "Deformation Gradient cannot be solved as the matrix dimension is not compatible!\n";
    for(int i = 0; i < dX.size(); ++i){
        A_dX.col(i) = dX[i];
        b_dx.col(i) = dx.at(i);
    } 
    MatrixXT A = A_dX.transpose();
    MatrixXT b = b_dx.transpose();
    TM x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    auto F = x.transpose();
    // assume constant thickness of the material 
    if(F(2,2) == 0.) F(2,2) = 1;

    F_gradients_wrt_x = std::vector<TM>(deformed_states.rows());
    for(int i = 0; i < deformed_states.rows(); ++i){
        MatrixXT diff_b_i = diff_b_dx[i].transpose();
        TM x_i = (A.transpose()*A).ldlt().solve(A.transpose()*diff_b_i);
        if(x_i(2,2) == 0.) x_i(2,2) = 1;
        F_gradients_wrt_x[i] = x_i.transpose();
    }

    return F;
}

Matrix<T, 3, 3> EoLRodSim::computeSecondPiolaStress(Rod* rod, int rod_idx, TV2 cross_section_coord){

    T E = rod->E;
    T poisson_ratio = rod->poisson_ratio;
    TV xi, xj, Xi, Xj;
    auto node_i = rod->indices[rod_idx];
    auto node_j = rod->indices[rod_idx+1];
    rod->x(node_i, xi); rod->x(node_j, xj);
    rod->X(node_i, Xi); rod->X(node_j, Xj);
    T l = (xi-xj).norm(); T L0 = (Xi-Xj).norm();
    TV N1 = rod->rest_tangents[rod_idx];
    TV n1 = (xj - xi).normalized();
    TV N2 = TV({0,0,1});
    TV n2 = TV({0,0,1});
    TV N3 = N1.cross(N2).normalized();
    TV n3 = n1.cross(n2).normalized();
    T lambda1 = l/L0;
    T lambda2 = 1-poisson_ratio*(lambda1-1);
    T lambda3 = 1-poisson_ratio*(lambda1-1);
    TM F = lambda1 * n1 * N1.transpose() + lambda2 * n2 * N2.transpose() + lambda3 * n3 * N3.transpose();
    // T stretch_strain = lambda1-1;

    // int left_rod = (rod->closed && rod_idx == 0) ? rod->numSeg() - 1 : rod_idx - 1;
    // T bend_strain = stretch_strain;
    // if(left_rod >= 0){
    //     std::cout << "Do bending calc\n";
    //     TV xk, Xk;
    //     auto node_k = rod->indices[left_rod];
    //     rod->x(node_k, xk); rod->X(node_k, Xk);
    //     TV t_left = (xi-xk).normalized();
    //     TV k = 2*t_left.cross(n1) / (1+t_left.dot(n1));
    //     TV n_left = rod->reference_frame_us[left_rod];
    //     TV b_left = t_left.cross(n_left);
    //     TV2 kappa; kappa(0) = k.dot(b_left) + k.dot(n3);
    //     kappa(1) = -(k.dot(n_left)+k.dot(n2));
    //     kappa /= (Xi-Xj).norm() + (Xk-Xi).norm();

    //     // natural curvature
    //     TV T_left = (Xi-Xk).normalized();
    //     TV K = 2*T_left.cross(N1) / (1+T_left.dot(N1));
    //     TV N_left = rod->rest_normals[left_rod];
    //     TV B_left = T_left.cross(N_left);
    //     TV2 Kappa; Kappa(0) = K.dot(B_left) + K.dot(N3);
    //     Kappa(1) = -(K.dot(N_left)+K.dot(N2));
    //     Kappa /= (Xi-Xj).norm() + (Xk-Xi).norm();

    //     bend_strain = (bend_strain*(1-kappa.dot(cross_section_coord)) - (kappa-Kappa).dot(cross_section_coord)) / (1-Kappa.dot(cross_section_coord));
    // }

    // TM cauchy_strain = bend_strain*(n1*n1.transpose()) -poisson_ratio*(bend_strain)*(n2*n2.transpose())-poisson_ratio*(bend_strain)*(n3*n3.transpose());

    T lambda = E*poisson_ratio/((1+poisson_ratio)*(1-2*poisson_ratio));
    T mu = 0.5*E/(1+poisson_ratio);
    // TM cauchy_stress = lambda * cauchy_strain.trace() * TM::Identity() + 2*mu*cauchy_strain;
    // TM F_inv = F.lu().solve(TM::Identity());
    // Matrix<T, 3, 3> S = F.determinant()*F_inv*cauchy_stress*F_inv.transpose();

    TM GreenStrain = 0.5*(F.transpose()*F-TM::Identity());
    Matrix<T, 3, 3> S = lambda * GreenStrain.trace() * TM::Identity() + 2*mu*GreenStrain;
    return S;
}

Vector<T, 3> EoLRodSim::computeWeightedStress(const TV sample_loc, const TV direction, 
    std::vector<TV>& gradients_wrt_thickness, std::vector<TV>& gradients_wrt_nodes, bool diff){

    TV stress = TV::Zero();
    bool cut = false;
    for(auto& rod: Rods){
        std::vector<int> cut_segments(0);
        std::vector<T> cut_point_barys;
        std::vector<TV> cut_points;
        if(lineCutRodinSegment(rod, sample_loc, direction, cut_segments, cut_point_barys)){
            
            cut = true;

            for(int i = 0; i < cut_segments.size(); ++i){
                TV xi, xj, Xi, Xj;
                int rod_idx = cut_segments[i];
                auto node_i = rod->indices[rod_idx];
                auto node_j = rod->indices[rod_idx+1];
                rod->X(node_i, Xi); rod->X(node_j, Xj);
                TV cut_point = Xj + cut_point_barys[i]*(Xi-Xj);
                cut_points.push_back(cut_point);
                // checking if the point in the same rod is already evaluated
                bool valid = true;
                for(int j = 0; j < cut_points.size()-1; ++j){
                    auto p = cut_points[j];
                    if((cut_point-p).norm() < 1e-7) {cut_points.pop_back(); valid = false; break;}
                }
                if(!valid) continue;
                T distance = (cut_point - sample_loc).norm();
                TV normal = direction.cross(TV{0,0,1}).normalized();
                if(cut_point_barys[i] <= 0.){
                    auto rod_d = (Xi-cut_point).normalized();
                    if(normal.dot(rod_d) < 0.) {cut_points.pop_back();continue;}
                } else if(cut_point_barys[i] >= 1.){
                    auto rod_d = (Xj-cut_point).normalized();
                    if(normal.dot(rod_d) < 0.) {cut_points.pop_back();continue;}
                }
                T cos_angle = rod->rest_tangents[rod_idx].dot(normal);

                TV stress_ellipse; 
                // stress over cross section area
                stress_ellipse = integrateOverEllipse(rod, i, cos_angle, normal, distance, gradients_wrt_thickness[rod->rod_id], gradients_wrt_nodes, diff);
                stress += stress_ellipse;
            }
        }
    }
    
    if(!cut){ std::cout << "No cut found in the direction " << direction.transpose() << "!\n";return stress;}
    T weight_sum = integrateKernelOverDomain(sample_loc, direction);
    for(int i = 0; i < Rods.size(); i++){
        gradients_wrt_thickness[i] /= weight_sum;
    }
    for(int i = 0; i < deformed_states.rows(); i++){
        gradients_wrt_nodes[i] /= weight_sum;
    }
    // std::cout << weight_sum << std::endl;
    return stress/weight_sum;
}

Matrix<T, 3, 3> EoLRodSim::findBestStressTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions){

    int c = line_directions.size();
    std::vector<MatrixXT> gradient_t(Rods.size(), MatrixXT(3,c));
    // deformed_states.rows() = # node DoF
    std::vector<MatrixXT> gradient_t_wrt_x(deformed_states.rows(), MatrixXT(3,c));
    MatrixXT n(3, c);
    MatrixXT t(3, c);
    for(int i = 0; i < c; ++i){
        TV direction = line_directions.at(i);
        TV direction_normal; direction_normal = direction.cross(TV{0,0,1});
        direction_normal = direction_normal.normalized(); 

        std::vector<TV> gradient_t_di(Rods.size(), TV::Zero());
        std::vector<TV> gradient_t_wrt_x_di(deformed_states.rows(), TV::Zero());

        t.col(i) = computeWeightedStress(sample_loc, direction, gradient_t_di, gradient_t_wrt_x_di, true);
        // for(int j = 0; j < 3; ++j){if(std::abs(t.col(i)(j))< 1e-10) t.col(i)(j) = 0;}
        n.col(i) = direction_normal;
        // for(int j = 0; j < 3; ++j){if(std::abs(n.col(i)(j))< 1e-10) n.col(i)(j) = 0;}
        for(int j = 0; j < gradient_t.size(); ++j){
            gradient_t[j].col(i) = gradient_t_di[j];
        }
        for(int j = 0; j < gradient_t_wrt_x.size(); ++j){
            gradient_t_wrt_x[j].col(i) = gradient_t_wrt_x_di[j];
        }
    }

    TM fitted_tensor; fitted_tensor.setZero();
    MatrixXT A = MatrixXT::Zero(3*c,6);
    VectorXT b(3*c);
    std::vector<VectorXT> b_diff(Rods.size(), VectorXT(3*c));
    std::vector<VectorXT> b_diff_wrt_x(deformed_states.rows(), VectorXT(3*c));
    for(int i = 0; i < c; ++i){
        MatrixXT A_block = MatrixXT::Zero(3,6);
        TV normal = n.col(i);
        A_block << normal(0), normal(1), normal(2), 0, 0, 0,
                0, normal(0), 0, normal(1), normal(2), 0,
                0, 0, normal(0), 0, normal(1), normal(2);
        A.block(i*3, 0, 3, 6) = A_block;
        b.segment(i*3, 3) = t.col(i);
        for(int j = 0; j < Rods.size(); ++j){
            b_diff[j].segment(i*3, 3) = gradient_t[j].col(i);
        }
        for(int j = 0; j < deformed_states.rows(); ++j){
            b_diff_wrt_x[j].segment(i*3, 3) = gradient_t_wrt_x[j].col(i);
        }
    }
    VectorXT x = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    stress_gradients_wrt_rod_thickness = std::vector<TV> (Rods.size(), TV::Zero());
    stress_gradients_wrt_x = std::vector<TV> (deformed_states.rows(), TV::Zero());
    for(int i = 0; i < Rods.size(); i++){
        VectorXT x = (A.transpose()*A).ldlt().solve(A.transpose()*b_diff[i]);
        stress_gradients_wrt_rod_thickness[i] = {x(0), x(3), 2*x(1)};
    }
    for(int i = 0; i < deformed_states.rows(); i++){
        VectorXT x = (A.transpose()*A).ldlt().solve(A.transpose()*b_diff_wrt_x[i]);
        stress_gradients_wrt_x[i] = {x(0), x(3), 2*x(1)};
    }
    fitted_tensor << x(0), x(1), x(2), 
                    x(1), x(3), x(4),
                    x(2), x(4), x(5);

    return fitted_tensor;
}

// Function to integrate over a single ellipse
Vector<T,3> EoLRodSim::integrateOverEllipse(Rod* rod, const int cut_idx, const T cos_angle, const TV normal, const T center_line_distance_to_sample, 
                TV& gradient_wrt_thickness, std::vector<TV>& gradient_wrt_nodes, bool diff){

    TV integral = {0.0, 0.0, 0.0};
    Vector<T, 18> integral_diff_traction; integral_diff_traction.setZero();
    int n = 10; // Discretization points for x

    T std = 0.1*unit;
    auto gaussian_kernel = [std](T distance){
        return std::exp(-0.5*distance*distance/(std*std)) / (std * std::sqrt(2 * M_PI)); /// width;
    };

    double weights = 0;
    
    T b = rod->b;
    for (int i = 0; i <= n; ++i) {
        double x = -b + 2.0 * b * i / n;
        // T between = 1.0 - (x * x) / (b * b);
        // if(std::abs(between) < 1e-7) between = 0; // numerical reason
        // T y_max = rod->a * std::sqrt(between);
        // T y_max = rod->a; // we consider constant thickness
        auto inner_integral = [=](T y) {
            TV2 cross_section_coord = {x, y};
            auto S = computeSecondPiolaStress(rod, cut_idx, cross_section_coord);
            auto f_val = S * normal;
            double kernel = gaussian_kernel(center_line_distance_to_sample+x);
            return TV{f_val(0) * kernel, f_val(1) * kernel, f_val(2) * kernel};
        };

        integral += inner_integral(0);
        // if(diff && i == 0) gradient_wrt_thickness += inner_integral(0);
        // if(diff && i == n) gradient_wrt_thickness += inner_integral(0);
        if(diff) gradient_wrt_thickness += inner_integral(0)*(-(center_line_distance_to_sample+x))/std/std*x/b;
        weights += gaussian_kernel(center_line_distance_to_sample+x);

        if(diff){
            Matrix<T, 18, 1> diff_traction = SGradientWrtx(rod, cut_idx) *normal;
            double kernel = gaussian_kernel(center_line_distance_to_sample+x);
            integral_diff_traction += kernel * diff_traction;
        }
    }

    integral *= 2.0 * b / n;
    weights *= 2.0 * b / n;
    integral_diff_traction *= 2.0 * b / n;
    gradient_wrt_thickness *= 2.0 * b / n;
    gradient_wrt_thickness += integral/b;
    // std::cout << "Weights in rod: " << weights << std::endl;

    Offset offset_i = rod->offset_map[rod->indices[cut_idx]];
    Offset offset_j = rod->offset_map[rod->indices[cut_idx+1]];
    for(int i = 0; i < 3; ++i){
        gradient_wrt_nodes[offset_i[i]] = integral_diff_traction.segment<3>(i*3);
        gradient_wrt_nodes[offset_j[i]] = integral_diff_traction.segment<3>(i*3+9);
    }

    return integral;
}

T EoLRodSim::integrateKernelOverDomain(const TV sample_loc, const TV line_direction){
    T std = 0.1*unit;
    TV bottom_left, top_right;
    computeUndeformedBoundingBox(bottom_left, top_right);
    std::vector<double> intersections(4);
    intersections[0] = (bottom_left[0] - sample_loc[0])/line_direction[0];
    intersections[1] = (top_right[0] - sample_loc[0])/line_direction[0];
    intersections[2] = (bottom_left[1] - sample_loc[1])/line_direction[1];
    intersections[3] = (top_right[1] - sample_loc[1])/line_direction[1];

    double min = -1e10, max = 1e10;
    for(double intersection: intersections){
        if(intersection > 0) max = std::min(max, intersection);
        else min = std::max(min, intersection);
    }
    if(std::abs(bottom_left[1]-top_right[1]) < 1e-8) {
        min = -Rods[0]->b;
        max = Rods[0]->b;
    }
    auto gaussian_kernel = [std](T distance){
        return std::exp(-0.5*distance*distance/(std*std)) / (std * std::sqrt(2 * M_PI));
    };

    T b = Rods[0]->b;
    int density = 4*std/b;
    int n = 10;
    double weights = 0;
    for(int s = -density; s <= density; ++s){
        double center_line_distance_to_sample = 2*b*s;
        if(center_line_distance_to_sample > max+1e-7 || center_line_distance_to_sample < min-1e-7) continue;
        for (int i = 0; i <= n; ++i) {
            double x = -b + 2.0 * b * i / n;
            // T between = 1.0 - (x * x) / (b * b);
            // if(std::abs(between) < 1e-7) between = 0; // numerical reason
            // T y_max = width * std::sqrt(between);
            // T y_max = rod->a; // we consider constant thickness
            // auto inner_integral = [=](T y) {
            //     double kernel = gaussian_kernel(center_line_distance_to_sample+x);
            //     return kernel;
            // };

            // weights += boost::math::quadrature::gauss_kronrod<double, 45>::integrate(
                    // [=](double y) { return inner_integral(y); }, -y_max, y_max);
            weights += gaussian_kernel(center_line_distance_to_sample+x);
        }          
    }
    weights *= 2.0 * b / n;  

    return weights;
}

Vector<T, 3> EoLRodSim::computeBoundaryStress(){
    VectorXT full_residual(deformed_states.rows());
    full_residual.setZero();
    addStretchingForce(full_residual);
    // add3DBendingAndTwistingForce(full_residual);

    T rec_width = 0.0001 * unit;
    TV bottom_left, top_right;
    TV force = TV::Zero();
    computeBoundingBox(bottom_left, top_right);
    std::vector<bool> node_flag;
    if(n_nodes != -1) node_flag = std::vector<bool>(n_nodes);

    for(auto& rod : Rods)
    {
        for (int idx : rod->indices)
        {   
            if(n_nodes != -1 && node_flag[idx] == true) continue; 
            TV node_i;
            rod->x(idx, node_i);
            Offset map = rod->offset_map[idx];
            if (node_i[0] < bottom_left[0] + rec_width) {
            // if (node_i[0] > top_right[0] - rec_width) {
                // force += full_residual.template segment<3>(map[0]);
                force += full_residual.template segment<3>(map[0]);//*4/M_PI;
                // std::cout << "Force:" << full_residual.template segment<3>(map[0]).transpose()/rod->a/rod->b/M_PI << std::endl;
                // std::cout << "a b:" << rod->a << "  " << rod->b << std::endl;
                if(n_nodes != -1) node_flag[idx] = true;
            }
        }
    }
    // std::cout << full_residual.template segment<3>((Rods[0]->offset_map[411])[0]).transpose() << std::endl;
    T length = top_right[1] - bottom_left[1];
    T thickness = Rods[0]->a*2;

    // deal with straight rod case
    if(length < 1e-8) return force/thickness/Rods[0]->b/M_PI;

    return force/(length*thickness);
}

Vector<T, 3> EoLRodSim::computeVerticalBoundaryStress(){
    VectorXT full_residual(deformed_states.rows());
    full_residual.setZero();
    addStretchingForce(full_residual);
    // add3DBendingAndTwistingForce(full_residual);

    T rec_width = 0.0001 * unit;
    TV bottom_left, top_right;
    TV force = TV::Zero();
    computeUndeformedBoundingBox(bottom_left, top_right);
    std::vector<bool> node_flag;
    if(n_nodes != -1) node_flag = std::vector<bool>(n_nodes);

    for(auto& rod : Rods)
    {
        for (int idx : rod->indices)
        {
            if(n_nodes != -1 && node_flag[idx] == true) continue; 
            TV node_i;
            rod->X(idx, node_i);
            Offset map = rod->offset_map[idx];
            if (node_i[1] < bottom_left[1] + rec_width && node_i[0] > bottom_left[0] && node_i[0]< top_right[0]) {
            // if (node_i[1] > top_right[1] - rec_width) {
                force += full_residual.template segment<3>(map[0]);
                if(n_nodes != -1) node_flag[idx] = true;
            }
        }
    }
    computeBoundingBox(bottom_left, top_right);
    T length = top_right[0] - bottom_left[0];
    T thickness = Rods[0]->a*2;

    return force/(length*thickness);
}

void EoLRodSim::computeUndeformedBoundingBox(TV& bottom_left, TV& top_right) const
{
    bottom_left.setConstant(1e6);
    top_right.setConstant(-1e6);

    for (auto& rod : Rods)
    {
        int cnt = 0;
        for (int idx : rod->indices)
        {
            TV x;
            rod->X(idx, x);
            for (int d = 0; d < 3; d++)
            {
                top_right[d] = std::max(top_right[d], x[d]);
                bottom_left[d] = std::min(bottom_left[d], x[d]);
            }
        }
    }   
}

// nodal stress tensor in mass-spring network
Matrix<T, 3, 3> EoLRodSim::computeNodeStress(const int node_idx){
    
    std::vector<TV> normal;
    int num_directions = 4;
    for(int i = 0; i < num_directions; ++i) {
        double angle = i*2*M_PI/num_directions; 
        normal.push_back(TV{std::cos(angle), std::sin(angle), 0});
    }
    std::vector<TV> traction(normal.size(), TV::Zero());
    for (auto& rod : Rods)
    {   
        for (int i = 0; i < (rod->indices).size(); ++i)
        {   
            T E = rod->E;
            T poisson_ratio = rod->poisson_ratio;
            int idx = rod->indices[i];
            if(node_idx == idx){
                TV xi, xj, Xi, Xj;
                int j;
                auto node_i = rod->indices[i];
                if(i == 0) j = i+1;
                else if(rod->indices.size()>2) j = i+1;
                else j = i-1; 
                auto node_j = rod->indices.at(j);
                // std::cout << "Found neighbor node: " << node_j << std::endl;
                rod->X(node_i, Xi); rod->X(node_j, Xj);
                rod->x(node_i, xi); rod->x(node_j, xj);

                // compute cauchy stress tensor
                T l = (xi-xj).norm(); T L0 = (Xi-Xj).norm();
                TV N1 = (Xj - Xi).normalized();
                TV n1 = (xj - xi).normalized();
                // TV N2 = TV({0,0,1});
                // TV n2 = TV({0,0,1});
                // TV N3 = N1.cross(N2).normalized();
                // TV n3 = n1.cross(n2).normalized();
                T lambda1 = l/L0;
                // T lambda2 = 1-poisson_ratio*(lambda1-1);
                // T lambda3 = 1-poisson_ratio*(lambda1-1);
                // TM F = lambda1 * n1 * N1.transpose() + lambda2 * n2 * N2.transpose() + lambda3 * n3 * N3.transpose();
                T stretch_strain = lambda1-1;
                // TM cauchy_strain = stretch_strain*(TV({1,0,0})*(TV({1,0,0}).transpose())) -poisson_ratio*(stretch_strain)*(TV({0,0,1})*(TV({0,0,1}).transpose()))-poisson_ratio*(stretch_strain)*(TV({0,1,0})*(TV({0,1,0}).transpose()));    

                // T lambda = E*poisson_ratio/((1+poisson_ratio)*(1-2*poisson_ratio));
                // T mu = 0.5*E/(1+poisson_ratio);
                // TM cauchy_stress = lambda * cauchy_strain.trace() * TM::Identity() + 2*mu*cauchy_strain;
                // TM R; R.col(0) = n1; R.col(1) = n3; R.col(2) = n2;
                // cauchy_stress = R*cauchy_stress*R.transpose();
                // TM F_inv = F.lu().solve(TM::Identity());
                // Matrix<T, 3, 3> S = F.determinant()*F_inv*cauchy_stress*F_inv.transpose();

                for(int k = 0; k < normal.size(); ++k){
                    if(N1.dot(normal[k]) > 0-1e-4){
                        // std::cout << "Accounting stress with node " << node_j << " \nrod direction: " << N1.transpose() << "for normal " \
                        // << normal[k].transpose() << " \nwith traction: \n " << (stretch_strain*E*N1/lambda1*N1.dot(normal[k])).transpose() << "\n with strain:\n"<< stretch_strain << std::endl; 
                        // traction[k] += stretch_strain/lambda1*E*N1*N1.dot(normal[k]);
                        traction[k] += (l*l-L0*L0)/(2*L0*L0)*E*N1*N1.dot(normal[k]);
                        // traction[k] += S*normal[k];
                    }   
                    // if(n1.dot(normal[k]) > 0-1e-4){
                    //         traction[k] += stretch_strain*E*n1*n1.dot(normal[k]);
                    // }  
                }

                // only for rods with crossing
                if(rod->indices.size()>2 && i != 0) {
                    j = i-1; 
                    node_j = rod->indices.at(j);
                    // std::cout << "Found neighbor node: " << node_j << std::endl;
                    rod->X(node_i, Xi); rod->X(node_j, Xj);
                    rod->x(node_i, xi); rod->x(node_j, xj);

                    // compute cauchy stress tensor
                    l = (xi-xj).norm(); L0 = (Xi-Xj).norm();
                    N1 = (Xj - Xi).normalized();
                    n1 = (xj - xi).normalized();
                    lambda1 = l/L0;
                    stretch_strain = lambda1-1;

                    for(int k = 0; k < normal.size(); ++k){
                        if(N1.dot(normal[k]) > 0+1e-4){
                            // std::cout << "Accounting stress with node " << node_j << " rod direction: " << n1.transpose() << "for normal " \
                            // << normal[k].transpose() << " with traction: " << (stretch_strain*E*n1/lambda1).transpose() << std::endl; 
                            // traction[k] += S*normal[k];
                            // traction[k] += stretch_strain*E*N1/lambda1*N1.dot(normal[k]);
                            traction[k] += (l*l-L0*L0)/(2*L0*L0)*E*N1*N1.dot(normal[k]);
                        }   
                        // if(n1.dot(normal[k]) > 0-1e-2){
                        //     traction[k] += stretch_strain*E*n1*n1.dot(normal[k]);
                        // }   
                    }
                }
            }
        }
    }
    // std::cout << "Force sum: " << force_sum.transpose() << std::endl;
    int c = normal.size();
    MatrixXT n(3, c);
    MatrixXT t(3, c);
    for(int i = 0; i < c; ++i){
        t.col(i) = traction.at(i);
        n.col(i) = normal.at(i);
        // std::cout << "Stress in direction normal: " << n.col(i).transpose() << " is : \n" << t.col(i).transpose() << std::endl;
    }   
    
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
    TM fitted_tensor; fitted_tensor << x(0), x(1), x(3), 
                        x(1), x(2), x(4),
                        x(3), x(4), x(5);
    return fitted_tensor;                    
}