#include "../include/EoLRodSim.h"

bool pointInsideWindow(Vector<T,3> point, Vector<T,3> window_top_right, Vector<T,3> window_bottom_left){
    return point(0) >= window_bottom_left(0) && point(0) <= window_top_right(0) && point(1) >= window_bottom_left(1) && point(1) <= window_top_right(1); 
}

// when one end is inside the window
T lengthInsideWindow(Vector<T,3> point_in, Vector<T,3> point_out, Vector<T,3> window_top_right, Vector<T,3> window_bottom_left){
    Vector<T,3> segment_direction = (point_out-point_in).normalized();
    // std::cout << "direction: " << segment_direction.transpose() << std::endl;
    if(point_out(0) > window_top_right(0)){
        Vector<T,3> cross_bp = segment_direction * (window_top_right(0)-point_in(0))/segment_direction(0)+point_in;
        if(cross_bp(1) > window_top_right(1)){
            cross_bp = segment_direction * (window_top_right(1) - point_in(1))/segment_direction(1)+point_in;
            return (cross_bp - point_in).norm();
        } else if(cross_bp(1) < window_bottom_left(1)){
            cross_bp = segment_direction * (window_bottom_left(1) - point_in(1))/segment_direction(1)+point_in;
            return (cross_bp - point_in).norm();
        } else {
            return (cross_bp - point_in).norm();
        }
    }else if(point_out(0) < window_bottom_left(0)){
        Vector<T,3> cross_bp = segment_direction * (window_bottom_left(0)-point_in(0))/segment_direction(0)+point_in;
        if(cross_bp(1) > window_top_right(1)){
            cross_bp = segment_direction * (window_top_right(1) - point_in(1))/segment_direction(1)+point_in;
            return (cross_bp - point_in).norm();
        } else if(cross_bp(1) < window_bottom_left(1)){
            cross_bp = segment_direction * (window_bottom_left(1) - point_in(1))/segment_direction(1)+point_in;
            return (cross_bp - point_in).norm();
        } else {
            return (cross_bp - point_in).norm();
        }
    } else if(point_out(1) < window_bottom_left(1)){
        Vector<T,3> cross_bp = segment_direction * (window_bottom_left(1)-point_in(1))/segment_direction(1)+point_in;
        return (cross_bp - point_in).norm();
    } else if(point_out(1) > window_top_right(1)){
        Vector<T,3> cross_bp = segment_direction * (window_top_right(1)-point_in(1))/segment_direction(1)+point_in;
        return (cross_bp - point_in).norm();
    }
}

Vector<T,2> solveLineIntersection(Vector<T,3> sample_point, const Vector<T,3> line_direction, const Vector<T,3> v1, const Vector<T,3> v2){
    Vector<T,3> e = v1 - v2;
    Vector<T,3> b; b << sample_point-v2;
    Matrix<T, 3, 2> A; A.col(0) = e; A.col(1) = -line_direction;
    Vector<T,2> r = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    if(std::abs(e.normalized().transpose()*line_direction.normalized())-1 >= -1e-8) r(0) = -128;
    if(std::abs(e.normalized().transpose()*line_direction.normalized())-1 >= -1e-8) r(1) = 0.5;

    return r;
}

T partInsideWindow(Vector<T,3> point_A, Vector<T,3> point_B, Vector<T,3> window_top_right, Vector<T,3> window_bottom_left){
    Vector<T, 3> line_dir = point_A-point_B;
    std::vector<Vector<T,3>> window_corner = std::vector<Vector<T,3>>(4, Vector<T,3>::Zero());
    window_corner[0] = window_bottom_left;
    window_corner[1] = Vector<T,3>(window_bottom_left(0), window_top_right(1), 0); // top left
    window_corner[2] = window_top_right;
    window_corner[3] = Vector<T,3>(window_top_right(0), window_bottom_left(1), 0); // bottom right
    std::vector<Vector<T,2>> intersections(4);
    std::vector<Vector<T,3>> sec_p;
    for(int i = 0; i < 4; ++i){
        intersections[i] = solveLineIntersection(point_B, line_dir, window_corner[(i+1)%4], window_corner[i]);
        if(intersections[i](1) < 1 && intersections[i](1) > 0){
            if(intersections[i](0) <= 1 && intersections[i](0) >= 0){
                sec_p.push_back(point_B+intersections[i](1)*line_dir);
            }
        }
    }
    if(sec_p.size() == 2) return (sec_p[0]-sec_p[1]).norm();
    // Vector<T,2> r_left = solveLineIntersection(point_B, line_dir, window_corner[1], window_corner[0]);
    // Vector<T,2> r_top = solveLineIntersection(point_B, line_dir, window_corner[2], window_corner[1]);
    // Vector<T,2> r_right = solveLineIntersection(point_B, line_dir, window_corner[3], window_corner[2]);
    // Vector<T,2> r_down = solveLineIntersection(point_B, line_dir, window_corner[0], window_corner[3]);

    return 0;
}

// assume test window is much larger than single rod
Matrix<T, 3, 3> EoLRodSim::computeWindowHomogenization(TV window_top_right, TV window_bottom_left){
    
    TM homogenised_S = TM::Zero();
    TM S = TM::Zero();
    T total_area = 0;
    // check for all segments
    for(auto& rod: Rods){

        for (int i = 0; i < (rod->indices).size()-1; ++i){
            TV node_i, node_j;
            rod->X(rod->indices[i], node_i); rod->X(rod->indices[i+1], node_j);
            if(pointInsideWindow(node_i, window_top_right, window_bottom_left)){
                if(pointInsideWindow(node_j, window_top_right, window_bottom_left)){
                    homogenised_S += computeSecondPiolaStress(rod, i, {0,0}) * (node_i-node_j).norm()*2*rod->a;
                    // total_area += (node_i-node_j).norm()*2*rod->a;
                } else {
                    homogenised_S += computeSecondPiolaStress(rod, i, {0,0}) * 
                    lengthInsideWindow(node_i, node_j, window_top_right, window_bottom_left)*2*rod->a;
                    // total_area += lengthInsideWindow(node_i, node_j, window_top_right, window_bottom_left)*2*rod->a;
                    // std::cout <<  lengthInsideWindow(node_i, node_j, window_top_right, window_bottom_left) << std::endl;
                    // std::cout <<  2*rod->a << std::endl;
                }
                // std::cout << computeSecondPiolaStress(rod, i, {0,0}) << std::endl;
            } 
            else if(pointInsideWindow(node_j, window_top_right, window_bottom_left)){
                if(!pointInsideWindow(node_i, window_top_right, window_bottom_left)){
                    homogenised_S += computeSecondPiolaStress(rod, i, {0,0}) * 
                    lengthInsideWindow(node_j, node_i, window_top_right, window_bottom_left)*2*rod->a;
                    // total_area += lengthInsideWindow(node_j, node_i, window_top_right, window_bottom_left)*2*rod->a;
                    // std::cout <<  lengthInsideWindow(node_j, node_i, window_top_right, window_bottom_left) << std::endl;
                }
                // std::cout << computeSecondPiolaStress(rod, i, {0,0}) << std::endl;
            } else {
                // part of the segment inside the window
                homogenised_S += computeSecondPiolaStress(rod, i, {0,0}) * 
                    partInsideWindow(node_j, node_i, window_top_right, window_bottom_left)*2*rod->a;
                // S += computeSecondPiolaStress(rod, i, {0,0}) * (partInsideWindow(node_j, node_i, window_top_right, window_bottom_left) > 0);    
            }
        }
    }
    // std::cout << S << std::endl;

    T total_volume = (window_top_right-window_bottom_left)(0) * (window_top_right-window_bottom_left)(1);
    return homogenised_S / total_volume;
}