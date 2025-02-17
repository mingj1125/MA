#include <Eigen/CholmodSupport>
#include <igl/readOBJ.h>
#include <igl/massmatrix.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include "../autodiff/Quadratic2DShell.h"
#include "../include/QuadraticTriangle.h"

#include <cmath> // for pi in cut directions
#include <random> // for rng in sampling for stress probing
#include <set>

// assume planar case for now
void QuadraticTriangle::setProbingLineDirections(unsigned int num_directions){
    direction = std::vector<TV>(num_directions);
    T angle = std::acos(-1) / num_directions;
    for(int i = 0; i < num_directions; ++i){
        direction[i] << std::cos(angle*i), std::sin(angle*i), 0;
    }
}

Matrix<T, 3, 3> QuadraticTriangle::findBestStressTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions){
    int tri = pointInTriangle(sample_loc);
    if(tri == -1) std::cout << "Sample point not in mesh!" << std::endl;
    // std::cout << "Found sample point in triangle: " << tri << std::endl;
    // TM2 F_2D_inv = optimization_homo_target_tensors[tri].lu().solve(TM2::Identity());
    int c = line_directions.size();
    MatrixXT n(3, c);
    MatrixXT t(3, c);
    for(int i = 0; i < c; ++i){
        TV direction = line_directions.at(i);
        TV2 direction_normal_2D; direction_normal_2D << direction(1), -direction(0);
        // direction_normal_2D = F_2D_inv.transpose()*direction_normal_2D;
        TV direction_normal; direction_normal.segment(0,2) = direction_normal_2D;
        // direction_normal = direction_normal.normalized(); 
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

Matrix<T, 2, 2> QuadraticTriangle::findBestStrainTensorviaProbing(const TV sample_loc, const std::vector<TV> line_directions){
    int c = line_directions.size();
    MatrixXT n(3, c);
    VectorXT t(c);
    for(int i = 0; i < c; ++i){
        TV direction = line_directions.at(i);
        t(i) = computeWeightedStrain(sample_loc, direction);
        // if(t(i) >= 1e3 || t(i) <= -1e3) {std::cout << "sample " << sample_loc.transpose() <<  " with direction " << direction.transpose() << " : with strain: " << t(i) << std::endl;} 
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

Matrix<T, 3, 3> QuadraticTriangle::findBestStressTensorviaAveraging(const TV sample_loc){
    T pi = std::acos(-1);
    T std = 7e-3;
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel = [pi, variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / std::sqrt(std::pow((2 * pi), 2)*variance_matrix.determinant());
    };

    T sum = 0.;
    TM stress = TM::Zero();

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);
            
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);

        stress.block(0,0,2,2) += gaussian_kernel(sample_loc, triangleCenterofMass(vertices)) * 
            stress_tensors[face_idx].block(0,0,2,2);
        sum += gaussian_kernel(sample_loc, triangleCenterofMass(vertices));

        if((sample_loc-sample[1]).norm() <= 1e-4) kernel_coloring_avg[face_idx] = gaussian_kernel(sample_loc, triangleCenterofMass(vertices));

    }); 
    return stress/sum;
}

Matrix<T, 3, 3> QuadraticTriangle::findBestStrainTensorviaAveraging(const TV sample_loc){
    T pi = std::acos(-1);
    T std = 7e-3;
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel = [pi, variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / std::sqrt(std::pow((2 * pi), 2)*variance_matrix.determinant());
    };

    T sum = 0.;
    TM strain = TM::Zero();

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);
            
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);

        strain += gaussian_kernel(sample_loc, triangleCenterofMass(vertices)) * strain_tensors[face_idx];
        sum += gaussian_kernel(sample_loc, triangleCenterofMass(vertices));

    }); 
    return strain/sum;
}

Vector<T, 3> QuadraticTriangle::computeWeightedStress(const TV sample_loc, TV direction){
    T pi = std::acos(-1);
    T std = 7e-3;
    int choose_gaussian_kernel = 1;
    auto gaussian_kernel1 = [pi, std](T distance){
        return std::exp(-0.5*distance*distance/(std*std)) / (std * std::sqrt(2 * pi));
    };
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel2 = [pi, variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / std::sqrt(std::pow((2 * pi), 2)*variance_matrix.determinant());
    };
    T sum = 0.;
    TV stress = TV::Zero();
    TV direction_normal; direction_normal << direction(1), -direction(0), 0;
    direction_normal = direction_normal.normalized(); 

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV cut_point_coordinate;
        if(lineCutTriangle(x0, x1, x2, sample_loc, direction, cut_point_coordinate)){
            TV middle = middlePointoflineCutTriangle(x0, x1, x2, cut_point_coordinate);
            if(choose_gaussian_kernel == 1){
                T distance = (sample_loc - middle).norm();
                stress += gaussian_kernel1(distance) * stress_tensors[face_idx] * direction_normal;
                sum += gaussian_kernel1(distance);

                // visulization
                if((sample_loc-sample[1]).norm() <= 1e-4) kernel_coloring_prob[face_idx] = gaussian_kernel1(distance);
            } else if(choose_gaussian_kernel == 2){
                stress += gaussian_kernel2(sample_loc, middle) * stress_tensors[face_idx] * direction_normal;
                sum += gaussian_kernel2(sample_loc, middle);

                // visulization
                if((sample_loc-sample[1]).norm() <= 1e-4) kernel_coloring_prob[face_idx] = gaussian_kernel2(sample_loc, middle);
            }
            
        }
    });
    if (sum <= 0.) std::cout << "Sum is 0 for direction " << direction.transpose() << std::endl; 

    return stress/sum;
}

T QuadraticTriangle::computeWeightedStrain(const TV sample_loc, TV direction){
    T pi = std::acos(-1);
    T std = 7e-3;
    int choose_gaussian_kernel = 1;
    auto gaussian_kernel1 = [pi, std](T distance){
        return std::exp(-0.5*distance*distance/(std*std)) / (std * std::sqrt(2 * pi));
    };
    TM2 variance_matrix; variance_matrix << std*std, 0, 0, std*std;
    auto gaussian_kernel2 = [pi, variance_matrix](TV sample_loc, TV CoM){
        TV2 dist = (CoM-sample_loc).segment(0,2);
        T upper = dist.transpose()*variance_matrix.ldlt().solve(dist);
        return std::exp(-0.5*upper) / std::sqrt(std::pow((2 * pi), 2)*variance_matrix.determinant());
    };
    T sum = 0.;
    T strain = 0;

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);

        TV cut_point_coordinate;
        bool compute_with_segments = true;
        if(lineCutTriangle(x0, x1, x2, sample_loc, direction, cut_point_coordinate)){
            TV middle = middlePointoflineCutTriangle(x0, x1, x2, cut_point_coordinate);
            // if((sample_loc-sample[1]).norm() <= 1e-4) {std::cout << "triangle "<< face_idx << " has strain : " << 
            //         strainInCut(face_idx, cut_point_coordinate) << " in direction: " << direction.transpose()<< 
            //         " with cut in " << cut_point_coordinate.transpose() << std::endl; 
            //         }
            if(choose_gaussian_kernel == 1){
                T distance = (sample_loc - middle).norm();
                if(compute_with_segments)
                    strain += gaussian_kernel1(distance)*strainInCut(face_idx, cut_point_coordinate);
                else strain += gaussian_kernel1(distance)*direction.transpose()*strain_tensors[face_idx]*direction;    
                sum += gaussian_kernel1(distance);

            } else if(choose_gaussian_kernel == 2){
                if(compute_with_segments)
                    strain += gaussian_kernel2(sample_loc, middle)*strainInCut(face_idx, cut_point_coordinate);
                else gaussian_kernel2(sample_loc, middle)*direction.transpose()*strain_tensors[face_idx]*direction; 
                sum += gaussian_kernel2(sample_loc, middle);
            }
        }
    });

    if(sum <= 0) {std::cout << "Weighted strain: "<< strain << " Something wrong with the weighting!\n";} 

    return strain/sum;
}

void QuadraticTriangle::visualizeCuts(const std::vector<TV> sample_points, const std::vector<TV> line_directions){
    unsigned int tag = 1;
    for(auto sample_point: sample_points){
        for(auto direction: line_directions){
            visualizeCut(sample_point, direction, tag);
            ++tag;
        }
    }
}

void QuadraticTriangle::visualizeCut(const TV sample_point, const TV line_direction, unsigned int line_tag){

    iterateFaceSerial([&](int face_idx)
    {
        if(cut_coloring[face_idx] == line_tag) cut_coloring[face_idx] = 0;
        FaceVtx vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);
        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV cut_point_coordinate;
        if(lineCutTriangle(x0, x1, x2, sample_point, line_direction, cut_point_coordinate)){
            cut_coloring[face_idx] = line_tag;
        }
    });
}

// return if a line cut through a triangle and return the cut points' barycentric coordinate
bool QuadraticTriangle::lineCutTriangle(const TV x1, const TV x2, const TV x3, const TV sample_point, const TV line_direction, TV &cut_point_coordinate){
    
    int count = 0;
    cut_point_coordinate << -1, -1, -1;
    std::vector<TV> points(3, TV::Zero());

    TV2 r1 = solveLineIntersection(sample_point, line_direction, x3, x2);
    TV2 r2 = solveLineIntersection(sample_point, line_direction, x2, x1);
    TV2 r3 = solveLineIntersection(sample_point, line_direction, x1, x3);
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

// find middle point of a cut segment through the triangle element
Vector<T,3> QuadraticTriangle::middlePointoflineCutTriangle(const TV x1, const TV x2, const TV x3, const TV cut_point_coordinate){
    
    TV middle_point = TV::Zero();
    if(cut_point_coordinate(0) >= 0.-1e-6 && cut_point_coordinate(0) <= 1.+1e-8) {middle_point += x1 + cut_point_coordinate(0)*(x2-x1);}
    if(cut_point_coordinate(1) >= 0.-1e-6 && cut_point_coordinate(1) <= 1.+1e-8) {middle_point += x2 + cut_point_coordinate(1)*(x3-x2);}
    if(cut_point_coordinate(2) >= 0.-1e-6 && cut_point_coordinate(2) <= 1.+1e-8) {middle_point += x3 + cut_point_coordinate(2)*(x1-x3);}

    return middle_point/2;

}

// calculate strain in current cut segment using barycentric coordinate
T QuadraticTriangle::strainInCut(const int face_idx, const TV cut_point_coordinate){

    FaceVtx vertices = getFaceVtxDeformed(face_idx).block(0,0,3,3);
    FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx).block(0,0,3,3);

    TM cuts = TM::Zero();
    TM cuts_undeformed = TM::Zero();
    std::vector<int> recorder;
    for(int i = 0; i < 3; ++i){
        T cut_point = cut_point_coordinate(i);
        if(cut_point >= 0.-1e-8 && cut_point <= 1.+1e-8) {
            cuts.col(i) = vertices.row(i) + cut_point*(vertices.row((i+1)%3) - vertices.row(i));
            cuts_undeformed.col(i) = undeformed_vertices.row(i) + cut_point*(undeformed_vertices.row((i+1)%3) - undeformed_vertices.row(i));
            recorder.push_back(i);
        }    
    }
    assert(recorder.size() == 2);
    T l = (cuts.col(recorder.at(0)) - cuts.col(recorder.at(1))).norm();
    T L0 = (cuts_undeformed.col(recorder.at(0)) - cuts_undeformed.col(recorder.at(1))).norm();

    return 0.5*(l*l-L0*L0)/(L0*L0);
}

Vector<T,2> QuadraticTriangle::solveLineIntersection(const TV sample_point, const TV line_direction, const TV v1, const TV v2){
    TV e = v1 - v2;
    TV b; b << sample_point-v2;
    Matrix<T, 3, 2> A; A.col(0) = e; A.col(1) = -line_direction;
    TV2 r = (A.transpose()*A).ldlt().solve(A.transpose()*b);
    if(std::abs(e.normalized().transpose()*line_direction.normalized())-1 >= 0.-1e-8) r(0) = -1;

    return r;
}
