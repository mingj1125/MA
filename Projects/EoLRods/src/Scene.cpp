#include "../include/Scene.h"
#include "../include/Util.h"
#include "../include/HybridC2Curve.h"

// for processing mesh to rod network
#include <igl/readOBJ.h>
#include <unordered_set>

// static double ROD_A = 3e-4;
// static double ROD_B = 3e-4;

void Scene::buildOneCrossSceneCurved(int sub_div)
{
    int sub_div_2 = sub_div / 2;
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.new_frame_work = true;

    clearSimData();
    std::vector<Eigen::Triplet<T>> w_entry;
    int full_dof_cnt = 0;
    int node_cnt = 0;
    std::unordered_map<int, Offset> offset_map;
    
    TV from(0.0, 0.5, 0.0);
    TV to(1.0, 0.5, 0.0);
    from *= sim.unit; to *= sim.unit;

    TV center = TV(0.5, 0.5, 0.0) * sim.unit;
    int center_id = 0;
    deformed_states.resize(3);
    deformed_states.template segment<3>(full_dof_cnt) = center;
    offset_map[node_cnt] = Offset::Zero();
    for (int d = 0; d < 3; d++) offset_map[node_cnt][d] = full_dof_cnt++;
    node_cnt++;
    auto center_offset = offset_map[center_id];

    std::vector<TV> points_on_curve;
    std::vector<int> rod0;
    std::vector<int> key_points_location_rod0;

    addStraightYarnCrossNPoints(from, to, {center}, {0}, sub_div, points_on_curve, rod0, key_points_location_rod0, node_cnt);

    deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (3 + 1));

    Rod* r0 = new Rod(deformed_states, sim.rest_states, 0, false, ROD_A, ROD_B);

    for (int i = 0; i < points_on_curve.size(); i++)
    {
        offset_map[node_cnt] = Offset::Zero();
        
        //push Lagrangian DoF
        
        deformed_states.template segment<3>(full_dof_cnt) = points_on_curve[i];
        
        for (int d = 0; d < 3; d++)
        {
            offset_map[node_cnt][d] = full_dof_cnt++;    
        }
        // push Eulerian DoF
        deformed_states[full_dof_cnt] = (points_on_curve[i] - from).norm() / (to - from).norm();
        offset_map[node_cnt][3] = full_dof_cnt++;
        node_cnt++;
    }
    deformed_states.conservativeResize(full_dof_cnt + 1);
    deformed_states[full_dof_cnt] = (center - from).norm() / (to - from).norm();
    offset_map[center_id][3] = full_dof_cnt++;

    r0->offset_map = offset_map;
    r0->indices = rod0;

    Vector<T, 3 + 1> q0, q1;
    r0->frontDoF(q0); r0->backDoF(q1);
    r0->rest_state = new LineCurvature(q0, q1);
    
    r0->dof_node_location = key_points_location_rod0;
    sim.Rods.push_back(r0);

    offset_map.clear();
    
    TV rod1_from(0.5, 0.0, 0.0);
    TV rod1_to(0.5, 1.0, 0.0);
    rod1_from *= sim.unit; rod1_to *= sim.unit;

    points_on_curve.clear();
    points_on_curve.resize(0);
    std::vector<int> rod1;
    std::vector<int> key_points_location_rod1;

    addStraightYarnCrossNPoints(rod1_from, rod1_to, {center}, {0}, sub_div, points_on_curve, rod1, key_points_location_rod1, node_cnt);

    deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (3 + 1));

    Rod* r1 = new Rod(deformed_states, sim.rest_states, 1, false, ROD_A, ROD_B);
    for (int i = 0; i < points_on_curve.size(); i++)
    {
        offset_map[node_cnt] = Offset::Zero();
        //push Lagrangian DoF
        deformed_states.template segment<3>(full_dof_cnt) = points_on_curve[i];
        // std::cout << points_on_curve[i].transpose() << std::endl;
        for (int d = 0; d < 3; d++)
        {
            offset_map[node_cnt][d] = full_dof_cnt++;    
        }
        // push Eulerian DoF
        deformed_states[full_dof_cnt] = (points_on_curve[i] - rod1_from).norm() / (rod1_to - rod1_from).norm();
        offset_map[node_cnt][3] = full_dof_cnt++;
        node_cnt++;
    }

    deformed_states.conservativeResize(full_dof_cnt + 1);

    deformed_states[full_dof_cnt] = (center - rod1_from).norm() / (rod1_to - rod1_from).norm();
    offset_map[center_id] = Offset::Zero();
    offset_map[center_id].template segment<3>(0) = center_offset.template segment<3>(0);
    offset_map[center_id][3] = full_dof_cnt++;

    r1->offset_map = offset_map;
    r1->indices = rod1;

    r1->frontDoF(q0); r1->backDoF(q1);
    r1->rest_state = new LineCurvature(q0, q1);
    
    r1->dof_node_location = key_points_location_rod1;
    sim.Rods.push_back(r1);

    RodCrossing* rc0 = new RodCrossing(0, {0, 1});
    rc0->sliding_ranges = { Range(0.2, 0.2), Range(0.2, 0.2)};
    rc0->on_rod_idx[0] = key_points_location_rod0[0];
    rc0->on_rod_idx[1] =  key_points_location_rod1[0];
    sim.rod_crossings.push_back(rc0);


    int dof_cnt = 0;
    markCrossingDoF(w_entry, dof_cnt);
    r0->markDoF(w_entry, dof_cnt);
    r1->markDoF(w_entry, dof_cnt);

    r0->theta_dof_start_offset = full_dof_cnt;
    r0->theta_reduced_dof_start_offset = dof_cnt;        
    int theta_reduced_dof_offset0 = dof_cnt;
    deformed_states.conservativeResize(full_dof_cnt + r0->indices.size() - 1);
    for (int i = 0; i < r0->indices.size() - 1; i++)
    {
        w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
    }   
    deformed_states.template segment(r0->theta_dof_start_offset, 
        r0->indices.size() - 1).setZero();

    r1->theta_dof_start_offset = full_dof_cnt;
    
    int theta_reduced_dof_offset1 = dof_cnt;
    r1->theta_reduced_dof_start_offset = dof_cnt;
    deformed_states.conservativeResize(full_dof_cnt + r1->indices.size() - 1);
    for (int i = 0; i < r1->indices.size() - 1; i++)
    {
        w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
    }   
    deformed_states.template segment(r1->theta_dof_start_offset, 
        r1->indices.size() - 1).setZero();

    deformed_states.conservativeResize(full_dof_cnt + sim.rod_crossings.size() * 3);
    deformed_states.template segment(full_dof_cnt, sim.rod_crossings.size() * 3).setZero();

    for (auto& crossing : sim.rod_crossings)
    {
        crossing->dof_offset = full_dof_cnt;
        crossing->reduced_dof_offset = dof_cnt;
        for (int d = 0; d < 3; d++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }
    }
    
    sim.rest_states = sim.deformed_states;
    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());

    // std::cout << "r0->theta_dof_start_offset " << r0->theta_dof_start_offset << " sim.W.cols() " << sim.W.cols() << std::endl;
    

    Offset ob, of;
    r0->backOffsetReduced(ob);
    r0->frontOffsetReduced(of);

    // std::cout << ob.transpose() << " " << of.transpose() << std::endl;
    r0->fixEndPointEulerian(sim.dirichlet_dof);
    r1->fixEndPointEulerian(sim.dirichlet_dof);

    // r1->fixEndPointLagrangian(sim.dirichlet_dof);

    // sim.fixCrossing();

    sim.dirichlet_dof[ob[0]] = -0.4 * sim.unit;
    sim.dirichlet_dof[ob[1]] = 0.2 * sim.unit;
    // sim.dirichlet_dof[ob[2]] = 0;
    sim.dirichlet_dof[ob[2]] = 0.0 * sim.unit;


    sim.dirichlet_dof[theta_reduced_dof_offset0] = 0;
    sim.dirichlet_dof[theta_reduced_dof_offset1] = 0;

    Offset ob1, of1;
    r1->backOffsetReduced(ob1);
    r1->frontOffsetReduced(of1);


    sim.dirichlet_dof[ob1[0]] = 0 * sim.unit;
    sim.dirichlet_dof[ob1[1]] = 0 * sim.unit;
    sim.dirichlet_dof[ob1[2]] = 0 * sim.unit;

    for (int d = 0; d < 3; d++)
    {
        sim.dirichlet_dof[of[d]] = 0;
        sim.dirichlet_dof[ob1[d]] = 0;
        sim.dirichlet_dof[of1[d]] = 0;
    }

    sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][0]]] = 0.0 * sim.unit;
    sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][1]]] = 0.0 * sim.unit;
    sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][2]]] = 0.0 * sim.unit;
    
    for (auto& rod : sim.Rods)
    {
        rod->setupBishopFrame();
    }
    sim.dq = VectorXT::Zero(dof_cnt);
}

void Scene::buildOneCrossScene(int sub_div)
{
    int sub_div_2 = sub_div / 2;
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.new_frame_work = true;

    clearSimData();
    std::vector<Eigen::Triplet<T>> w_entry;
    int full_dof_cnt = 0;
    int node_cnt = 0;
    std::unordered_map<int, Offset> offset_map;
    
    TV from(0.0, 0.5, 0.0);
    TV to(1.0, 0.5, 0.0);
    from *= sim.unit; to *= sim.unit;

    TV center = TV(0.5, 0.5, 0.0) * sim.unit;
    int center_id = 0;
    deformed_states.resize(3);
    deformed_states.template segment<3>(full_dof_cnt) = center;
    offset_map[node_cnt] = Offset::Zero();
    for (int d = 0; d < 3; d++) offset_map[node_cnt][d] = full_dof_cnt++;
    node_cnt++;
    auto center_offset = offset_map[center_id];

    std::vector<TV> points_on_curve;
    std::vector<int> rod0;
    std::vector<int> key_points_location_rod0;

    addStraightYarnCrossNPoints(from, to, {center}, {0}, sub_div, points_on_curve, rod0, key_points_location_rod0, node_cnt);

    deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (3 + 1));

    Rod* r0 = new Rod(deformed_states, sim.rest_states, 0, false, ROD_A, ROD_B);

    for (int i = 0; i < points_on_curve.size(); i++)
    {
        offset_map[node_cnt] = Offset::Zero();
        
        //push Lagrangian DoF
        
        deformed_states.template segment<3>(full_dof_cnt) = points_on_curve[i];
        
        for (int d = 0; d < 3; d++)
        {
            offset_map[node_cnt][d] = full_dof_cnt++;    
        }
        // push Eulerian DoF
        deformed_states[full_dof_cnt] = (points_on_curve[i] - from).norm() / (to - from).norm();
        offset_map[node_cnt][3] = full_dof_cnt++;
        node_cnt++;
    }
    deformed_states.conservativeResize(full_dof_cnt + 1);
    deformed_states[full_dof_cnt] = (center - from).norm() / (to - from).norm();
    offset_map[center_id][3] = full_dof_cnt++;

    r0->offset_map = offset_map;
    r0->indices = rod0;

    Vector<T, 3 + 1> q0, q1;
    r0->frontDoF(q0); r0->backDoF(q1);
    r0->rest_state = new LineCurvature(q0, q1);
    
    r0->dof_node_location = key_points_location_rod0;
    sim.Rods.push_back(r0);

    offset_map.clear();
    
    TV rod1_from(0.5, 0.0, 0.0);
    TV rod1_to(0.5, 1.0, 0.0);
    rod1_from *= sim.unit; rod1_to *= sim.unit;

    points_on_curve.clear();
    points_on_curve.resize(0);
    std::vector<int> rod1;
    std::vector<int> key_points_location_rod1;

    addStraightYarnCrossNPoints(rod1_from, rod1_to, {center}, {0}, sub_div, points_on_curve, rod1, key_points_location_rod1, node_cnt);

    deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (3 + 1));

    Rod* r1 = new Rod(deformed_states, sim.rest_states, 1, false, ROD_A, ROD_B);
    for (int i = 0; i < points_on_curve.size(); i++)
    {
        offset_map[node_cnt] = Offset::Zero();
        //push Lagrangian DoF
        deformed_states.template segment<3>(full_dof_cnt) = points_on_curve[i];
        // std::cout << points_on_curve[i].transpose() << std::endl;
        for (int d = 0; d < 3; d++)
        {
            offset_map[node_cnt][d] = full_dof_cnt++;    
        }
        // push Eulerian DoF
        deformed_states[full_dof_cnt] = (points_on_curve[i] - rod1_from).norm() / (rod1_to - rod1_from).norm();
        offset_map[node_cnt][3] = full_dof_cnt++;
        node_cnt++;
    }

    deformed_states.conservativeResize(full_dof_cnt + 1);

    deformed_states[full_dof_cnt] = (center - rod1_from).norm() / (rod1_to - rod1_from).norm();
    offset_map[center_id] = Offset::Zero();
    offset_map[center_id].template segment<3>(0) = center_offset.template segment<3>(0);
    offset_map[center_id][3] = full_dof_cnt++;

    r1->offset_map = offset_map;
    r1->indices = rod1;

    r1->frontDoF(q0); r1->backDoF(q1);
    r1->rest_state = new LineCurvature(q0, q1);
    
    r1->dof_node_location = key_points_location_rod1;
    sim.Rods.push_back(r1);

    RodCrossing* rc0 = new RodCrossing(0, {0, 1});
    rc0->sliding_ranges = { Range(0.2, 0.2), Range(0.2, 0.2)};
    rc0->on_rod_idx[0] = key_points_location_rod0[0];
    rc0->on_rod_idx[1] =  key_points_location_rod1[0];
    sim.rod_crossings.push_back(rc0);


    int dof_cnt = 0;
    markCrossingDoF(w_entry, dof_cnt);
    r0->markDoF(w_entry, dof_cnt);
    r1->markDoF(w_entry, dof_cnt);

    r0->theta_dof_start_offset = full_dof_cnt;
    r0->theta_reduced_dof_start_offset = dof_cnt;        
    int theta_reduced_dof_offset0 = dof_cnt;
    deformed_states.conservativeResize(full_dof_cnt + r0->indices.size() - 1);
    for (int i = 0; i < r0->indices.size() - 1; i++)
    {
        w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
    }   
    deformed_states.template segment(r0->theta_dof_start_offset, 
        r0->indices.size() - 1).setZero();

    r1->theta_dof_start_offset = full_dof_cnt;
    
    int theta_reduced_dof_offset1 = dof_cnt;
    r1->theta_reduced_dof_start_offset = dof_cnt;
    deformed_states.conservativeResize(full_dof_cnt + r1->indices.size() - 1);
    for (int i = 0; i < r1->indices.size() - 1; i++)
    {
        w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
    }   
    deformed_states.template segment(r1->theta_dof_start_offset, 
        r1->indices.size() - 1).setZero();

    deformed_states.conservativeResize(full_dof_cnt + sim.rod_crossings.size() * 3);
    deformed_states.template segment(full_dof_cnt, sim.rod_crossings.size() * 3).setZero();

    for (auto& crossing : sim.rod_crossings)
    {
        crossing->dof_offset = full_dof_cnt;
        crossing->reduced_dof_offset = dof_cnt;
        for (int d = 0; d < 3; d++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }
    }
    
    sim.rest_states = sim.deformed_states;
    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());

    // std::cout << "r0->theta_dof_start_offset " << r0->theta_dof_start_offset << " sim.W.cols() " << sim.W.cols() << std::endl;
    

    Offset ob, of;
    r0->backOffsetReduced(ob);
    r0->frontOffsetReduced(of);

    // std::cout << ob.transpose() << " " << of.transpose() << std::endl;
    r0->fixEndPointEulerian(sim.dirichlet_dof);
    r1->fixEndPointEulerian(sim.dirichlet_dof);

    // r1->fixEndPointLagrangian(sim.dirichlet_dof);

    // sim.fixCrossing();

    sim.dirichlet_dof[ob[0]] = -0.4 * sim.unit;
    sim.dirichlet_dof[ob[1]] = 0.2 * sim.unit;
    // sim.dirichlet_dof[ob[2]] = 0;
    sim.dirichlet_dof[ob[2]] = 0.0 * sim.unit;


    sim.dirichlet_dof[theta_reduced_dof_offset0] = 0;
    sim.dirichlet_dof[theta_reduced_dof_offset1] = 0;

    Offset ob1, of1;
    r1->backOffsetReduced(ob1);
    r1->frontOffsetReduced(of1);


    sim.dirichlet_dof[ob1[0]] = 0 * sim.unit;
    sim.dirichlet_dof[ob1[1]] = 0 * sim.unit;
    sim.dirichlet_dof[ob1[2]] = 0 * sim.unit;

    for (int d = 0; d < 3; d++)
    {
        sim.dirichlet_dof[of[d]] = 0;
        sim.dirichlet_dof[ob1[d]] = 0;
        sim.dirichlet_dof[of1[d]] = 0;
    }

    sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][0]]] = 0.0 * sim.unit;
    sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][1]]] = 0.0 * sim.unit;
    sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][2]]] = 0.0 * sim.unit;
    
    for (auto& rod : sim.Rods)
    {
        rod->setupBishopFrame();
    }
    sim.dq = VectorXT::Zero(dof_cnt);
}

void Scene::buildGridScene(int sub_div, bool bc_data)
{
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.add_pbc_twisting = false;
    sim.add_pbc = false;

    sim.add_contact_penalty=true;
    sim.new_frame_work = true;
    sim.add_eularian_reg = false;
    sim.add_bending = false;
    sim.add_twisting = false;

    sim.ke = 1e-4;

    sim.unit = 0.09;
    
    std::vector<Eigen::Triplet<T>> w_entry;
    int full_dof_cnt = 0;
    int node_cnt = 0;

    int n_row = 20, n_col = 20;

    // push crossings first 
    T dy = 1.0 / n_row * sim.unit;
    T dx = 1.0 / n_col * sim.unit;
    
    //num of crossing
    deformed_states.resize(n_col * n_row * 3);
    
    std::unordered_map<int, Offset> crossing_offset_copy;

    auto getXY = [=](int row, int col, T& x, T& y)
    {
        if (row == 0) y = 0.5 * dy;
        else if (row == n_row) y = n_row * dy;
        else y = 0.5 * dy + (row ) * dy;
        if (col == 0) x = 0.5 * dx;
        else if (col == n_col) x = n_col * dx;
        else x = 0.5 * dx + (col ) * dx;
    };


    for (int row = 0; row < n_row; row++)
    {
        for (int col = 0; col < n_col; col++)
        {
            T x, y;
            getXY(row, col, x, y);
            deformed_states.template segment<3>(node_cnt * 3) = TV(x, y, 0);
            
            full_dof_cnt += 3;
            node_cnt ++;       
        }
    }

    int rod_cnt = 0;
    for (int row = 0; row < n_row; row++)
    {
        T x0 = 0.0, x1 = 1.0 * sim.unit;
        T x, y;
        
        std::vector<int> passing_points_id;
        std::vector<TV> passing_points;
        
        for (int col = 0; col < n_col; col++)
        {
            int node_idx = row * n_col + col;
            passing_points_id.push_back(node_idx);
            passing_points.push_back(deformed_states.template segment<3>(node_idx * 3));
        }

        getXY(row, 0, x, y);

        TV from = TV(x0, y, 0);
        TV to = TV(x1, y, 0);
    
        addAStraightRod(from, to, passing_points, passing_points_id, 
            sub_div, full_dof_cnt, node_cnt, rod_cnt);
        
    }
    
    for (int col = 0; col < n_col; col++)
    {
        T y0 = 0.0, y1 = 1.0 * sim.unit;
        T x, y;
        std::vector<int> passing_points_id;
        std::vector<TV> passing_points;
        getXY(0, col, x, y);
        for (int row = 0; row < n_row; row++)
        {
            int node_idx = row * n_col + col;
            passing_points_id.push_back(node_idx);
            passing_points.push_back(deformed_states.template segment<3>(node_idx * 3));
        }
        
        TV from = TV(x, y0, 0);
        TV to = TV(x, y1, 0);

        addAStraightRod(from, to, passing_points, passing_points_id, sub_div, 
                        full_dof_cnt, node_cnt, rod_cnt);
        
    }
    
    for (auto& rod : sim.Rods)
        rod->fixed_by_crossing.resize(rod->dof_node_location.size(), false);

    T dv = 1.0 / n_row;
    T du = 1.0 / n_col;

    int odd_even_cnt = 0;
    for (int row = 0; row < n_row; row++)
    {
        for (int col = 0; col < n_col; col++)
        {
            int node_idx = row * n_col + col;
            RodCrossing* crossing = 
                new RodCrossing(node_idx, {row, n_row + col});

            crossing->on_rod_idx[row] = sim.Rods[row]->dof_node_location[col];
            crossing->on_rod_idx[n_row + col] = sim.Rods[n_row + col]->dof_node_location[row];
            
            // if (odd_even_cnt % 2 == 0)
            //     crossing->is_fixed = true;

            // if (row % 2 == 0)
            //     crossing->is_fixed = true;

            // if (row ==  col)
            //     crossing->is_fixed = true;

            // if (row != col)
            //     crossing->is_fixed = true;

            // crossing->is_fixed = true;

            // if (row == 0 || row == n_row - 1 || col == 0 || col == n_col - 1)                    
                crossing->is_fixed = true;

            sim.Rods[row]->fixed_by_crossing[col] = false;
            sim.Rods[n_row + col]->fixed_by_crossing[row] = false;
            // if (col % 2 == 0)
            {
                // crossing->sliding_ranges.push_back(Range(0, 0));    
                // crossing->sliding_ranges.push_back(Range(1.0/20.0 - 1e3, 1.0/20.0 - 1e3));
                // crossing->sliding_ranges.push_back(Range(1, 1));    
                crossing->sliding_ranges.push_back(Range(1, 1));    
                crossing->sliding_ranges.push_back(Range(0, 0));    
            }
            // else
            // {
            //     crossing->sliding_ranges.push_back(Range(0.02, 0.02));    
            //     crossing->sliding_ranges.push_back(Range(0, 0));
            // }
            
            sim.rod_crossings.push_back(crossing);
            odd_even_cnt++;
        }
    }    

    int dof_cnt = 0;
    markCrossingDoF(w_entry, dof_cnt);

    for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
    
    appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
    
    sim.rest_states = deformed_states;

    
    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
    
    
    for (auto& rod : sim.Rods)
    {
        rod->fixEndPointEulerian(sim.dirichlet_dof);
        rod->setupBishopFrame();
    }
    

    T r = 0.02 * sim.unit;
    TV center1, center2;
    sim.getCrossingPosition(0, center1);
    sim.getCrossingPosition(n_row * n_col - 1, center2);

    TV delta1 = TV(-0.1, -0.1, -1e-2)*2 * sim.unit;

    auto circle1 = [r, center1, delta1](const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {
        mask = Vector<bool, 3>(true, false, true);
        delta = delta1;
        return (x - center1).norm() < r;
    };

    TV delta2 = TV(0.0, 0.0, 0) * sim.unit;
    auto circle2 = [r, center2, delta2](const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {
        mask = Vector<bool, 3>(true, false, true);
        delta = delta2;
        return (x - center2).norm() < r;

    };
    
    TV bottom_left, top_right;
    sim.computeBoundingBox(bottom_left, top_right);
    TV shear_y_down = TV(0., -0.6, 0.0) * sim.unit;
    // TV shear_y_down = TV(std::sqrt(2)*1.6/2, -1.6*std::sqrt(2)/2+1, 0.0) * sim.unit;
    TV shear_y_up = TV(0.0, 0.0, 0) * sim.unit;
    TV shear_y_right = TV(0.1, 0.0, 0.0) * sim.unit;
    TV shear_y_left = TV(-0.1, 0., 0) * sim.unit;
    // TV shear_y_left = TV(1-std::sqrt(2)*1.6/2, -1.6*std::sqrt(2)/2, 0.0) * sim.unit;
    
    T rec_width = 0.0001 * sim.unit;

    auto rec_down = [bottom_left, top_right, shear_y_down, rec_width](
        const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {
        mask = Vector<bool, 3>(true, true, true);
        delta = shear_y_down;
        T one_third = (top_right[1] - bottom_left[1]) / 3.0;
        if (x[1] < bottom_left[1] + rec_width 
            // &&
            // (x[1] > bottom_left[1] + one_third && x[1] < bottom_left[1] + one_third * 2)
            )
            return true;
        return false;
    };

    auto rec_up = [bottom_left, top_right, shear_y_up, rec_width](
        const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {
        mask = Vector<bool, 3>(true, false, true);
        delta = shear_y_up;
        T one_third = (top_right[1] - bottom_left[1]) / 3.0;

        if (x[1] > top_right[1] - rec_width 
            // && 
            // (x[1] > bottom_left[1] + one_third && x[1] < bottom_left[1] + one_third * 2)
            )
            return true;
        return false;
    };

    auto rec_left = [bottom_left, top_right, shear_y_left, rec_width](
        const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {
        mask = Vector<bool, 3>(true, true, true);
        delta = shear_y_left;
        T one_third = (top_right[1] - bottom_left[1]) / 3.0;
        if (x[0] < bottom_left[0] + rec_width 
            // &&
            // (x[1] > bottom_left[1] + one_third && x[1] < bottom_left[1] + one_third * 2)
            )
            return true;
        return false;
    };

    auto rec_right = [bottom_left, top_right, shear_y_right, rec_width](
        const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {
        mask = Vector<bool, 3>(true, false, true);
        delta = shear_y_right;
        T one_third = (top_right[1] - bottom_left[1]) / 3.0;

        if (x[0] > top_right[0] - rec_width 
            // && 
            // (x[1] > bottom_left[1] + one_third && x[1] < bottom_left[1] + one_third * 2)
            )
            return true;
        return false;
    };

    // sim.fixRegionalDisplacement(circle1);
    // sim.fixRegionalDisplacement(circle2);

    // sim.fixRegionalDisplacement(rec_down);
    // sim.fixRegionalDisplacement(rec_up);
    if(!bc_data){
    sim.fixRegionalDisplacement(rec_left);
    sim.fixRegionalDisplacement(rec_right);}


    sim.fixCrossing();

    sim.perturb = VectorXT::Zero(sim.W.cols());
    for (auto& crossing : sim.rod_crossings)
    {
        if (crossing->is_fixed)
            continue;
        Offset off;
        sim.Rods[crossing->rods_involved.front()]->getEntry(crossing->node_idx, off);
        T r = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
        int z_off = sim.Rods[crossing->rods_involved.front()]->reduced_map[off[3-1]];
        sim.perturb[z_off] += 0.001 * (r - 0.5) * sim.unit;
        // sim.perturb[z_off] += 0.001 * r * sim.unit;
    }

    sim.dq = VectorXT::Zero(dof_cnt);
    sim.n_nodes = node_cnt;
}

void Scene::buildFullScaleSquareScene(int sub_div)
{
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    
    clearSimData();

    sim.add_rotation_penalty = true;
    sim.add_pbc_bending = false;
    sim.add_pbc_twisting = false;
    sim.add_pbc = false;

    sim.add_contact_penalty=true;
    sim.new_frame_work = true;
    sim.add_eularian_reg = true;

    sim.ke = 1e-6;

    sim.unit = 0.09;
    sim.visual_R = 0.005;

    std::vector<Eigen::Triplet<T>> w_entry;
    int full_dof_cnt = 0;
    int node_cnt = 0;
    int rod_cnt = 0;

    std::vector<TV> nodal_positions;


    T square_width = 0.012; 
    T overlap = square_width * 0.3;

    auto addCrossingData = [&](int crossing_idx, int rod_idx, int location)
    {
        sim.rod_crossings[crossing_idx]->is_fixed = true;
        sim.rod_crossings[crossing_idx]->rods_involved.push_back(rod_idx);
        sim.rod_crossings[crossing_idx]->on_rod_idx[rod_idx] = location;
        sim.rod_crossings[crossing_idx]->sliding_ranges.push_back(Range::Zero());
    };

    auto addSquare = [&](const TV& bottom_left)
    {
        addCrossingPoint(nodal_positions, bottom_left, full_dof_cnt, node_cnt);
        TV bottom_right = bottom_left + TV(square_width, 0, 0);
        addCrossingPoint(nodal_positions, bottom_right, full_dof_cnt, node_cnt);
        TV top_right = bottom_left + TV(square_width, square_width, 0);
        addCrossingPoint(nodal_positions, top_right, full_dof_cnt, node_cnt);
        TV top_left = bottom_left + TV(0, square_width, 0);
        addCrossingPoint(nodal_positions, top_left, full_dof_cnt, node_cnt);
    };


    auto addInnerSquare = [&](const TV& bottom_left)
    {
        TV v0, v1;

        // add bottom left corner
        addCrossingPoint(nodal_positions, bottom_left, full_dof_cnt, node_cnt);
        v0 = bottom_left + TV(overlap, 0, 0);
        v1 = bottom_left + TV(0, overlap, 0);
        addCrossingPoint(nodal_positions, v0, full_dof_cnt, node_cnt);
        addCrossingPoint(nodal_positions, v1, full_dof_cnt, node_cnt);
        
        // add bottom right corner
        TV bottom_right = bottom_left + TV(square_width, 0, 0);
        addCrossingPoint(nodal_positions, bottom_right, full_dof_cnt, node_cnt);
        v0 = bottom_right - TV(overlap, 0, 0);
        v1 = bottom_right + TV(0, overlap, 0);
        addCrossingPoint(nodal_positions, v0, full_dof_cnt, node_cnt);
        addCrossingPoint(nodal_positions, v1, full_dof_cnt, node_cnt);
        
        TV top_right = bottom_left + TV(square_width, square_width, 0);
        addCrossingPoint(nodal_positions, top_right, full_dof_cnt, node_cnt);
        v0 = top_right - TV(overlap, 0, 0);
        v1 = top_right - TV(0, overlap, 0);
        addCrossingPoint(nodal_positions, v0, full_dof_cnt, node_cnt);
        addCrossingPoint(nodal_positions, v1, full_dof_cnt, node_cnt);
        
        TV top_left = bottom_left + TV(0, square_width, 0);
        addCrossingPoint(nodal_positions, top_left, full_dof_cnt, node_cnt);
        v0 = top_left + TV(overlap, 0, 0);
        v1 = top_left - TV(0, overlap, 0);
        addCrossingPoint(nodal_positions, v0, full_dof_cnt, node_cnt);
        addCrossingPoint(nodal_positions, v1, full_dof_cnt, node_cnt);
        
    };

    auto addRodsForASquare = [&](int bottom_left_node_idx)
    {
        TV bottom_left = nodal_positions[bottom_left_node_idx];
        TV bottom_right = nodal_positions[bottom_left_node_idx + 1];
        TV top_right = nodal_positions[bottom_left_node_idx + 2];
        TV top_left = nodal_positions[bottom_left_node_idx + 3];

        addAStraightRod(bottom_left, bottom_right, 
            {bottom_left, bottom_right}, {bottom_left_node_idx, bottom_left_node_idx + 1},
            sub_div, full_dof_cnt, node_cnt, rod_cnt );
        
        addCrossingData(bottom_left_node_idx, rod_cnt - 1, 0);
        addCrossingData(bottom_left_node_idx + 1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());

        addAStraightRod(bottom_right, top_right, 
            {bottom_right, top_right}, {bottom_left_node_idx + 1, bottom_left_node_idx + 2},
            sub_div, full_dof_cnt, node_cnt, rod_cnt );
        addCrossingData(bottom_left_node_idx + 1, rod_cnt - 1, 0);
        addCrossingData(bottom_left_node_idx + 2, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
        
        addAStraightRod(top_right, top_left, 
            {top_right, top_left}, {bottom_left_node_idx + 2, bottom_left_node_idx + 3},
            sub_div, full_dof_cnt, node_cnt, rod_cnt );
        addCrossingData(bottom_left_node_idx + 2, rod_cnt - 1, 0);
        addCrossingData(bottom_left_node_idx + 3, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());

        addAStraightRod(top_left, bottom_left, 
            {top_left, bottom_left}, {bottom_left_node_idx + 3, bottom_left_node_idx},
            sub_div, full_dof_cnt, node_cnt, rod_cnt );
        addCrossingData(bottom_left_node_idx + 3, rod_cnt - 1, 0);
        addCrossingData(bottom_left_node_idx, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
    };

    int n_row = 5, n_col = 5;
    
    T length_x = square_width * n_col + (square_width - 2.0 * overlap) * (n_col - 1);
    T length_y = length_x;


    // "longer" rows
    for (int row = 0; row < n_row; row++)
    {
        for (int col = 0; col < n_col; col++)
        {
            T left = col * 2.0 * (square_width - overlap);
            T bottom = row * 2.0 * (square_width - overlap);
            TV bottom_left = TV(left, bottom, 0.0);
            addSquare(bottom_left);
        }    
    }

    for (int row = 0; row < n_row - 1; row++)
    {
        for (int col = 0; col < n_col - 1; col++)
        {
            T left = square_width - overlap + col * 2.0 * (square_width - overlap);
            T bottom = square_width - overlap + row * 2.0 * (square_width - overlap);
            TV bottom_left = TV(left, bottom, 0.0);
            // std::cout << bottom_left.transpose() << std::endl;
            addInnerSquare(bottom_left);
        }    
    }

    // std::cout << nodal_positions.size() << std::endl;

    // add Boundary rods first
    // these are the top row and bottom row

    for (int col = 0; col < n_col; col++)
    {
        int idx0 = (0 * n_col + col) * 4; // 4 is four nodes per square
        int idx1 = ((n_row - 1) * n_col + col) * 4; // 4 is four nodes per square
        TV v0 = nodal_positions[idx0], v1 = nodal_positions[idx1 + 3];
        TV v0_next = nodal_positions[idx0 + 1], v1_next = nodal_positions[idx1 + 2]; 
        addAStraightRod(v0, v0_next, 
            {v0, v0_next}, {idx0, idx0 + 1},
            sub_div, full_dof_cnt, node_cnt, rod_cnt );
        addCrossingData(idx0, rod_cnt - 1, 0);
        addCrossingData(idx0 + 1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
    }

    for (int col = n_col - 1; col > -1; col--)
    {
        
        int idx1 = ((n_row - 1) * n_col + col) * 4; // 4 is four nodes per square
        TV v1 = nodal_positions[idx1 + 2];
        TV v1_next = nodal_positions[idx1 + 3]; 
        
        addAStraightRod(v1, v1_next, 
            {v1, v1_next}, {idx1 + 2, idx1 + 3},
            sub_div, full_dof_cnt, node_cnt, rod_cnt );
        addCrossingData(idx1 + 2, rod_cnt - 1, 0);
        addCrossingData(idx1 + 3, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
    }
    
    for (int col = 0; col < n_col; col++)
    {
        auto rod0 = sim.Rods[col];
        auto rod1 = sim.Rods[n_col + n_col - col - 1];
        Offset end0, end1;
        // std::cout << rod0->indices.front() << " " << rod1->indices.back() << std::endl;
        rod0->frontOffset(end0); rod1->backOffset(end1);
        sim.pbc_pairs.push_back(std::make_pair(1, std::make_pair(end0, end1)));
        if (col == 0)
        {
            sim.pbc_pairs_reference[1] = std::make_pair(std::make_pair(end0, end1), 
                std::make_pair(rod0->rod_id, rod1->rod_id));
        }
        rod0->backOffset(end0); rod1->frontOffset(end1);
        // std::cout << rod0->indices.back() << " " << rod1->indices.front() << std::endl;
        sim.pbc_pairs.push_back(std::make_pair(1, std::make_pair(end0, end1)));
    }
    

    // these are the left and right most rows
    for (int row = n_row - 1; row > -1; row--)
    {
        int idx0 = (row * n_col + 0) * 4; // 4 is four nodes per square
        int idx1 = (row * n_col + n_col - 1) * 4; // 4 is four nodes per square
        
        TV v0 = nodal_positions[idx0 + 3];
        TV v0_next = nodal_positions[idx0 + 0];

        addAStraightRod(v0, v0_next, 
            {v0, v0_next}, {idx0 + 3, idx0},
            sub_div, full_dof_cnt, node_cnt, rod_cnt );
        addCrossingData(idx0 + 3, rod_cnt - 1, 0);
        addCrossingData(idx0, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
    }

    
    for (int row = 0; row < n_row; row++)
    {
        
        int idx1 = (row * n_col + n_col - 1) * 4; // 4 is four nodes per square
        
        TV v1 = nodal_positions[idx1 + 1];
        TV v1_next = nodal_positions[idx1 + 2]; 

        addAStraightRod(v1, v1_next, 
            {v1, v1_next}, {idx1 + 1, idx1 + 2},
            sub_div, full_dof_cnt, node_cnt, rod_cnt );
        addCrossingData(idx1 + 1, rod_cnt - 1, 0);
        addCrossingData(idx1 + 2, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
    }

    for (int row = 0; row < n_row; row++)
    {
        auto rod0 = sim.Rods[row + 2 * n_col];
        auto rod1 = sim.Rods[2 * n_col + 2 * n_row - row - 1];

        Offset end0, end1;
        rod0->frontOffset(end0); rod1->backOffset(end1);
        // std::cout << rod0->indices.front() << " " << rod1->indices.back() << std::endl;
        sim.pbc_pairs.push_back(std::make_pair(0, std::make_pair(end0, end1)));
        if (row == 0)
        {
            sim.pbc_pairs_reference[0] = std::make_pair(std::make_pair(end0, end1), 
                std::make_pair(rod0->rod_id, rod1->rod_id));
        }
        rod0->backOffset(end0); rod1->frontOffset(end1);
        sim.pbc_pairs.push_back(std::make_pair(0, std::make_pair(end0, end1)));
        // std::cout << rod0->indices.back() << " " << rod1->indices.front() << std::endl;
    }
    
    for (int col = 0; col < n_col - 1; col++)
    {
        // vertical ones
        for (int row = 0; row < n_row - 1; row++)    
        {
            int idx0 = (row * n_col + col) * 4 + 1; // 4 is four nodes per square
            int idx1 = (row * n_col + col) * 4 + 2; 

            int idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 1;
            int idx_middle2 = n_row * n_col * 4 + ((row + 1) * (n_col - 1) + col) * 12 + 1;
            // std::cout << idx0 << " " << idx_middle1 << " " << idx1 << std::endl;

            TV v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
            TV v_middle1 = nodal_positions[idx_middle1];
            TV v_middle2;
            if (row == 0)
            {
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v1}, {idx0, idx_middle1, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            }
            if (row == n_row - 2)
            {
                idx0 = ((row + 1) * n_col + col) * 4 + 1; // 4 is four nodes per square
                idx1 = ((row + 1) * n_col + col) * 4 + 2; 

                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 10;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1];
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v1}, {idx0, idx_middle1, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());

            }
            if (row != n_row - 2)
            {
                idx0 = ((row + 1) * n_col + col) * 4 + 1; // 4 is four nodes per square
                idx1 = ((row + 1) * n_col + col) * 4 + 2; 
                
                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 10;
                idx_middle2 = n_row * n_col * 4 + ((row + 1) * (n_col - 1) + col) * 12 + 1;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1]; v_middle2 = nodal_positions[idx_middle2];
                
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v_middle2, v1}, {idx0, idx_middle1, idx_middle2,  idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx_middle2, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[2]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            }
        }
    }

    // add inner rods
    for (int col = 0; col < n_col - 1; col++)
    {
        // vertical ones
        for (int row = 0; row < n_row - 1; row++)
        {
            //these are right column of each square
            int idx0 = (row * n_col + col) * 4 + 1; // 4 is four nodes per square
            int idx1 = (row * n_col + col) * 4 + 2; 

            int idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 1;
            int idx_middle2 = n_row * n_col * 4 + ((row + 1) * (n_col - 1) + col) * 12 + 1;
            // std::cout << idx0 << " " << idx_middle1 << " " << idx1 << std::endl;

            TV v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
            TV v_middle1 = nodal_positions[idx_middle1];
            TV v_middle2;
            if (row == 0)
            {
                

                idx0 = (row * n_col + col + 1) * 4 + 0; // 4 is four nodes per square
                idx1 = (row * n_col + col + 1) * 4 + 3; 

                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 4;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1];
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v1}, {idx0, idx_middle1, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            }
            if (row == n_row - 2)
            {
                

                idx0 = ((row + 1) * n_col + col + 1) * 4 + 0; // 4 is four nodes per square
                idx1 = ((row + 1) * n_col + col + 1) * 4 + 3; 

                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 7;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1];
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v1}, {idx0, idx_middle1, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());

            }
            if (row != n_row - 2)
            {
                

                idx0 = ((row + 1) * n_col + col + 1) * 4 + 0; // 4 is four nodes per square
                idx1 = ((row + 1) * n_col + col + 1) * 4 + 3; 

                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 7;
                idx_middle2 = n_row * n_col * 4 + ((row + 1) * (n_col - 1) + col) * 12 + 4;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1]; v_middle2 = nodal_positions[idx_middle2];
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v_middle2, v1}, {idx0, idx_middle1, idx_middle2,  idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx_middle2, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[2]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            }
        }
    }

    
    

    // add inner rods
    for (int row = 0; row < n_row - 1; row++)
    {
        // vertical ones
        for (int col = 0; col < n_col - 1; col++)
        {
            //these are right column of each square
            int idx0 = (row * n_col + col) * 4 + 1; // 4 is four nodes per square
            int idx1 = (row * n_col + col) * 4 + 2; 

            int idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 1;
            int idx_middle2 = n_row * n_col * 4 + ((row + 1) * (n_col - 1) + col) * 12 + 1;
            // std::cout << idx0 << " " << idx_middle1 << " " << idx1 << std::endl;

            TV v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
            TV v_middle1 = nodal_positions[idx_middle1];
            TV v_middle2;
            

            if (col == 0)
            {
                idx0 = (row * n_col + col) * 4 + 3; // 4 is four nodes per square
                idx1 = (row * n_col + col) * 4 + 2; 

                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 2;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1];
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v1}, {idx0, idx_middle1, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            }
            if (col == n_col - 2)
            {
                idx0 = (row * n_col + col + 1) * 4 + 3; // 4 is four nodes per square
                idx1 = (row * n_col + col + 1) * 4 + 2; 

                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 5;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1];
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v1}, {idx0, idx_middle1, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            }
            if (n_col - 2 != col)
            {
                idx0 = (row * n_col + col + 1) * 4 + 3; // 4 is four nodes per square
                idx1 = (row * n_col + col + 1) * 4 + 2; 
                
                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 5;
                idx_middle2 = n_row * n_col * 4 + (row * (n_col - 1) + col + 1) * 12 + 2;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1]; v_middle2 = nodal_positions[idx_middle2];
                
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v_middle2, v1}, {idx0, idx_middle1, idx_middle2,  idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx_middle2, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[2]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            }
        }
    }




    // add inner rods
    for (int row = 0; row < n_row - 1; row++)
    {
        // vertical ones
        for (int col = 0; col < n_col - 1; col++)
        {
            //these are right column of each square
            int idx0 = (row * n_col + col) * 4 + 1; // 4 is four nodes per square
            int idx1 = (row * n_col + col) * 4 + 2; 

            int idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 1;
            int idx_middle2 = n_row * n_col * 4 + ((row + 1) * (n_col - 1) + col) * 12 + 1;
            // std::cout << idx0 << " " << idx_middle1 << " " << idx1 << std::endl;

            TV v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
            TV v_middle1 = nodal_positions[idx_middle1];
            TV v_middle2;
            
            if (col == 0)
            {
                

                idx0 = ((row + 1) * n_col + col) * 4 + 0; // 4 is four nodes per square
                idx1 = ((row + 1) * n_col + col) * 4 + 1; 

                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 11;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1];
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v1}, {idx0, idx_middle1, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            }
            if (col == n_col - 2)
            {
                
                idx0 = ((row + 1) * n_col + col + 1) * 4 + 0; // 4 is four nodes per square
                idx1 = ((row + 1) * n_col + col + 1) * 4 + 1; 

                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 8;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1];
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v1}, {idx0, idx_middle1, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            }
            if (n_col - 2 != col)
            {
                

                idx0 = ((row + 1) * n_col + col + 1) * 4 + 0; // 4 is four nodes per square
                idx1 = ((row + 1) * n_col + col + 1) * 4 + 1; 

                idx_middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 8;
                idx_middle2 = n_row * n_col * 4 + (row * (n_col - 1) + col + 1) * 12 + 11;
                v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
                v_middle1 = nodal_positions[idx_middle1]; v_middle2 = nodal_positions[idx_middle2];
                addAStraightRod(v0, v1, 
                    {v0, v_middle1, v_middle2, v1}, {idx0, idx_middle1, idx_middle2,  idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
                addCrossingData(idx0, rod_cnt - 1, 0);
                addCrossingData(idx_middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
                addCrossingData(idx_middle2, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[2]);
                addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            }
        }
    }



    // std::cout << "inner" << std::endl;
    for (int row = 0; row < n_row - 1; row++)
    {
        for (int col = 0; col < n_col - 1; col++)
        {
            int idx0 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12;
            int idx1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 3; 
            int middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 1; 
            int middle2 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 4; 

            
            TV v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
            TV v_middle1 = nodal_positions[middle1], v_middle2 = nodal_positions[middle2];
            addAStraightRod(v0, v1, 
                    {v0, v_middle1, v_middle2, v1}, {idx0, middle1, middle2, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
            addCrossingData(idx0, rod_cnt - 1, 0);
            addCrossingData(middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
            addCrossingData(middle2, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[2]);
            addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());

            idx0 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 3;
            idx1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 6; 
            middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 5; 
            middle2 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 8; 

            v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
            v_middle1 = nodal_positions[middle1], v_middle2 = nodal_positions[middle2];
            addAStraightRod(v0, v1, 
                    {v0, v_middle1, v_middle2, v1}, {idx0, middle1, middle2, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
            addCrossingData(idx0, rod_cnt - 1, 0);
            addCrossingData(middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
            addCrossingData(middle2, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[2]);
            addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());

            idx0 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 6;
            idx1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 9; 
            middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 7; 
            middle2 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 10; 

            v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
            v_middle1 = nodal_positions[middle1], v_middle2 = nodal_positions[middle2];
            addAStraightRod(v0, v1, 
                    {v0, v_middle1, v_middle2, v1}, {idx0, middle1, middle2, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
            addCrossingData(idx0, rod_cnt - 1, 0);
            addCrossingData(middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
            addCrossingData(middle2, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[2]);
            addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
            
            idx0 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 9;
            idx1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 0; 
            middle1 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 11; 
            middle2 = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12 + 2; 

            v0 = nodal_positions[idx0], v1 = nodal_positions[idx1];
            v_middle1 = nodal_positions[middle1], v_middle2 = nodal_positions[middle2];
            addAStraightRod(v0, v1, 
                    {v0, v_middle1, v_middle2, v1}, {idx0, middle1, middle2, idx1},
                    sub_div, full_dof_cnt, node_cnt, rod_cnt );
            addCrossingData(idx0, rod_cnt - 1, 0);
            addCrossingData(middle1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[1]);
            addCrossingData(middle2, rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[2]);
            addCrossingData(idx1, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
        }
    }

    for (auto& rod : sim.Rods)
        rod->fixed_by_crossing = std::vector<bool>(rod->dof_node_location.size(), true);

    // for (int row = 0; row < n_row - 1; row++)
    // {
    //     for (int col = 0; col < n_col - 1; col++)
    //     {
    //         int base = n_row * n_col * 4 + (row * (n_col - 1) + col) * 12;
    //         for (int corner = 0; corner < 4; corner++)
    //         {
                
    //             auto crossing = sim.rod_crossings[base + corner * 3 + 1];
    //             crossing->is_fixed = false;
    //             crossing->sliding_ranges[1] = Range::Ones();
    //             sim.Rods[crossing->rods_involved[1]]->fixed_by_crossing[1] = false;
    //             sim.Rods[crossing->rods_involved[1]]->fixed_by_crossing[2] = false;

    //             sim.Rods[crossing->rods_involved[0]]->fixed_by_crossing[1] = false;

    //             crossing = sim.rod_crossings[base + corner * 3 + 2];
    //             crossing->is_fixed = false;
    //             crossing->sliding_ranges[0] = Range::Ones();
    //             sim.Rods[crossing->rods_involved[0]]->fixed_by_crossing[1] = false;

    //             sim.Rods[crossing->rods_involved[1]]->fixed_by_crossing[1] = false;
    //             sim.Rods[crossing->rods_involved[1]]->fixed_by_crossing[2] = false;

    //         }
    //     }
    // }

    

    

    int dof_cnt = 0;
    markCrossingDoF(w_entry, dof_cnt);
    // std::cout << "mark dof" << std::endl;
    for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
    
    appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
    
    sim.rest_states = deformed_states;
    
    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
    
    for (auto& rod : sim.Rods)
    {
        rod->fixEndPointEulerian(sim.dirichlet_dof);
        rod->setupBishopFrame();
    }
    
    if (sim.add_pbc)
        sim.Rods[0]->fixPointLagrangianByID(0, TV::Zero(), Mask::Ones(), sim.dirichlet_dof);

    TV bottom_left, top_right;
    sim.computeBoundingBox(bottom_left, top_right);

    TV shear_x_right = TV(0.1, 0.0, 0.0) * length_x;
    TV shear_x_left = TV(0.0, 0.0, 0) * sim.unit;


    T rec_width = 0.015 * sim.unit;

    auto rec1 = [bottom_left, top_right, shear_x_left, rec_width](
        const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {
        mask = Vector<bool, 3>(true, true, true);
        delta = shear_x_left;
        if (x[0] < bottom_left[0] + rec_width 
            )
            return true;
        return false;
    };

    auto rec2 = [bottom_left, top_right, shear_x_right, rec_width](
        const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {
        mask = Vector<bool, 3>(true, true, true);
        delta = shear_x_right;

        if (x[0] > top_right[0] - rec_width)
            return true;
        return false;
    };

    if (!sim.add_pbc)
    {
        sim.fixRegionalDisplacement(rec2);
        sim.fixRegionalDisplacement(rec1);
    }

    sim.fixCrossing();
    // std::cout << "fix crossing" << std::endl;
    sim.perturb = VectorXT::Zero(sim.W.cols());
    for (auto& crossing : sim.rod_crossings)
    {
        Offset off;
        sim.Rods[crossing->rods_involved.front()]->getEntry(crossing->node_idx, off);
        T r = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
        int z_off = sim.Rods[crossing->rods_involved.front()]->reduced_map[off[3-1]];
        sim.perturb[z_off] += 0.001 * (r - 0.5) * sim.unit;
        
    }
    sim.dq = VectorXT::Zero(dof_cnt);
}

void Scene::buildFullCircleScene(int sub_div)
{
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    
    clearSimData();

    sim.add_rotation_penalty = true;
    sim.add_pbc_bending = true;
    sim.add_pbc_twisting = true;
    sim.add_pbc = true;

    sim.add_contact_penalty=true;
    sim.new_frame_work = true;
    sim.add_eularian_reg = true;

    sim.ke = 1e-4;

    sim.unit = 0.01;
    // sim.unit = 1.0;
    sim.visual_R = 0.012;

    auto addCrossingData = [&](int crossing_idx, int rod_idx, int location)
    {
        sim.rod_crossings[crossing_idx]->is_fixed = true;
        sim.rod_crossings[crossing_idx]->rods_involved.push_back(rod_idx);
        sim.rod_crossings[crossing_idx]->on_rod_idx[rod_idx] = location;
        sim.rod_crossings[crossing_idx]->sliding_ranges.push_back(Range::Zero());
    };        

    std::vector<Eigen::Triplet<T>> w_entry;
    int full_dof_cnt = 0;
    int node_cnt = 0;
    int rod_cnt = 0;

    std::vector<TV> nodal_positions;

    T r = 0.55 * sim.unit;

    TV center0 = TV(0, 0, 0) * sim.unit;
    TV center1 = TV(1, 0, 0) * sim.unit;
    TV center2 = TV(1, 1, 0) * sim.unit;
    TV center3 = TV(0, 1, 0) * sim.unit;
    
    std::vector<TV> centers = {center0, center1, center2, center3};

    // add 8 boundary points counterclock wise
    for (int i = 0; i < centers.size(); i++)
    {
        TV current = centers[i], next = centers[(i + 1) % centers.size()];
        TV v0 = next - r  / sim.unit * (next - current);
        TV v1 = current + r / sim.unit * (next - current);
        addCrossingPoint(nodal_positions, v0, full_dof_cnt, node_cnt);
        addCrossingPoint(nodal_positions, v1, full_dof_cnt, node_cnt);
    }
    

    TV vtx8, vtx9, vtx10, vtx11, dummy;
    circleCircleIntersection(center0, r, center1, r, vtx8, dummy);
    circleCircleIntersection(center1, r, center2, r, vtx9, dummy);
    circleCircleIntersection(center2, r, center3, r, vtx10, dummy);
    circleCircleIntersection(center3, r, center0, r, vtx11, dummy);

    addCrossingPoint(nodal_positions, vtx8, full_dof_cnt, node_cnt);
    addCrossingPoint(nodal_positions, vtx9, full_dof_cnt, node_cnt);
    addCrossingPoint(nodal_positions, vtx10, full_dof_cnt, node_cnt);
    addCrossingPoint(nodal_positions, vtx11, full_dof_cnt, node_cnt);


    auto addCurvedRodFromIDs = [&](const std::vector<int>& ids)
    {
        std::vector<TV> passing_points;
        std::vector<TV2> data_points;
        for (int id : ids)
        {
            passing_points.push_back(nodal_positions[id]);
            data_points.push_back(passing_points.back().template head<2>());
        }
        addCurvedRod(data_points, passing_points, ids, sub_div, full_dof_cnt, node_cnt, rod_cnt, false);
        for (int i = 0; i < ids.size(); i++)
        {
            addCrossingData(ids[i], rod_cnt - 1, sim.Rods[rod_cnt-1]->dof_node_location[i]);
        }
    };

    addCurvedRodFromIDs({0, 8, 9, 3});
    addCurvedRodFromIDs({2, 9, 10, 5});
    addCurvedRodFromIDs({4, 10, 11, 7});
    addCurvedRodFromIDs({6, 11, 8, 1});
    

    auto setPBCData = [&](int rod_idx0, int rod_idx1, int direction, bool unique, bool reverse)
    {
        auto rod0 = sim.Rods[rod_idx0];
        auto rod1 = sim.Rods[rod_idx1];
        Offset end0, end1;
        if (reverse)
        {
            rod0->backOffset(end0); rod1->frontOffset(end1);
        }
        else
        {
            rod0->frontOffset(end0); rod1->backOffset(end1);
        }
        if (unique)
            sim.pbc_pairs_reference[direction] = std::make_pair(std::make_pair(end0, end1), 
                    std::make_pair(rod0->rod_id, rod1->rod_id));
        sim.pbc_pairs.push_back(std::make_pair(direction, std::make_pair(end0, end1)));

        Offset a, b;
        if (reverse)
        {
            rod0->getEntryByLocation(rod1->indices.size() - 2, a);
            rod1->getEntryByLocation(1, b); 
            sim.pbc_bending_pairs.push_back({end0, a, b, end1});
            sim.pbc_bending_pairs_rod_id.push_back({rod0->rod_id, rod0->rod_id, rod1->rod_id, rod1->rod_id});
        }
        else
        {
            rod0->getEntryByLocation(1, a); 
            rod1->getEntryByLocation(rod1->indices.size() - 2, b);
            sim.pbc_bending_pairs.push_back({end0, a, b, end1});
            sim.pbc_bending_pairs_rod_id.push_back({rod0->rod_id, rod0->rod_id, rod1->rod_id, rod1->rod_id});
        }
        
    };

    // now we set the periodic data
    setPBCData(3, 0, 0, true, false);
    setPBCData(2, 1, 0, false, true);

    setPBCData(0, 1, 1, true, false);
    setPBCData(3, 2, 1, false, true);

    for (auto crossing : sim.rod_crossings)
    {
        if (crossing->node_idx < 8)
            continue;
        // crossing->is_fixed = false;
        // crossing->sliding_ranges[0] = Range(0.9, 0.9);
    }
    
    // auto crossing = sim.rod_crossings[8];
    // crossing->is_fixed = false;
    // crossing->sliding_ranges[0] = Range::Ones();
    // crossing = sim.rod_crossings[11];
    // crossing->is_fixed = false;
    // crossing->sliding_ranges[0] = Range::Ones();


    int dof_cnt = 0;
    markCrossingDoF(w_entry, dof_cnt);
    
    for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
    
    appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
    
    sim.rest_states = deformed_states;
    
    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
    
    for (auto& rod : sim.Rods)
    {
        rod->fixEndPointEulerian(sim.dirichlet_dof);
        rod->setupBishopFrame();
    }

    Offset offset;
    sim.Rods[0]->frontOffsetReduced(offset);
    for (int d = 0; d < 3; d++) sim.dirichlet_dof[offset[d]] = 0;

    sim.fixCrossing();

    // sim.boundary_spheres.push_back(std::make_pair(center, r * 0.5));

    sim.perturb = VectorXT::Zero(sim.W.cols());

    for (auto& crossing : sim.rod_crossings)
    {
        Offset off;
        sim.Rods[crossing->rods_involved.front()]->getEntry(crossing->node_idx, off);
        T r = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
        int z_off = sim.Rods[crossing->rods_involved.front()]->reduced_map[off[3-1]];
        sim.perturb[z_off] += 0.001 * (r - 0.5) * sim.unit;
        
    }
}

void Scene::buildInterlockingSquareScene(int sub_div)
{
    sim.yarn_map.clear();
        
    clearSimData();

    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.add_pbc_twisting = false;
    sim.add_pbc = false;

    sim.add_contact_penalty=true;
    sim.new_frame_work = true;
    sim.add_eularian_reg = true;

    sim.ke = 1e-6;

    sim.unit = 0.09;

    auto addCrossingData = [&](int crossing_idx, int rod_idx, int location)
    {
        sim.rod_crossings[crossing_idx]->is_fixed = true;
        sim.rod_crossings[crossing_idx]->rods_involved.push_back(rod_idx);
        sim.rod_crossings[crossing_idx]->on_rod_idx[rod_idx] = location;
        sim.rod_crossings[crossing_idx]->sliding_ranges.push_back(Range::Zero());
    };

    std::vector<Eigen::Triplet<T>> w_entry;
    int full_dof_cnt = 0;
    int node_cnt = 0;
    int rod_cnt = 0;

    std::vector<TV> nodal_positions;

    auto addSquare = [&](const TV& bottom_left, T width)
    {
        
        sim.rod_crossings.push_back(new RodCrossing(node_cnt, std::vector<int>()));
        addPoint(bottom_left, full_dof_cnt, node_cnt);
        nodal_positions.push_back(bottom_left);

        sim.rod_crossings.push_back(new RodCrossing(node_cnt, std::vector<int>()));
        TV bottom_right = bottom_left + TV(width, 0, 0);
        addPoint(bottom_right, full_dof_cnt, node_cnt);
        nodal_positions.push_back(bottom_right);

        sim.rod_crossings.push_back(new RodCrossing(node_cnt, std::vector<int>()));
        TV top_right = bottom_left + TV(width, width, 0);
        addPoint(top_right, full_dof_cnt, node_cnt);
        nodal_positions.push_back(top_right);

        sim.rod_crossings.push_back(new RodCrossing(node_cnt, std::vector<int>()));
        TV top_left = bottom_left + TV(0, width, 0);
        addPoint(top_left, full_dof_cnt, node_cnt);
        nodal_positions.push_back(top_left);
    };

    TV s0 = TV(0, 0, 0) * sim.unit;
    addSquare(s0, 1.0 * sim.unit);

    TV s1 = TV(0.75, 0.75, 0) * sim.unit;
    addSquare(s1, 1.0 * sim.unit);

    TV crossing0 = TV(1.0, 0.75, 0.0) * sim.unit;
    sim.rod_crossings.push_back(new RodCrossing(node_cnt, std::vector<int>()));
    addPoint(crossing0, full_dof_cnt, node_cnt);
    nodal_positions.push_back(crossing0);

    sim.rod_crossings.push_back(new RodCrossing(node_cnt, std::vector<int>()));
    TV crossing1 = TV(0.75, 1.0, 0.0) * sim.unit;
    addPoint(crossing1, full_dof_cnt, node_cnt);
    nodal_positions.push_back(crossing1);

    
    std::vector<TV> passing_points = {nodal_positions[0], nodal_positions[1]};
    std::vector<int> passing_points_id = {0, 1};

    addAStraightRod(passing_points.front(), passing_points.back(), 
        passing_points, passing_points_id, sub_div, full_dof_cnt, node_cnt, rod_cnt);
        
    for (int i = 0; i < passing_points_id.size(); i++) 
        addCrossingData(passing_points_id[i], rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[i]);

    
    passing_points = {nodal_positions[1], nodal_positions[8], nodal_positions[2]};
    passing_points_id = {1, 8, 2};
    addAStraightRod(passing_points.front(), passing_points.back(), passing_points, passing_points_id, sub_div, full_dof_cnt, node_cnt, rod_cnt);
    for (int i = 0; i < passing_points_id.size(); i++) 
        addCrossingData(passing_points_id[i], rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[i]);

    passing_points = {nodal_positions[2], nodal_positions[9], nodal_positions[3]};
    passing_points_id = {2, 9, 3};
    addAStraightRod(passing_points.front(), passing_points.back(), passing_points, passing_points_id, sub_div, full_dof_cnt, node_cnt, rod_cnt);
    for (int i = 0; i < passing_points_id.size(); i++) 
        addCrossingData(passing_points_id[i], rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[i]);

    passing_points = {nodal_positions[3], nodal_positions[0]};
    passing_points_id = {3, 0};
    addAStraightRod(passing_points.front(), passing_points.back(), passing_points, passing_points_id, sub_div, full_dof_cnt, node_cnt, rod_cnt);
    for (int i = 0; i < passing_points_id.size(); i++) 
        addCrossingData(passing_points_id[i], rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[i]);

    passing_points = {nodal_positions[4], nodal_positions[8], nodal_positions[5]};
    passing_points_id = {4, 8, 5};
    addAStraightRod(passing_points.front(), passing_points.back(), passing_points, passing_points_id, sub_div, full_dof_cnt, node_cnt, rod_cnt);
    for (int i = 0; i < passing_points_id.size(); i++) 
        addCrossingData(passing_points_id[i], rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[i]);

    passing_points = {nodal_positions[5], nodal_positions[6]};
    passing_points_id = {5, 6};
    addAStraightRod(passing_points.front(), passing_points.back(), passing_points, passing_points_id, sub_div, full_dof_cnt, node_cnt, rod_cnt);
    for (int i = 0; i < passing_points_id.size(); i++) 
        addCrossingData(passing_points_id[i], rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[i]);

    passing_points = {nodal_positions[6], nodal_positions[7]};
    passing_points_id = {6, 7};
    addAStraightRod(passing_points.front(), passing_points.back(), passing_points, passing_points_id, sub_div, full_dof_cnt, node_cnt, rod_cnt);
    for (int i = 0; i < passing_points_id.size(); i++) 
        addCrossingData(passing_points_id[i], rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[i]);

    passing_points = {nodal_positions[7], nodal_positions[9], nodal_positions[4]};
    passing_points_id = {7, 9, 4};
    addAStraightRod(passing_points.front(), passing_points.back(), passing_points, passing_points_id, sub_div, full_dof_cnt, node_cnt, rod_cnt);
    for (int i = 0; i < passing_points_id.size(); i++) 
        addCrossingData(passing_points_id[i], rod_cnt - 1, sim.Rods[rod_cnt - 1]->dof_node_location[i]);

    // for (auto& crossing : sim.rod_crossings)
    // {
    //     std::sort( crossing->rods_involved.begin(), crossing->rods_involved.end() );
    //     crossing->rods_involved.erase( std::unique( crossing->rods_involved.begin(), crossing->rods_involved.end() ), crossing->rods_involved.end() );
    // }
    
    for (auto& rod : sim.Rods)
        rod->fixed_by_crossing = std::vector<bool>(rod->dof_node_location.size(), true);

    int dof_cnt = 0;
    markCrossingDoF(w_entry, dof_cnt);

    for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
    
    appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
    
    sim.rest_states = deformed_states;
    
    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
    
    for (auto& rod : sim.Rods)
    {
        rod->fixEndPointEulerian(sim.dirichlet_dof);
        rod->setupBishopFrame();
    }
    
    TV center0;
    T r = 0.05 * sim.unit;
    sim.Rods[0]->x(sim.Rods[0]->indices.front(), center0);
    auto circle0 = [r, center0](const TV& x)->bool
    {
        return (x - center0).norm() < r;
    };

    sim.fixRegion(circle0);

    Offset offset;
    sim.Rods[5]->frontOffsetReduced(offset);
    sim.dirichlet_dof[offset[0]] = 0.3 * sim.unit;
    sim.dirichlet_dof[offset[1]] = 0;
    sim.dirichlet_dof[offset[2]] = 0;

    sim.Rods[5]->backOffsetReduced(offset);
    sim.dirichlet_dof[offset[0]] = 0.3 * sim.unit;
    sim.dirichlet_dof[offset[1]] = 0;
    sim.dirichlet_dof[offset[2]] = 0;

    auto crossing = sim.rod_crossings[8];
    crossing->is_fixed = false;
    crossing->sliding_ranges[1] = Range::Ones();

    crossing = sim.rod_crossings[9];
    crossing->is_fixed = false;
    crossing->sliding_ranges[0] = Range::Ones();

    sim.fixCrossing();

    sim.perturb = VectorXT::Zero(sim.W.cols());
    for (auto& crossing : sim.rod_crossings)
    // for (int i = 0; i < 10; i++)
    {
        // auto crossing = rod_crossings[i];
        Offset off;
        sim.Rods[crossing->rods_involved.front()]->getEntry(crossing->node_idx, off);
        T r = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
        int z_off = sim.Rods[crossing->rods_involved.front()]->reduced_map[off[3-1]];
        sim.perturb[z_off] += 0.001 * (r - 0.5) * sim.unit;
        // break;
        // sim.perturb[z_off] += 0.001 * r * sim.unit;
        // sim.perturb[z_off] += 0.001 * sim.unit;
        
    }
    sim.boundary_spheres.push_back(std::make_pair(center0, r));   

    sim.dq = VectorXT::Zero(dof_cnt);
}

void Scene::buildStraightRodScene(int sub_div)
{
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.new_frame_work = true;

    clearSimData();

    std::vector<Eigen::Triplet<T>> w_entry;
    int full_dof_cnt = 0;
    int node_cnt = 0;
    int rod_cnt = 0;

    sim.unit = 0.09;
    TV from = TV(0, 0.5, 0) * sim.unit;
    TV to = TV(1, 0.5, 0.0) * sim.unit;

    std::vector<int> passing_points_id;
    std::vector<TV> passing_points;

    addAStraightRod(from, to, passing_points, passing_points_id, 
            sub_div, full_dof_cnt, node_cnt, rod_cnt);
    
    int dof_cnt;
    for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
    
    appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
    
    sim.rest_states = deformed_states;

    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());


    int cnt = 0;
    for (auto& rod : sim.Rods)
    {
        // std::cout << rod->kt << std::endl;
        // rod->kt  =0 ;
        rod->fixEndPointEulerian(sim.dirichlet_dof);
        sim.dirichlet_dof[rod->theta_reduced_dof_start_offset] = 0;
        // sim.dirichlet_dof[rod->theta_reduced_dof_start_offset + rod->indices.size()-1] = 0;
        rod->setupBishopFrame();
        Offset end0, end1;
        rod->frontOffset(end0); rod->backOffset(end1);
        sim.pbc_pairs.push_back(std::make_pair(0, std::make_pair(end0, end1)));
    }

    Offset end0, end1;
    sim.Rods[0]->frontOffset(end0); sim.Rods[0]->backOffset(end1);
    sim.pbc_pairs_reference[0] = std::make_pair(std::make_pair(end0, end1), std::make_pair(0, 0));
    
    TV delta1 = TV(-0.6, 0., 0) * sim.unit;
    // TV delta1 = TV(1-1.6*std::sqrt(2)/2, -1.6*std::sqrt(2)/2, 0) * sim.unit;
    sim.Rods[0]->fixPointLagrangian(0, delta1, sim.dirichlet_dof);
    // std::cout << "Rod index: \n";
    // for (auto idx: sim.Rods[0]->indices){
    //     std::cout << idx << std::endl;
    // }
    sim.Rods[0]->fixPointLagrangian(sim.Rods[0]->indices.size()-1, TV::Zero(), sim.dirichlet_dof);

    sim.dq = VectorXT::Zero(dof_cnt);
}

struct MeshEdge{
    int u_, v_;

    bool operator==(const MeshEdge& other) const {
        return (u_ == other.u_ && v_ == other.v_) || (u_ == other.v_ && v_ == other.u_);
    }
    MeshEdge(int u, int v): u_(u), v_(v){}
};

struct MeshEdgeHash {
    size_t operator()(const MeshEdge& e) const {
        return std::hash<int>()(e.u_) ^ std::hash<int>()(e.v_);
    }
};

void Scene::buildFEMRodScene(const std::string& filename, int sub_div, bool bc_data){
    
    Eigen::MatrixXd V; Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);
    mesh_file = filename;
    sim.n_nodes = V.rows();
    TV min_corner = V.colwise().minCoeff();
    TV max_corner = V.colwise().maxCoeff();
    T length = max_corner(0)-min_corner(0);
    V /= length;
    std::unordered_set<MeshEdge, MeshEdgeHash> edges;
    for(int i = 0; i < F.rows(); ++i){
        Eigen::Vector<int, 3> face = F.row(i);
        edges.insert(MeshEdge(face(0), face(1)));
        edges.insert(MeshEdge(face(0), face(2)));
        edges.insert(MeshEdge(face(2), face(1)));
    }

    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    
    clearSimData();

    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.add_pbc_twisting = false;
    sim.add_pbc = false;

    sim.add_contact_penalty=false;
    sim.new_frame_work = false;
    sim.add_eularian_reg = false;
    sim.add_bending = false;
    sim.add_twisting = false;

    sim.ke = 1e-6;

    sim.unit = 1.0;

    std::vector<Eigen::Triplet<T>> w_entry;
    int full_dof_cnt = 0;
    int node_cnt = 0;
    int rod_cnt = 0;

    std::vector<TV> nodal_positions;

    auto addCrossingData = [&](int crossing_idx, int rod_idx, int location)
    {
        sim.rod_crossings[crossing_idx]->is_fixed = true;
        sim.rod_crossings[crossing_idx]->rods_involved.push_back(rod_idx);
        sim.rod_crossings[crossing_idx]->on_rod_idx[rod_idx] = location;
        sim.rod_crossings[crossing_idx]->sliding_ranges.push_back(Range::Zero());
    };

    for(int i = 0; i < V.rows(); ++i){
        TV node_pos = V.row(i); 
        addCrossingPoint(nodal_positions, node_pos, full_dof_cnt, node_cnt);
    }

    for(auto edge: edges){
        TV node1 = V.row(edge.u_);
        TV node2 = V.row(edge.v_);
        if(node1(0) > node2(0)) {
            std::swap(node1, node2);
            std::swap(edge.u_, edge.v_);
        }

        addAStraightRod(node1, node2, {node1, node2}, {edge.u_, edge.v_}, sub_div,
        full_dof_cnt, node_cnt, rod_cnt);

        addCrossingData(edge.u_, rod_cnt - 1, 0);
        addCrossingData(edge.v_, rod_cnt - 1, sim.Rods[rod_cnt - 1]->numSeg());
    }

    for (auto& rod : sim.Rods)
        rod->fixed_by_crossing = std::vector<bool>(rod->dof_node_location.size(), true);

    int dof_cnt = 0;
    markCrossingDoF(w_entry, dof_cnt);
    // std::cout << "mark dof" << std::endl;
    for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
    
    appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
    
    sim.rest_states = deformed_states;
    
    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
    
    for (auto& rod : sim.Rods)
    {
        rod->fixEndPointEulerian(sim.dirichlet_dof);
        rod->setupBishopFrame();
    }
    
    if (sim.add_pbc)
        sim.Rods[0]->fixPointLagrangianByID(0, TV::Zero(), Mask::Ones(), sim.dirichlet_dof);

    TV bottom_left, top_right;
    sim.computeBoundingBox(bottom_left, top_right);

    // TV shear_x_right = TV(0.1, 0.0, 0.0) * sim.unit;
    // TV shear_x_left = TV(-0.1, 0.0, 0) * sim.unit;
    TV shear_x_right = TV(0.0, 0.01, 0.0) * sim.unit;
    TV shear_x_left = TV(0.0, 0.0, 0) * sim.unit;


    T rec_width = 0.0001 * sim.unit;

    auto rec1 = [bottom_left, top_right, shear_x_left, rec_width](
        const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {
        mask = Vector<bool, 3>(true, true, true);
        // mask = Vector<bool, 3>(true, true, true);
        // double tol = 1e-5;
        // if(x[0] < bottom_left[0]+tol && x[1]>top_right[1]-tol) mask = Vector<bool, 3>(true, true, true);
        // else if(x[0] > top_right[0]-tol && x[1]>top_right[1]-tol) mask = Vector<bool, 3>(true, true, true);
        // else mask = Vector<bool, 3>(true, false, true);
        delta = shear_x_left;
        if (x[0] < bottom_left[0] + rec_width)
            return true;
        return false;
    };

    auto rec2 = [bottom_left, top_right, shear_x_right, rec_width](
        const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {   
        mask = Vector<bool, 3>(true, true, true);
        // mask = Vector<bool, 3>(true, true, true);
        // double tol = 1e-5;
        // if(x[0] < bottom_left[0]+tol && x[1]>top_right[1]-tol) mask = Vector<bool, 3>(true, true, true);
        // else if(x[0] > top_right[0]-tol && x[1]>top_right[1]-tol) mask = Vector<bool, 3>(true, true, true);
        // else mask = Vector<bool, 3>(true, false, true);
        delta = shear_x_right;

        if (x[0] > top_right[0] - rec_width)
            return true;
        return false;
    };

    if (!bc_data)
    {
        sim.fixRegionalDisplacement(rec2);
        sim.fixRegionalDisplacement(rec1);
    }
    TV shear_x_down = TV(0.0, -0.1, 0.0) * sim.unit;
    TV shear_x_up = TV(0.0, 0.1, 0) * sim.unit;
    auto rec3 = [bottom_left, top_right, shear_x_down, rec_width](
        const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {
        // mask = Vector<bool, 3>(true, false, true);
        mask = Vector<bool, 3>(true, true, true);
        delta = shear_x_down;
        if (x[1] < bottom_left[1] + rec_width)
            return true;
        return false;
    };

    auto rec4 = [bottom_left, top_right, shear_x_up, rec_width](
        const TV& x, TV& delta, Vector<bool, 3>& mask)->bool
    {   
        // mask = Vector<bool, 3>(true, false, true);
        mask = Vector<bool, 3>(true, true, true);
        delta = shear_x_up;

        if (x[1] > top_right[1] - rec_width)
            return true;
        return false;
    };
    // sim.fixRegionalDisplacement(rec3);
    // sim.fixRegionalDisplacement(rec4);

    sim.fixCrossing();
    // std::cout << "fix crossing" << std::endl;
    sim.perturb = VectorXT::Zero(sim.W.cols());
    for (auto& crossing : sim.rod_crossings)
    {
        Offset off;
        sim.Rods[crossing->rods_involved.front()]->getEntry(crossing->node_idx, off);
        T r = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
        int z_off = sim.Rods[crossing->rods_involved.front()]->reduced_map[off[3-1]];
        sim.perturb[z_off] += 0.001 * (r - 0.5) * sim.unit;
        
    }
    sim.dq = VectorXT::Zero(dof_cnt);

}

void Scene::clearSimData()
{
    sim.kc = 1e8;
    sim.add_pbc = true;

    if(sim.disable_sliding)
    {
        sim.add_shearing = true;
        sim.add_eularian_reg = false;
        sim.k_pbc = 1e8;
        sim.k_strain = 1e8;
    }
    else
    {
        sim.add_shearing = false;
        sim.add_eularian_reg = true;
        sim.ke = 1e-4;    
        sim.k_yc = 1e8;
    }
    sim.k_pbc = 1e4;
    sim.k_strain = 1e7;
    sim.kr = 1e3;
    sim.yarns.clear();
}

void Scene::markCrossingDoF(std::vector<Eigen::Triplet<T>>& w_entry,
        int& dof_cnt)
{
    for (auto& crossing : sim.rod_crossings)
    {
        int node_idx = crossing->node_idx;
        // std::cout << "node " << node_idx << std::endl;
        std::vector<int> rods_involved = crossing->rods_involved;

        Offset entry_rod0; 
        sim.Rods[rods_involved.front()]->getEntry(node_idx, entry_rod0);

        // push Lagrangian dof first
        for (int d = 0; d < 3; d++)
        {
            
            for (int rod_idx : rods_involved)
            {
                // std::cout << "rods involved " << rod_idx << std::endl;
                // if (node_idx == 21)
                //     std::cout << "rods involved " << rod_idx << std::endl;
                sim.Rods[rod_idx]->reduced_map[entry_rod0[d]] = dof_cnt;
            }    
            w_entry.push_back(Entry(entry_rod0[d], dof_cnt++, 1.0));
        }
        
        // push Eulerian dof for all rods
        for (int rod_idx : rods_involved)
        {
            // std::cout << "3 on rod " <<  rod_idx << std::endl;
            sim.Rods[rod_idx]->getEntry(node_idx, entry_rod0);
            // std::cout << "3 dof on rod " <<  entry_rod0[3] << std::endl;
            sim.Rods[rod_idx]->reduced_map[entry_rod0[3]] = dof_cnt;
            w_entry.push_back(Entry(entry_rod0[3], dof_cnt++, 1.0));
        }
        // std::getchar();
        
    }
}

void Scene::appendThetaAndJointDoF(std::vector<Entry>& w_entry, 
    int& full_dof_cnt, int& dof_cnt)
{
    // for (auto& rod : sim.Rods)
    // {
    //     rod->theta_dof_start_offset = full_dof_cnt;
    //     rod->theta_reduced_dof_start_offset = dof_cnt;
    //     deformed_states.conservativeResize(full_dof_cnt + rod->indices.size() - 1);
    //     for (int i = 0; i < rod->indices.size() - 1; i++)
    //         w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
    //     deformed_states.template segment(rod->theta_dof_start_offset, 
    //         rod->indices.size() - 1).setZero();
    // }   

    deformed_states.conservativeResize(full_dof_cnt + sim.rod_crossings.size() * 3);
    deformed_states.template segment(full_dof_cnt, sim.rod_crossings.size() * 3).setZero();

    for (auto& crossing : sim.rod_crossings)
    {
        crossing->dof_offset = full_dof_cnt;
        crossing->reduced_dof_offset = dof_cnt;
        for (int d = 0; d < 3; d++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }
    }

    for (auto& rod : sim.Rods)
    {
        rod->theta_dof_start_offset = full_dof_cnt;
        rod->theta_reduced_dof_start_offset = dof_cnt;
        deformed_states.conservativeResize(full_dof_cnt + rod->indices.size() - 1);
        for (int i = 0; i < rod->indices.size() - 1; i++)
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        deformed_states.template segment(rod->theta_dof_start_offset, 
            rod->indices.size() - 1).setZero();
    }
}

void Scene::addCurvedRod(const std::vector<TV2>& data_points,
    const std::vector<TV>& passing_points, 
    const std::vector<int>& passing_points_id, 
    int sub_div, int& full_dof_cnt, int& node_cnt, int& rod_cnt, bool closed)
{
    if (passing_points.size() != passing_points_id.size())
        std::cout << " passing_points.size() != passing_points_id.size() " << std::endl;

    int first_node_idx = node_cnt;
    int sub_div_2 = sub_div / 2;
    HybridC2Curve<2>* curve = new HybridC2Curve<2>(sub_div);
    for (const auto& pt : data_points)
        curve->data_points.push_back(pt);

    std::vector<TV2> points_on_curve;
    curve->sampleCurves(points_on_curve);
    // for (auto data_pt : data_points)
    //     std::cout << data_pt.transpose() << " | ";
    // std::cout << std::endl;
    // for (auto data_pt : points_on_curve)
    //     std::cout << data_pt.transpose() << " | ";
    // std::cout << std::endl;
    // // std::cout << points_on_curve.size() << std::endl;
    // std::getchar();

    std::unordered_map<int, int> dof_node_location;
    if (closed)
        deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size() - 1 - passing_points_id.size()) * (3 + 1));
    else
        deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size() - passing_points_id.size()) * (3 + 1));



    Rod* rod = new Rod(deformed_states, sim.rest_states, rod_cnt, closed, ROD_A, ROD_B);
    std::unordered_map<int, Offset> offset_map;
    std::vector<int> node_index_list;
    std::vector<T> data_points_discrete_arc_length;
    int full_dof_before = full_dof_cnt;
    int not_found_cnt = 0;
    for (int i = 0; i < points_on_curve.size(); i++)
    {
        
        TV2 pt = points_on_curve[i];
        TV pt_search;
        pt_search.template segment<2>(0) = pt;
        // std::cout << "pt on curve " << pt.transpose() << std::endl;
        //if points already added as crossings
        auto find_node_iter = std::find_if(passing_points.begin(), passing_points.end(), 
            [&pt_search](TV pt_in_vec)
            { return (pt_search - pt_in_vec).norm() < 1e-8; }
        );
        
        if (find_node_iter == passing_points.end())
        {
            if (closed && i == points_on_curve.size() - 1)
            {
                node_index_list.push_back(first_node_idx);
                break;
            }
            offset_map[node_cnt] = Offset::Zero();
            node_index_list.push_back(node_cnt);
            //push Lagrangian DoF
            for (int d = 0; d < 3; d++) 
            {
                deformed_states[full_dof_cnt] = pt_search[d];
                offset_map[node_cnt][d] = full_dof_cnt++;  
            }
            // push Eulerian DoF
            deformed_states[full_dof_cnt] = T(i) * (curve->data_points.size() - 1) / (points_on_curve.size() - 1);
            offset_map[node_cnt][3] = full_dof_cnt++;
            node_cnt++;
        }
        else
        {
            not_found_cnt++;
            int dof_loc = std::distance(passing_points.begin(), find_node_iter);
            
            // std::cout << passing_points[dof_loc].transpose() << std::endl;

            node_index_list.push_back(passing_points_id[dof_loc]);
            dof_node_location[dof_loc] = i;
            if (closed && i == points_on_curve.size() - 1)
                dof_node_location[dof_loc] = 0;
        }
        
    }
    // checking scene
    // if (not_found_cnt != passing_points.size() + int(closed))
        // std::cout << "not_found_cnt " << not_found_cnt << " should be: passing_points.size() " << passing_points.size() + int(closed) << std::endl;
    int dof_added = full_dof_cnt - full_dof_before;

    int dof_added_should_be = (points_on_curve.size() - passing_points.size() - int(closed)) * (3 + 1);
    if (dof_added != dof_added_should_be)
        std::cout << "after Lagrangian dof_added " << dof_added << " dof_added should be " << dof_added_should_be << std::endl;

    // now we add the Eulerian Dof of the passing points
    deformed_states.conservativeResize(full_dof_cnt + passing_points.size());

    for (int i = 0; i < passing_points.size(); i++)
    {
        offset_map[passing_points_id[i]] = Offset::Zero();
        for (int d = 0; d < 3; d++)
            offset_map[passing_points_id[i]][d] = passing_points_id[i] * 3 + d;
        
        deformed_states[full_dof_cnt] = T(dof_node_location[i]) * (curve->data_points.size() - 1) / (points_on_curve.size() - 1);
        
        offset_map[passing_points_id[i]][3] = full_dof_cnt++; 
    }

    dof_added = full_dof_cnt - full_dof_before;
    dof_added_should_be += passing_points.size();
    if (dof_added != dof_added_should_be)
        std::cout << "after eulerian:  dof_added " << dof_added << " dof_added should be " << dof_added_should_be << std::endl;

    rod->offset_map = offset_map;
    rod->indices = node_index_list;

    for(int i = 0; i < curve->data_points.size(); i++)
    {
        int node_idx = rod->indices[i * sub_div_2];        
        data_points_discrete_arc_length.push_back(deformed_states[offset_map[node_idx][3]]);
    }

    Vector<T, 3 + 1> q0, q1;
    rod->frontDoF(q0); rod->backDoF(q1);

    DiscreteHybridCurvature* rest_state_rod0 = new DiscreteHybridCurvature(q0, q1);
    sim.curvature_functions.push_back(rest_state_rod0);
    rest_state_rod0->setData(curve, data_points_discrete_arc_length);


    rod->rest_state = rest_state_rod0;
    std::vector<int> ordered_location;
    for (auto item : dof_node_location)
        ordered_location.push_back(item.second);
    std::sort(ordered_location.begin(), ordered_location.end());


    rod->dof_node_location = ordered_location;

    sim.Rods.push_back(rod);

    rod_cnt++;

}

void Scene::addAStraightRod(const TV& from, const TV& to, 
        const std::vector<TV>& passing_points, 
        const std::vector<int>& passing_points_id, 
        int sub_div, int& full_dof_cnt, int& node_cnt, int& rod_cnt)
{
    
    std::unordered_map<int, Offset> offset_map;

    std::vector<TV> points_on_curve;
    std::vector<int> rod_indices;
    std::vector<int> key_points_location_rod;
    addStraightYarnCrossNPoints(from, to, passing_points, passing_points_id,
                                sub_div, points_on_curve, rod_indices,
                                key_points_location_rod, node_cnt);
                   

    deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (3 + 1));

    Rod* rod = new Rod(deformed_states, sim.rest_states, rod_cnt, false, ROD_A, ROD_B);

    for (int i = 0; i < points_on_curve.size(); i++)
    {
        offset_map[node_cnt] = Offset::Zero();
        //push Lagrangian DoF    
        deformed_states.template segment<3>(full_dof_cnt) = points_on_curve[i];
        for (int d = 0; d < 3; d++)
        {
            offset_map[node_cnt][d] = full_dof_cnt++;  
        }
        // push Eulerian DoF
        deformed_states[full_dof_cnt] = (points_on_curve[i] - from).norm() / (to - from).norm();
        offset_map[node_cnt][3] = full_dof_cnt++;
        node_cnt++;
    }
    
    deformed_states.conservativeResize(full_dof_cnt + passing_points.size());

    for (int i = 0; i < passing_points.size(); i++)
    {
        deformed_states[full_dof_cnt] = (passing_points[i] - from).norm() / (to - from).norm();
        offset_map[passing_points_id[i]] = Offset::Zero();
        offset_map[passing_points_id[i]][3] = full_dof_cnt++; 
        Vector<int, 3> offset_dof_lag;
        for (int d = 0; d < 3; d++)
        {
            offset_dof_lag[d] = passing_points_id[i] * 3 + d;
        }
        offset_map[passing_points_id[i]].template segment<3>(0) = offset_dof_lag;
    }
    
    rod->offset_map = offset_map;
    rod->indices = rod_indices;
    Vector<T, 3 + 1> q0, q1;
    rod->frontDoF(q0); rod->backDoF(q1);

    rod->rest_state = new LineCurvature(q0, q1);
    
    rod->dof_node_location = key_points_location_rod;
    
    sim.Rods.push_back(rod);
    rod_cnt++;
}

void Scene::addStraightYarnCrossNPoints(const TV& from, const TV& to,
    const std::vector<TV>& passing_points, 
    const std::vector<int>& passing_points_id, int sub_div,
    std::vector<TV>& sub_points, std::vector<int>& node_idx, 
    std::vector<int>& key_points_location, 
    int start, bool pbc)
{
    
    int cnt = 1;
    if(passing_points.size())
    {
        if ((from - passing_points[0]).norm() < 1e-6 )
        {
            node_idx.push_back(passing_points_id[0]);
            cnt = 0;
        }
        else
        {
            node_idx.push_back(start);
            sub_points.push_back(from);
        }
    }
    else
    {
        node_idx.push_back(start);
        sub_points.push_back(from);
    }
    
    T length_yarn = (to - from).norm();
    TV length_vec = (to - from).normalized();
    
    TV loop_point = from;
    TV loop_left = from;
    for (int i = 0; i < passing_points.size(); i++)
    {
        if ((from - passing_points[i]).norm() < 1e-6 )
        {
            key_points_location.push_back(0);
            continue;
        }
        T fraction = (passing_points[i] - loop_point).norm() / length_yarn;
        int n_sub_nodes = std::ceil(fraction * sub_div);
        T length_sub = (passing_points[i] - loop_point).norm() / T(n_sub_nodes);
        for (int j = 0; j < n_sub_nodes - 1; j++)
        {
            sub_points.push_back(loop_left + length_sub * length_vec);
            loop_left = sub_points.back();
            node_idx.push_back(start + cnt);
            cnt++;
        }
        node_idx.push_back(passing_points_id[i]);
        key_points_location.push_back(cnt + i);
        loop_point = passing_points[i];
        loop_left = passing_points[i];
    }
    if (passing_points.size())
    {
        if ((passing_points.back() - to).norm() < 1e-6)
        {
            
            return;
        }
    }
    T fraction;
    int n_sub_nodes;
    T length_sub;
    if( passing_points.size() )
    {
        fraction = (to - passing_points.back()).norm() / length_yarn;
        n_sub_nodes = std::ceil(fraction * sub_div);
        length_sub = (to - passing_points.back()).norm() / T(n_sub_nodes);
    }
    else
    {
        n_sub_nodes = sub_div + 1;
        length_sub = (to - from).norm() / T(sub_div);
    }
    for (int j = 0; j < n_sub_nodes - 1; j++)
    {
        if (j == 0)
        {
            if(passing_points.size())
            {
                sub_points.push_back(passing_points.back() + length_sub * length_vec);
                loop_left = sub_points.back();
            }
        }
        else
        {
            sub_points.push_back(loop_left + length_sub * length_vec);
            loop_left = sub_points.back();
        }
        if(passing_points.size() == 0 && j == 0)
            continue;
        node_idx.push_back(start + cnt);
        cnt++;
    }
    node_idx.push_back(start + cnt);
    sub_points.push_back(to);
}

void Scene::addPoint(const TV& point, int& full_dof_cnt, int& node_cnt)
{
    deformed_states.conservativeResize(full_dof_cnt + 3);
    deformed_states.template segment<3>(full_dof_cnt) = point;
    full_dof_cnt += 3;
    node_cnt++;
}

void Scene::addCrossingPoint(std::vector<TV>& existing_nodes, 
        const TV& point, int& full_dof_cnt, int& node_cnt)
{
    sim.rod_crossings.push_back(new RodCrossing(node_cnt, std::vector<int>())); 
    deformed_states.conservativeResize(full_dof_cnt + 3);
    deformed_states.template segment<3>(full_dof_cnt) = point;
    existing_nodes.push_back(point);
    full_dof_cnt += 3;
    node_cnt++;
}