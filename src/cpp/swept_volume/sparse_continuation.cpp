#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
#include <unordered_map>
#include <array>
#include <vector>
#include <queue>

void sparse_continuation(const Eigen::RowVector3d p0, const std::vector<Eigen::RowVector3i> init_voxels, const std::vector<double> t0, const std::function<double(const Eigen::RowVector3d &, double &, std::vector<std::vector<double>> &, std::vector<std::vector<double>> &, std::vector<std::vector<double>> &)> scalarFunc, const double eps, const int expected_number_of_cubes, Eigen::VectorXd & CS, Eigen::MatrixXd & CV, Eigen::MatrixXi & CI, Eigen::VectorXd & CV_argmins_vector){
    
    struct IndexRowVectorHash  {
        std::size_t operator()(const Eigen::RowVector3i& key) const {
            std::size_t seed = 0;
            std::hash<int> hasher;
            for (int i = 0; i < 3; i++) {
                seed ^= hasher(key[i]) + 0x9e3779b9 + (seed<<6) + (seed>>2); // Copied from boost::hash_combine
            }
            return seed;
        }
    };
    
    auto sgn = [](double val) -> int {
        return (double(0) < val) - (val < double(0));
    };
    
    double half_eps = 0.5 * eps;
    
    std::vector<Eigen::Matrix<int,1,8>> CI_vector;
    std::vector<Eigen::RowVector3d> CV_vector;
    std::vector<std::vector<std::vector<double>>> CV_intervals;
    std::vector<std::vector<std::vector<double>>> CV_values;
    std::vector<std::vector<std::vector<double>>> CV_minima;
    std::vector<double> CS_vector;
    CI_vector.reserve(expected_number_of_cubes);
    CV_vector.reserve(8 * expected_number_of_cubes);
    CS_vector.reserve(8 * expected_number_of_cubes);
    std::vector<std::vector<double>> argmins;
    std::vector<double> CV_argmins;
    
    argmins.reserve(32 * expected_number_of_cubes);
    int counter = 0;
    
    // Track visisted neighbors
    std::unordered_map<Eigen::RowVector3i, int, IndexRowVectorHash> visited;
    visited.reserve(6 * expected_number_of_cubes);
    visited.max_load_factor(0.5);
    
    // BFS Queue
    auto cmp = [](std::tuple<Eigen::RowVector3i, double, int, double> left, std::tuple<Eigen::RowVector3i, double, int, double> right) {
        double pleft, pright;
        pleft = std::get<3>(left);
        pright = std::get<3>(right);
        return pleft > pright;
    };
    
    std::priority_queue<std::tuple<Eigen::RowVector3i, double, int, double>, std::vector<std::tuple<Eigen::RowVector3i, double, int, double>>, decltype(cmp)> p_queue(cmp);
    
    std::vector<Eigen::RowVector3i> queue;
    queue.reserve(expected_number_of_cubes * 8);
    std::vector<double> time_queue;
    time_queue.reserve(expected_number_of_cubes * 8);
    std::vector<int> correspondence_queue;
    correspondence_queue.reserve(expected_number_of_cubes * 8);
    std::vector<double> intervals_turn, values_turn, minima_turn;
    
    for (int seed_ind = 0; seed_ind < init_voxels.size(); seed_ind++) {
        
        double min_turn = 1000.0;
        double final_seed = t0[seed_ind];
//        for (double tt = 0; tt <=1.0; tt = tt + 0.1) {
//            intervals_turn.resize(0);
//            values_turn.resize(0);
//            minima_turn.resize(0);
//            double seed_turn = tt;
//            Eigen::RowVector3i pi_turn = init_voxels[seed_ind];
//            Eigen::RowVector3d ctr_turn = p0 + eps*pi_turn.cast<double>();
//            double val_turn = scalarFunc(ctr_turn,seed_turn,intervals_turn,values_turn,minima_turn);
//            if (val_turn < min_turn) {
//                final_seed = seed_turn;
//                min_turn = val_turn;
//            }
//        }
        queue.push_back(init_voxels[seed_ind]);
        time_queue.push_back(final_seed);
        correspondence_queue.push_back(-1);
        auto bar = std::make_tuple(init_voxels[seed_ind], final_seed, -1, 0.0);
        p_queue.push(bar);
    }
    //queue.push_back(Eigen::RowVector3i(0, 0, 0));
    //time_queue.push_back(t0);
    //std::cout << "test" << std::endl;
    
    int additions_normal, additions_corrections, additions_self;
    additions_normal = 0;
    additions_corrections = 0;
    additions_self = 0;
    while (queue.size() > 0)
    {
        Eigen::RowVector3i pi = queue.back();
        queue.pop_back();
        double time_seed = time_queue.back();
        time_queue.pop_back();
        int correspondence = correspondence_queue.back();
        correspondence_queue.pop_back();
        
        
//        std::tuple<Eigen::RowVector3i, double, int, double> val;
//        val = p_queue.top();
//        p_queue.pop();
//        pi = std::get<0>(val);
//        time_seed = std::get<1>(val);
//        correspondence = std::get<2>(val);
        
        
        Eigen::RowVector3d ctr = p0 + eps*pi.cast<double>(); // R^3 center of this cube
        
        // X, Y, Z basis vectors, and array of neighbor offsets used to construct cubes
        const Eigen::RowVector3i bx(1, 0, 0), by(0, 1, 0), bz(0, 0, -1);
        const std::array<Eigen::RowVector3i, 30> neighbors = {
            bx, -bx, by, -by, bz, -bz,
            by-bz, -by+bz, // 1-2 4-7
            bx+by, -bx-by, // 0-1 7-6
            by+bz, -by-bz,  // 0-3 6-5
            by-bx, -by+bx,  // 2-3 5-4
            bx-bz, -bx+bz, // 1-5 3-7
            bx+bz, -bx-bz, // 0-4 2-6
            -bx+by+bz, bx-by-bz, // 3 5
            bx+by+bz, -bx-by-bz, // 0 6
            bx+by-bz, -bx-by+bz, //1 7
            -bx+by-bz, bx-by+bz, // 2 4,
            bx-bx, bx-bx,
            bx-bx, bx-bx
        };
        
        // Compute the position of the cube corners and the scalar values at those corners
        std::array<Eigen::RowVector3d, 8> cubeCorners = {
            ctr+half_eps*(bx+by+bz).cast<double>(), ctr+half_eps*(bx+by-bz).cast<double>(), ctr+half_eps*(-bx+by-bz).cast<double>(), ctr+half_eps*(-bx+by+bz).cast<double>(),
            ctr+half_eps*(bx-by+bz).cast<double>(), ctr+half_eps*(bx-by-bz).cast<double>(), ctr+half_eps*(-bx-by-bz).cast<double>(), ctr+half_eps*(-bx-by+bz).cast<double>()
        };
        std::array<double, 8> cubeScalars;
        //double time_seed = 0.0;
        //std::cout << time_seed << std::endl;
        double time_test;
        argmins[CI_vector.size()].resize(8);
        std::vector<double> argmins_cube;
        
        
        
        // Add the cube vertices and indices to the output arrays if they are not there already
        
        uint8_t vertexAlreadyAdded = 0; // This is a bimask. If a bit is 1, it has been visited already by the BFS
        constexpr std::array<uint8_t, 30> zv = {
            (1 << 0) | (1 << 1) | (1 << 4) | (1 << 5),
            (1 << 2) | (1 << 3) | (1 << 6) | (1 << 7),
            (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
            (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),
            (1 << 0) | (1 << 3) | (1 << 4) | (1 << 7),
            (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6),
            (1 << 1) | (1 << 2),
            (1 << 4) | (1 << 7),
            (1 << 0) | (1 << 1),
            (1 << 6) | (1 << 7),
            (1 << 0) | (1 << 3),
            (1 << 5) | (1 << 6),
            (1 << 2) | (1 << 3),
            (1 << 4) | (1 << 5),
            (1 << 1) | (1 << 5),
            (1 << 3) | (1 << 7),
            (1 << 0) | (1 << 4),
            (1 << 2) | (1 << 6),
            (1 << 3), (1 << 5), // diagonals
            (1 << 0), (1 << 6),
            (1 << 1), (1 << 7),
            (1 << 2), (1 << 4),
            (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
            (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
            (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),
            (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),
        };
        constexpr std::array<std::array<int, 4>, 30> zvv {{
            {{0, 1, 4, 5}}, {{3, 2, 7, 6}}, {{0, 1, 2, 3}},
            {{4, 5, 6, 7}}, {{0, 3, 4, 7}}, {{1, 2, 5, 6}},
            {{-1,-1,1,2}}, {{-1,-1,4,7}}, {{-1,-1,0,1}},{{-1,-1,7,6}},
            {{-1,-1,0,3}}, {{-1,-1,5,6}}, {{-1,-1,2,3}}, {{-1,-1,5,4}},
            {{-1,-1,1,5}}, {{-1,-1,3,7}}, {{-1,-1,0,4}}, {{-1,-1,2,6}},
            {{-1,-1,-1,3}}, {{-1,-1,-1,5}}, {{-1,-1,-1,0}}, {{-1,-1,-1,6}},
            {{-1,-1,-1,1}}, {{-1,-1,-1,7}}, {{-1,-1,-1,2}}, {{-1,-1,-1,4}},
            {{0,1,2,3}}, {{0,1,2,3}}, {{4,5,6,7}}, {{4,5,6,7}}
        }};
        bool flag = false;
        
        
        Eigen::Matrix<int,1,8> cube;
        cube << -1, -1, -1, -1, -1, -1, -1, -1;
        for (int n = 0; n < 30; n++) { // For each neighbor, check the hash table to see if its been added before
            Eigen::RowVector3i nkey = pi + neighbors[n];
            auto nbr = visited.find(nkey);
            if (nbr != visited.end()) {
                for (int i = 0; i < 4; i++) {
                    if (zvv[n][i]!=-1) {
                        cube[zvv[n][i]] = CI_vector[nbr->second][zvv[n % 2 == 0 ? n + 1 : n - 1][i]];
                    }
                }
            }
        }
        
        
        // do we already know we're inside?
        bool we_in = true;
        for (int i = 0; i<8; i++) {
            if(cube[i]==-1){
                we_in = false;
                break;
            }
            if (CS_vector[cube[i]] > 0.0) {
                we_in = false;
                break;
            }
        }
        
        if (we_in) {
            continue;
        }
        
        
        
        std::vector<std::vector<std::vector<double>>> intervals;
        intervals.resize(8);
        std::vector<std::vector<std::vector<double>>> values;
        values.resize(8);
        std::vector<std::vector<std::vector<double>>> minima;
        minima.resize(8);
        
        bool debug_flag = false;
        bool in_existing_interval = false;
        bool intersecting_interval = false;
        time_test = time_seed;
        double running_argmin = 0.0;
        for (int i = 0; i < 8; i++){
            time_test = time_seed;
            //            cubeScalars[i] = scalarFunc(cubeCorners[i],time_test);
            //            argmins[CI_vector.size()][i] = time_test;
            //   running_argmin = running_argmin + (time_test/8.0);
            
            if (cube[i] >= 0) {
                cubeScalars[i] = scalarFunc(cubeCorners[i],time_test,CV_intervals[cube[i]], CV_values[cube[i]], CV_minima[cube[i]]);
                argmins[CI_vector.size()][i] = time_test;
                
                if (correspondence==-1) {
                double temp = cubeScalars[i];
                    //std::cout << "We got " << temp << " at " << time_seed << "...";
                int temp_i = -1;
                    int temp_s = -1;
                    for (int s = 0; s < CV_intervals[cube[i]].size(); s++) {
                for (int mm = 0; mm < (CV_intervals[cube[i]][s].size()/2); mm++){
                    if ( (CV_values[cube[i]][s][mm]+1e-3) < temp) {
                        temp = CV_values[cube[i]][s][mm];
                        temp_i = mm;
                        temp_s = s;
                    }
                }}
                if (temp_i > -1) {
                     //JAN 24 change this
                    queue.push_back(pi);
                    //std::cout << " but we found " << temp << " at " << CV_minima[cube[i]][temp_s][temp_i] << " (correspondence " << correspondence << std::endl;
                    // Otherwise, we have not visited the neighbor, put it in the BFS queue
                    // time_queue.push_back(time_test);
                    time_queue.push_back(CV_minima[cube[i]][temp_s][temp_i]);
                    correspondence_queue.push_back(1);
                    auto bar = std::make_tuple(pi, CV_minima[cube[i]][temp_s][temp_i], 1, temp);
                    p_queue.push(bar);
                    additions_self++;
                    
                    //time_seed = CV_minima[cube[i]][temp_s][temp_i];
//                    cubeScalars[i] = temp;
//                    argmins[CI_vector.size()][i] = CV_minima[cube[i]][temp_s][temp_i];
                }else{
                    //std::cout << std::endl;
                }
                }
                // DEBUGGING
                //                if (cube[i]==10) {
//                std::cout << "______________";
//                std::cout << "Time seed: " << time_seed << std::endl;
//                for (int mm = 0; mm < (CV_intervals[cube[i]].size()/2); mm++){
//                    std::cout << "Existing interval: " << CV_intervals[cube[i]][2*mm] << " " <<  CV_intervals[cube[i]][2*mm + 1] << " value: " << CV_values[cube[i]][mm] << std::endl;
//                }
                //                }
                //                            in_existing_interval = false;
                //                            for (int mm = 0; mm < (CV_intervals[cube[i]].size()/2); mm++) {
                //                                if ((time_test >= CV_intervals[cube[i]][2*mm]) && (time_test <= CV_intervals[cube[i]][2*mm+1]) ){
                //                                    cubeScalars[i] = CV_values[cube[i]][mm];
                //                                    argmins[CI_vector.size()][i] = CV_minima[cube[i]][mm];
                //                                    in_existing_interval = true;
                //                                }
                //                            }
                //                            if (!in_existing_interval) {
                //                                intersecting_interval = false;
                //                                time_test = time_seed;
                //                                cubeScalars[i] = scalarFunc(cubeCorners[i],time_test,CV_intervals[cube[i]], CV_values[cube[i]], CV_minima[cube[i]]);
                //                                argmins[CI_vector.size()][i] = time_test;
                //                                for (int mm = 0; mm < (CV_intervals[cube[i]].size()/2); mm++) {
                //                                    if ((time_test >= CV_intervals[cube[i]][2*mm]) && (time_test <= CV_intervals[cube[i]][2*mm+1]) ){
                //                                        intersecting_interval = true;
                //                                        CV_intervals[cube[i]][2*mm] = std::min(CV_intervals[cube[i]][2*mm],time_seed);
                //                                        CV_intervals[cube[i]][2*mm+1] = std::max(CV_intervals[cube[i]][2*mm+1],time_seed);
                //                                        CV_values[cube[i]][mm] = std::min(CV_values[cube[i]][mm],cubeScalars[i]); //  ??
                //                                    }
                //                                }
                //                            }
                //                            if (!in_existing_interval && !intersecting_interval) {
                //                                std::vector<double> interval_i;
                //                                interval_i.push_back(time_test);
                //                                interval_i.push_back(time_seed);
                //                                std::sort(interval_i.begin(), interval_i.end());
                //                                CV_intervals[cube[i]].push_back(interval_i[0]);
                //                                CV_intervals[cube[i]].push_back(interval_i[1]);
                //                                CV_values[cube[i]].push_back(cubeScalars[i]);
                //                                CV_minima[cube[i]].push_back(time_test);
                //                            }
                //                time_test = time_seed;
                //                double debug = scalarFunc(cubeCorners[i],time_test);
                //                if (fabs(debug - cubeScalars[i])>1e-3 ) {
                //                                std::cout << "______________";
                ////                                std::cout << "Seed: " << time_seed << " value " << debug << " at time " << time_test << std::endl;
                //                                for (int mm = 0; mm < (CV_intervals[cube[i]].size()/2); mm++){
                //                                    std::cout << "Existing interval :" << CV_intervals[cube[i]][2*mm] << " " <<  CV_intervals[cube[i]][2*mm + 1] << " value: " << CV_values[cube[i]][mm] << std::endl;
                //                                }
                //                                std::cout << "Point: " << CV_vector[cube[i]] << std::endl;
                //                }
                
            }else{
                time_test = time_seed;
                intervals[i].resize(0);
                values[i].resize(0);
                minima[i].resize(0);
                cubeScalars[i] = scalarFunc(cubeCorners[i],time_test,intervals[i],values[i],minima[i]);
                argmins[CI_vector.size()][i] = time_test;
            }
            
            running_argmin = running_argmin + (argmins[CI_vector.size()][i]/8.0);
            //            if (cubeScalars[i]>0.3) {
            //                debug_flag = true;
            //            }
            
        }
        //std::cout << time_test << std::endl;
        // If this cube doesn't intersect the surface, disregard it
        bool validCube = false;
        int sign = sgn(cubeScalars[0]);
        for (int i = 1; i < 8; i++) {
            if (sign != sgn(cubeScalars[i])) {
                validCube = true;
                break;
            }
        }
        
        
        
        
        //    if (!validCube) {
        //continue;
        //  }
        
        // Debugging
        //        if (debug_flag) {
        //            std::cout << "voxel" << std::endl;
        //            for (int i = 0; i<8; i++) {
        //                std::cout << "value: " << cubeScalars[i] << " at time" << argmins[CI_vector.size()][i] << " with seed" << time_seed << std::endl;
        //
        //            }
        //        }
        
        
        
        
        
        
        for (int n = 0; n < 30; n++) { // For each neighbor, check the hash table to see if its been added before
            Eigen::RowVector3i nkey = pi + neighbors[n];
            auto nbr = visited.find(nkey);
            flag = false;
            if (nbr != visited.end()) { // We've already visited this neighbor, use references to its vertices instead of duplicating them
                vertexAlreadyAdded |= zv[n];
                for (int i = 0; i < 4; i++) {
                    if (zvv[n][i]!=-1) {
                        cube[zvv[n][i]] = CI_vector[nbr->second][zvv[n % 2 == 0 ? n + 1 : n - 1][i]];
                        //              if( CS_vector[cube[zvv[n][i]]] < cubeScalars[zvv[n][i]] ){
                        //                  // It was better before
                        //                  cubeScalars[zvv[n][i]] = CS_vector[cube[zvv[n][i]]];
                        //              }else
                        if((CS_vector[cube[zvv[n][i]]]>cubeScalars[zvv[n][i]] && (CS_vector[cube[zvv[n][i]]]*cubeScalars[zvv[n][i]])<0) || CS_vector[cube[zvv[n][i]]]>(cubeScalars[zvv[n][i]] + 1e-3)){
                            if (!flag) {
                                queue.push_back(nkey);
                                //time_queue.push_back(time_test);
                                // JAN 24 CHANGE THIS
                                time_queue.push_back(running_argmin);
                                correspondence_queue.push_back(nbr->second);
                                additions_corrections++;
                                flag = true;
                                auto bar = std::make_tuple(nkey, running_argmin, nbr->second, cubeScalars[zvv[n][i]]);
                                p_queue.push(bar);
                            }
                        }
                        
                        
                        // WARNING: THIS BELOW IS WRONG, HAVE TO CHANGE
                        //cubeScalars[zvv[n][i]] = std::min(CS_vector[cube[zvv[n][i]]],cubeScalars[zvv[n][i]]);
                        //CS_vector[cube[zvv[n][i]]] = cubeScalars[zvv[n][i]];
                    }
                }
                //} else if(correspondence==-1) {
            }
        }
        
        validCube = false;
        sign = sgn(cubeScalars[0]);
        for (int i = 1; i < 8; i++) {
            if (sign != sgn(cubeScalars[i])) {
                validCube = true;
            }
        }
        bool validCube_before = validCube;
        
        for (int n = 0; n < 30; n++) { // For each neighbor, check the hash table to see if its been added before
            Eigen::RowVector3i nkey = pi + neighbors[n];
            auto nbr = visited.find(nkey);
            flag = false;
            if (nbr != visited.end()) { // We've already visited this neighbor, use references to its vertices instead of duplicating them
                for (int i = 0; i < 4; i++) {
                    if (zvv[n][i]!=-1) {
                        cube[zvv[n][i]] = CI_vector[nbr->second][zvv[n % 2 == 0 ? n + 1 : n - 1][i]];
                        // WARNING: THIS BELOW IS WRONG, HAVE TO CHANGE
                        cubeScalars[zvv[n][i]] = std::min(CS_vector[cube[zvv[n][i]]],cubeScalars[zvv[n][i]]);
                        CS_vector[cube[zvv[n][i]]] = cubeScalars[zvv[n][i]];
                    }
                }
                //} else if(correspondence==-1) {
            }else{
                //                validCube = false;
                //                sign = sgn(cubeScalars[0]);
                //                for (int i = 1; i < 8; i++) {
                //                    if (sign != sgn(cubeScalars[i])) {
                //                        validCube = true;
                //                    }
                //                }
                if (validCube) {
                    //                    if (debug_flag) {
                    //                        std::cout << "BAD THING HAPPENED" << std::endl;
                    //                    }
                    //                    queue.push_back(nkey);
                    //                    // Otherwise, we have not visited the neighbor, put it in the BFS queue
                    //                   // time_queue.push_back(time_test);
                    //                    time_queue.push_back(running_argmin);
                    //                    correspondence_queue.push_back(-1);
                }
                
            }
        }
  // WARNING THIS SHOULDNT BE COMMENTED!!
        validCube = false;
        sign = sgn(cubeScalars[0]);
        for (int i = 1; i < 8; i++) {
            if (sign != sgn(cubeScalars[i])) {
                validCube = true;
            }
        }
        
        
        for (int n = 0; n < 6; n++) { // For each neighbor, check the hash table to see if its been added before
            Eigen::RowVector3i nkey = pi + neighbors[n];
            auto nbr = visited.find(nkey);
            flag = false;
            if (nbr == visited.end()) {
                if(validCube && validCube_before){
                    queue.push_back(nkey);
                    // Otherwise, we have not visited the neighbor, put it in the BFS queue
                    // time_queue.push_back(time_test);
                    
                    time_queue.push_back(running_argmin);
                    correspondence_queue.push_back(-1);
                    auto bar = std::make_tuple(nkey, running_argmin, -1, 0.0);
                    p_queue.push(bar);
                    additions_normal++;
                }
            }
        }
        
        
        
        
        
        auto did_we_visit_this_one = visited.find(pi);
        if (correspondence==-1 && did_we_visit_this_one==visited.end()) {
            for (int i = 0; i < 8; i++) { // Add new, non-visited,2 vertices to the arrays
                //if (0 == ((1 << i) & vertexAlreadyAdded)) {
                if (0 == ((1 << i) & vertexAlreadyAdded)) {
//                    std::vector<double> interval;
//                    std::vector<double> interval_values;
//                    std::vector<double> interval_minima;
//                    std::vector<double> interval_big;
//                    std::vector<double> interval_values_big;
//                    std::vector<double> interval_minima_big;
//                    interval_minima.push_back(argmins[CI_vector.size()][i]);
//                    interval.push_back(argmins[CI_vector.size()][i]);
//                    interval.push_back(time_seed);
//                    std::sort(interval.begin(), interval.end());
//                    interval_values.push_back(cubeScalars[i]);
                    
                    
                    CV_intervals.push_back(intervals[i]);
                    CV_values.push_back(values[i]);
                    CV_minima.push_back(minima[i]);
                    cube[i] = CS_vector.size();
                    CV_vector.push_back(cubeCorners[i]);
                    CS_vector.push_back(cubeScalars[i]);
                    CV_argmins.push_back(argmins[CI_vector.size()][i]);
                }
            }
            
            visited[pi] = CI_vector.size();
            CI_vector.push_back(cube);
        }
        
        //std::cout << queue.size() << std::endl;
        
        
        bool debug = false;
        if (debug) {
            CV.resize(CV_vector.size(), 3);
            CV_argmins_vector.resize(CV_vector.size(), 1);
            CS.resize(CS_vector.size(), 1);
            CI.resize(CI_vector.size(), 8);
            Eigen::MatrixXi Q;
            Q.resize(queue.size(),3);
            for (int i = 0; i < queue.size(); i++) {
                Q.row(i) = queue[i];
            }
            // If you pass in column-major matrices, this is going to be slooooowwwww
            for (int i = 0; i < CV_vector.size(); i++) {
                CV.row(i) = CV_vector[i];
            }
            for (int i = 0; i < CS_vector.size(); i++) {
                CS(i) = CS_vector[i];
            }
            for (int i = 0; i < CI_vector.size(); i++) {
                CI.row(i) = CI_vector[i];
            }
            igl::writeDMAT("/Volumes/Seagate Hard Drive/debug/CS" + std::to_string(counter) + ".dmat",CS);
            igl::writeDMAT("/Volumes/Seagate Hard Drive/debug/CV" + std::to_string(counter) + ".dmat",CV);
            igl::writeDMAT("/Volumes/Seagate Hard Drive/debug/CI" + std::to_string(counter) + ".dmat",CI);
            igl::writeDMAT("/Volumes/Seagate Hard Drive/debug/Q" + std::to_string(counter) + ".dmat",Q);
            counter = counter + 1;
        }
        
        
        
        
        
    }
    //std::cout << "test" << std::endl;
//    std::cout << " Normal: " << additions_normal << std::endl;
//    std::cout << " Corrections: " << additions_corrections << std::endl;
//    std::cout << " Self: " << additions_self << std::endl;
    
    
    
    CV.conservativeResize(CV_vector.size(), 3);
    CV_argmins_vector.conservativeResize(CV_vector.size(), 1);
    CS.conservativeResize(CS_vector.size(), 1);
    CI.conservativeResize(CI_vector.size(), 8);
//    // If you pass in column-major matrices, this is going to be slooooowwwww
//    for (int j = 0; j < CV_vector.size(); j++) {
//        std::cout << "______________" << std::endl;
//        for (int mm = 0; mm < (CV_intervals[j].size()/2); mm++){
//            std::cout << "Existing interval: " << CV_intervals[j][2*mm] << " " <<  CV_intervals[j][2*mm + 1] << " value: " << CV_values[j][mm] << std::endl;
//        }
//    }
                    
    for (int i = 0; i < CV_vector.size(); i++) {
        CV.row(i) = CV_vector[i];
    }
    for (int i = 0; i < CS_vector.size(); i++) {
        CS(i) = CS_vector[i];
        //CS(i) = std::min_element(CV_values[i].begin(), CV_values[i].end());
    }
    for (int i = 0; i < CI_vector.size(); i++) {
        CI.row(i) = CI_vector[i];
    }
    for (int i = 0; i < CV_argmins.size(); i++) {
        double val = 100.0;
        double argmin = 0.0;
        for(int s = 0; s<CV_values[i].size(); s++){
        for (int mm = 0; mm<CV_values[i][s].size(); mm++) {
            if (CV_values[i][s][mm]<val) {
                val = CV_values[i][s][mm];
                argmin = CV_minima[i][s][mm];
            }
        }
        }
        CV_argmins_vector(i) = argmin;
    }
}





void sparse_continuation(const Eigen::RowVector3d p0, const std::vector<Eigen::RowVector3i> init_voxels, const std::vector<Eigen::RowVectorXd> t0, const  std::function<double(const Eigen::RowVector3d &, Eigen::RowVectorXd &, std::vector<std::vector<Eigen::RowVectorXd>> &, std::vector<std::vector<double>> &, std::vector<std::vector<Eigen::RowVectorXd>> &)> scalarFunc, const double eps, const int expected_number_of_cubes, Eigen::VectorXd & CS, Eigen::MatrixXd & CV, Eigen::MatrixXi & CI, Eigen::MatrixXd & CV_argmins_vector){
    
    struct IndexRowVectorHash  {
        std::size_t operator()(const Eigen::RowVector3i& key) const {
            std::size_t seed = 0;
            std::hash<int> hasher;
            for (int i = 0; i < 3; i++) {
            seed ^= hasher(key[i]) + 0x9e3779b9 + (seed<<6) + (seed>>2); // Copied from boost::hash_combine
            }
            return seed;
        }
    };
    
    auto sgn = [](double val) -> int {
        return (double(0) < val) - (val < double(0));
    };
    
    double half_eps = 0.5 * eps;
    
    std::vector<Eigen::Matrix<int,1,8>> CI_vector;
    std::vector<Eigen::RowVector3d> CV_vector;
    std::vector<std::vector<std::vector<Eigen::RowVectorXd>>> CV_intervals;
    std::vector<std::vector<std::vector<double>>> CV_values;
    std::vector<std::vector<std::vector<Eigen::RowVectorXd>>> CV_minima;
    std::vector<double> CS_vector;
    CI_vector.reserve(expected_number_of_cubes);
    CV_vector.reserve(8 * expected_number_of_cubes);
    CS_vector.reserve(8 * expected_number_of_cubes);
    std::vector<std::vector<Eigen::RowVectorXd>> argmins;
    std::vector<Eigen::RowVectorXd> CV_argmins;
    
    argmins.reserve(32 * expected_number_of_cubes);
    int counter = 0;
    
    // Track visisted neighbors
    std::unordered_map<Eigen::RowVector3i, int, IndexRowVectorHash> visited;
    visited.reserve(6 * expected_number_of_cubes);
    visited.max_load_factor(0.5);
    
    // BFS Queue
    std::vector<Eigen::RowVector3i> queue;
    queue.reserve(expected_number_of_cubes * 8);
    std::vector<Eigen::RowVectorXd> time_queue;
    time_queue.reserve(expected_number_of_cubes * 8);
    std::vector<int> correspondence_queue;
    correspondence_queue.reserve(expected_number_of_cubes * 8);
    std::vector<double> intervals_turn, values_turn, minima_turn;
    
    for (int seed_ind = 0; seed_ind < init_voxels.size(); seed_ind++) {
        
        double min_turn = 1000.0;
        Eigen::VectorXd final_seed = t0[seed_ind];

        queue.push_back(init_voxels[seed_ind]);
        time_queue.push_back(final_seed);
        correspondence_queue.push_back(-1);
    }

    while (queue.size() > 0)
    {
        Eigen::RowVector3i pi = queue.back();
        queue.pop_back();
        Eigen::VectorXd time_seed = time_queue.back();
        time_queue.pop_back();
        int correspondence = correspondence_queue.back();
        correspondence_queue.pop_back();
        
        Eigen::RowVector3d ctr = p0 + eps*pi.cast<double>(); // R^3 center of this cube
        
        // X, Y, Z basis vectors, and array of neighbor offsets used to construct cubes
        const Eigen::RowVector3i bx(1, 0, 0), by(0, 1, 0), bz(0, 0, -1);
        const std::array<Eigen::RowVector3i, 30> neighbors = {
            bx, -bx, by, -by, bz, -bz,
            by-bz, -by+bz, // 1-2 4-7
            bx+by, -bx-by, // 0-1 7-6
            by+bz, -by-bz,  // 0-3 6-5
            by-bx, -by+bx,  // 2-3 5-4
            bx-bz, -bx+bz, // 1-5 3-7
            bx+bz, -bx-bz, // 0-4 2-6
            -bx+by+bz, bx-by-bz, // 3 5
            bx+by+bz, -bx-by-bz, // 0 6
            bx+by-bz, -bx-by+bz, //1 7
            -bx+by-bz, bx-by+bz, // 2 4,
            bx-bx, bx-bx,
            bx-bx, bx-bx
        };
        
        // Compute the position of the cube corners and the scalar values at those corners
        std::array<Eigen::RowVector3d, 8> cubeCorners = {
            ctr+half_eps*(bx+by+bz).cast<double>(), ctr+half_eps*(bx+by-bz).cast<double>(), ctr+half_eps*(-bx+by-bz).cast<double>(), ctr+half_eps*(-bx+by+bz).cast<double>(),
            ctr+half_eps*(bx-by+bz).cast<double>(), ctr+half_eps*(bx-by-bz).cast<double>(), ctr+half_eps*(-bx-by-bz).cast<double>(), ctr+half_eps*(-bx-by+bz).cast<double>()
        };
        std::array<double, 8> cubeScalars;
        //double time_seed = 0.0;
        //std::cout << time_seed << std::endl;
        Eigen::RowVectorXd time_test;
        argmins[CI_vector.size()].resize(8);
        std::vector<Eigen::RowVectorXd> argmins_cube;
        
        
        
        // Add the cube vertices and indices to the output arrays if they are not there already
        
        uint8_t vertexAlreadyAdded = 0; // This is a bimask. If a bit is 1, it has been visited already by the BFS
        constexpr std::array<uint8_t, 30> zv = {
            (1 << 0) | (1 << 1) | (1 << 4) | (1 << 5),
            (1 << 2) | (1 << 3) | (1 << 6) | (1 << 7),
            (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
            (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),
            (1 << 0) | (1 << 3) | (1 << 4) | (1 << 7),
            (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6),
            (1 << 1) | (1 << 2),
            (1 << 4) | (1 << 7),
            (1 << 0) | (1 << 1),
            (1 << 6) | (1 << 7),
            (1 << 0) | (1 << 3),
            (1 << 5) | (1 << 6),
            (1 << 2) | (1 << 3),
            (1 << 4) | (1 << 5),
            (1 << 1) | (1 << 5),
            (1 << 3) | (1 << 7),
            (1 << 0) | (1 << 4),
            (1 << 2) | (1 << 6),
            (1 << 3), (1 << 5), // diagonals
            (1 << 0), (1 << 6),
            (1 << 1), (1 << 7),
            (1 << 2), (1 << 4),
            (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
            (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
            (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),
            (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),
        };
        constexpr std::array<std::array<int, 4>, 30> zvv {{
            {{0, 1, 4, 5}}, {{3, 2, 7, 6}}, {{0, 1, 2, 3}},
            {{4, 5, 6, 7}}, {{0, 3, 4, 7}}, {{1, 2, 5, 6}},
            {{-1,-1,1,2}}, {{-1,-1,4,7}}, {{-1,-1,0,1}},{{-1,-1,7,6}},
            {{-1,-1,0,3}}, {{-1,-1,5,6}}, {{-1,-1,2,3}}, {{-1,-1,5,4}},
            {{-1,-1,1,5}}, {{-1,-1,3,7}}, {{-1,-1,0,4}}, {{-1,-1,2,6}},
            {{-1,-1,-1,3}}, {{-1,-1,-1,5}}, {{-1,-1,-1,0}}, {{-1,-1,-1,6}},
            {{-1,-1,-1,1}}, {{-1,-1,-1,7}}, {{-1,-1,-1,2}}, {{-1,-1,-1,4}},
            {{0,1,2,3}}, {{0,1,2,3}}, {{4,5,6,7}}, {{4,5,6,7}}
        }};
        bool flag = false;
        
        
        Eigen::Matrix<int,1,8> cube;
        cube << -1, -1, -1, -1, -1, -1, -1, -1;
        for (int n = 0; n < 30; n++) { // For each neighbor, check the hash table to see if its been added before
            Eigen::RowVector3i nkey = pi + neighbors[n];
            auto nbr = visited.find(nkey);
            if (nbr != visited.end()) {
                for (int i = 0; i < 4; i++) {
                    if (zvv[n][i]!=-1) {
                        cube[zvv[n][i]] = CI_vector[nbr->second][zvv[n % 2 == 0 ? n + 1 : n - 1][i]];
                    }
                }
            }
        }
        
        std::vector<std::vector<std::vector<Eigen::RowVectorXd>>> intervals;
        intervals.resize(8);
        std::vector<std::vector<std::vector<double>>> values;
        values.resize(8);
        std::vector<std::vector<std::vector<Eigen::RowVectorXd>>> minima;
        minima.resize(8);
        
        bool debug_flag = false;
        bool in_existing_interval = false;
        bool intersecting_interval = false;
        time_test = time_seed;
        Eigen::RowVectorXd running_argmin;
        running_argmin.resize(time_seed.size());
        running_argmin.setZero();
        for (int i = 0; i < 8; i++){
            time_test = time_seed;
            //            cubeScalars[i] = scalarFunc(cubeCorners[i],time_test);
            //            argmins[CI_vector.size()][i] = time_test;
            //   running_argmin = running_argmin + (time_test/8.0);
            
            if (cube[i] >= 0) {
               // std::cout << " CALL FUNC " << std::endl;
                cubeScalars[i] = scalarFunc(cubeCorners[i],time_test,CV_intervals[cube[i]], CV_values[cube[i]], CV_minima[cube[i]]);
               // std::cout << " end CALL FUNC " << std::endl;
                argmins[CI_vector.size()][i] = time_test;
                minima[i] = CV_minima[cube[i]];
                values[i] = CV_values[cube[i]];
                intervals[i] = CV_intervals[cube[i]];
                
                if (correspondence==-1) {
                double temp = cubeScalars[i];
                int temp_i = -1;
                    int temp_s = -1;
                    for (int s = 0; s < CV_intervals[cube[i]].size(); s++) {
                for (int mm = 0; mm < (CV_intervals[cube[i]][s].size()/2); mm++){
                    if (CV_values[cube[i]][s][mm]+1e-3 < temp) {
                        temp = CV_values[cube[i]][s][mm];
                        temp_i = mm;
                        temp_s = s;
                    }
                }}
                if (temp_i > -1) {
                    queue.push_back(pi);
                    // Otherwise, we have not visited the neighbor, put it in the BFS queue
                    // time_queue.push_back(time_test);
                    time_queue.push_back(CV_minima[cube[i]][temp_s][temp_i]);
                    correspondence_queue.push_back(1);
                }
                }
                // DEBUGGING
                //                if (cube[i]==10) {
//                std::cout << "______________";
//                std::cout << "Time seed: " << time_seed << std::endl;
//                for (int mm = 0; mm < (CV_intervals[cube[i]].size()/2); mm++){
//                    std::cout << "Existing interval: " << CV_intervals[cube[i]][2*mm] << " " <<  CV_intervals[cube[i]][2*mm + 1] << " value: " << CV_values[cube[i]][mm] << std::endl;
//                }
                //                }
                //                            in_existing_interval = false;
                //                            for (int mm = 0; mm < (CV_intervals[cube[i]].size()/2); mm++) {
                //                                if ((time_test >= CV_intervals[cube[i]][2*mm]) && (time_test <= CV_intervals[cube[i]][2*mm+1]) ){
                //                                    cubeScalars[i] = CV_values[cube[i]][mm];
                //                                    argmins[CI_vector.size()][i] = CV_minima[cube[i]][mm];
                //                                    in_existing_interval = true;
                //                                }
                //                            }
                //                            if (!in_existing_interval) {
                //                                intersecting_interval = false;
                //                                time_test = time_seed;
                //                                cubeScalars[i] = scalarFunc(cubeCorners[i],time_test,CV_intervals[cube[i]], CV_values[cube[i]], CV_minima[cube[i]]);
                //                                argmins[CI_vector.size()][i] = time_test;
                //                                for (int mm = 0; mm < (CV_intervals[cube[i]].size()/2); mm++) {
                //                                    if ((time_test >= CV_intervals[cube[i]][2*mm]) && (time_test <= CV_intervals[cube[i]][2*mm+1]) ){
                //                                        intersecting_interval = true;
                //                                        CV_intervals[cube[i]][2*mm] = std::min(CV_intervals[cube[i]][2*mm],time_seed);
                //                                        CV_intervals[cube[i]][2*mm+1] = std::max(CV_intervals[cube[i]][2*mm+1],time_seed);
                //                                        CV_values[cube[i]][mm] = std::min(CV_values[cube[i]][mm],cubeScalars[i]); //  ??
                //                                    }
                //                                }
                //                            }
                //                            if (!in_existing_interval && !intersecting_interval) {
                //                                std::vector<double> interval_i;
                //                                interval_i.push_back(time_test);
                //                                interval_i.push_back(time_seed);
                //                                std::sort(interval_i.begin(), interval_i.end());
                //                                CV_intervals[cube[i]].push_back(interval_i[0]);
                //                                CV_intervals[cube[i]].push_back(interval_i[1]);
                //                                CV_values[cube[i]].push_back(cubeScalars[i]);
                //                                CV_minima[cube[i]].push_back(time_test);
                //                            }
                //                time_test = time_seed;
                //                double debug = scalarFunc(cubeCorners[i],time_test);
                //                if (fabs(debug - cubeScalars[i])>1e-3 ) {
                //                                std::cout << "______________";
                ////                                std::cout << "Seed: " << time_seed << " value " << debug << " at time " << time_test << std::endl;
                //                                for (int mm = 0; mm < (CV_intervals[cube[i]].size()/2); mm++){
                //                                    std::cout << "Existing interval :" << CV_intervals[cube[i]][2*mm] << " " <<  CV_intervals[cube[i]][2*mm + 1] << " value: " << CV_values[cube[i]][mm] << std::endl;
                //                                }
                //                                std::cout << "Point: " << CV_vector[cube[i]] << std::endl;
                //                }
                
            }else{
                time_test = time_seed;
                intervals[i].resize(0);
                values[i].resize(0);
                minima[i].resize(0);
          //      std::cout << " CALL FUNC " << std::endl;
                cubeScalars[i] = scalarFunc(cubeCorners[i],time_test,intervals[i],values[i],minima[i]);
          //      std::cout << " end CALL FUNC " << std::endl;
                argmins[CI_vector.size()][i] = time_test;
            }
            
            running_argmin = running_argmin + (argmins[CI_vector.size()][i]/8.0);
            //            if (cubeScalars[i]>0.3) {
            //                debug_flag = true;
            //            }
            
        }
        //std::cout << time_test << std::endl;
        // If this cube doesn't intersect the surface, disregard it
        bool validCube = false;
        int sign = sgn(cubeScalars[0]);
        for (int i = 1; i < 8; i++) {
            if (sign != sgn(cubeScalars[i])) {
                validCube = true;
                break;
            }
        }
        
        
        
        
        //    if (!validCube) {
        //continue;
        //  }
        
        // Debugging
        //        if (debug_flag) {
        //            std::cout << "voxel" << std::endl;
        //            for (int i = 0; i<8; i++) {
        //                std::cout << "value: " << cubeScalars[i] << " at time" << argmins[CI_vector.size()][i] << " with seed" << time_seed << std::endl;
        //
        //            }
        //        }
        
        
        
        
        
        
        for (int n = 0; n < 6; n++) { // For each neighbor, check the hash table to see if its been added before
            Eigen::RowVector3i nkey = pi + neighbors[n];
            auto nbr = visited.find(nkey);
            flag = false;
            if (nbr != visited.end()) { // We've already visited this neighbor, use references to its vertices instead of duplicating them
                vertexAlreadyAdded |= zv[n];
                for (int i = 0; i < 4; i++) {
                    if (zvv[n][i]!=-1) {
                        cube[zvv[n][i]] = CI_vector[nbr->second][zvv[n % 2 == 0 ? n + 1 : n - 1][i]];
                        //              if( CS_vector[cube[zvv[n][i]]] < cubeScalars[zvv[n][i]] ){
                        //                  // It was better before
                        //                  cubeScalars[zvv[n][i]] = CS_vector[cube[zvv[n][i]]];
                        //              }else
                        if((CS_vector[cube[zvv[n][i]]]>cubeScalars[zvv[n][i]] && (CS_vector[cube[zvv[n][i]]]*cubeScalars[zvv[n][i]])<0) || CS_vector[cube[zvv[n][i]]]>(cubeScalars[zvv[n][i]] + 1e-3)){
                            if (!flag) {
                                queue.push_back(nkey);
                                //time_queue.push_back(time_test);
                                time_queue.push_back(running_argmin);
                                correspondence_queue.push_back(nbr->second);
                                flag = true;
                            }
                        }
                        
                        
                        // WARNING: THIS BELOW IS WRONG, HAVE TO CHANGE
                        //cubeScalars[zvv[n][i]] = std::min(CS_vector[cube[zvv[n][i]]],cubeScalars[zvv[n][i]]);
                        //CS_vector[cube[zvv[n][i]]] = cubeScalars[zvv[n][i]];
                    }
                }
                //} else if(correspondence==-1) {
            }
        }
        
        validCube = false;
        sign = sgn(cubeScalars[0]);
        for (int i = 1; i < 8; i++) {
            if (sign != sgn(cubeScalars[i])) {
                validCube = true;
            }
        }
        
        
        for (int n = 0; n < 30; n++) { // For each neighbor, check the hash table to see if its been added before
            Eigen::RowVector3i nkey = pi + neighbors[n];
            auto nbr = visited.find(nkey);
            flag = false;
            if (nbr != visited.end()) { // We've already visited this neighbor, use references to its vertices instead of duplicating them
                for (int i = 0; i < 4; i++) {
                    if (zvv[n][i]!=-1) {
                        cube[zvv[n][i]] = CI_vector[nbr->second][zvv[n % 2 == 0 ? n + 1 : n - 1][i]];
                        // WARNING: THIS BELOW IS WRONG, HAVE TO CHANGE
                        cubeScalars[zvv[n][i]] = std::min(CS_vector[cube[zvv[n][i]]],cubeScalars[zvv[n][i]]);
                        CS_vector[cube[zvv[n][i]]] = cubeScalars[zvv[n][i]];
                    }
                }
                //} else if(correspondence==-1) {
            }else{
                //                validCube = false;
                //                sign = sgn(cubeScalars[0]);
                //                for (int i = 1; i < 8; i++) {
                //                    if (sign != sgn(cubeScalars[i])) {
                //                        validCube = true;
                //                    }
                //                }
                if (validCube) {
                    //                    if (debug_flag) {
                    //                        std::cout << "BAD THING HAPPENED" << std::endl;
                    //                    }
                    //                    queue.push_back(nkey);
                    //                    // Otherwise, we have not visited the neighbor, put it in the BFS queue
                    //                   // time_queue.push_back(time_test);
                    //                    time_queue.push_back(running_argmin);
                    //                    correspondence_queue.push_back(-1);
                }
                
            }
        }
  // WARNING THIS SHOULDNT BE COMMENTED!!
//        validCube = false;
//        sign = sgn(cubeScalars[0]);
//        for (int i = 1; i < 8; i++) {
//            if (sign != sgn(cubeScalars[i])) {
//                validCube = true;
//            }
//        }
        for (int n = 0; n < 6; n++) { // For each neighbor, check the hash table to see if its been added before
            Eigen::RowVector3i nkey = pi + neighbors[n];
            auto nbr = visited.find(nkey);
            flag = false;
            if (nbr == visited.end()) {
                if(validCube){
                    queue.push_back(nkey);
                    // Otherwise, we have not visited the neighbor, put it in the BFS queue
                    // time_queue.push_back(time_test);
                    time_queue.push_back(running_argmin);
                    correspondence_queue.push_back(-1);
                }
            }
        }
        
        
        
        
        
        auto did_we_visit_this_one = visited.find(pi);
        if (correspondence==-1 && did_we_visit_this_one==visited.end()) {
            for (int i = 0; i < 8; i++) { // Add new, non-visited,2 vertices to the arrays
                //if (0 == ((1 << i) & vertexAlreadyAdded)) {
                if (0 == ((1 << i) & vertexAlreadyAdded)) {
//                    std::vector<double> interval;
//                    std::vector<double> interval_values;
//                    std::vector<double> interval_minima;
//                    std::vector<double> interval_big;
//                    std::vector<double> interval_values_big;
//                    std::vector<double> interval_minima_big;
//                    interval_minima.push_back(argmins[CI_vector.size()][i]);
//                    interval.push_back(argmins[CI_vector.size()][i]);
//                    interval.push_back(time_seed);
//                    std::sort(interval.begin(), interval.end());
//                    interval_values.push_back(cubeScalars[i]);
                    
                    
                    CV_intervals.push_back(intervals[i]);
                    CV_values.push_back(values[i]);
                    CV_minima.push_back(minima[i]);
                    cube[i] = CS_vector.size();
                    CV_vector.push_back(cubeCorners[i]);
                    CS_vector.push_back(cubeScalars[i]);
                    CV_argmins.push_back(argmins[CI_vector.size()][i]);
                }
            }
            
            visited[pi] = CI_vector.size();
            CI_vector.push_back(cube);
        }
        
        //std::cout << queue.size() << std::endl;
        
        
        bool debug = false;
        if (debug) {
            CV.resize(CV_vector.size(), 3);
            CV_argmins_vector.resize(CV_vector.size(), 1);
            CS.resize(CS_vector.size(), 1);
            CI.resize(CI_vector.size(), 8);
            Eigen::MatrixXi Q;
            Q.resize(queue.size(),3);
            for (int i = 0; i < queue.size(); i++) {
                Q.row(i) = queue[i];
            }
            // If you pass in column-major matrices, this is going to be slooooowwwww
            for (int i = 0; i < CV_vector.size(); i++) {
                CV.row(i) = CV_vector[i];
            }
            for (int i = 0; i < CS_vector.size(); i++) {
                CS(i) = CS_vector[i];
            }
            for (int i = 0; i < CI_vector.size(); i++) {
                CI.row(i) = CI_vector[i];
            }
            igl::writeDMAT("/Volumes/Seagate Hard Drive/debug/CS" + std::to_string(counter) + ".dmat",CS);
            igl::writeDMAT("/Volumes/Seagate Hard Drive/debug/CV" + std::to_string(counter) + ".dmat",CV);
            igl::writeDMAT("/Volumes/Seagate Hard Drive/debug/CI" + std::to_string(counter) + ".dmat",CI);
            igl::writeDMAT("/Volumes/Seagate Hard Drive/debug/Q" + std::to_string(counter) + ".dmat",Q);
            counter = counter + 1;
        }
        
        
        
        
        
    }
    //std::cout << "test" << std::endl;
    
    CV.conservativeResize(CV_vector.size(), 3);
    CV_argmins_vector.conservativeResize(CV_vector.size(), 2);
    CS.conservativeResize(CS_vector.size(), 1);
    CI.conservativeResize(CI_vector.size(), 8);
//    // If you pass in column-major matrices, this is going to be slooooowwwww
//    for (int j = 0; j < CV_vector.size(); j++) {
//        std::cout << "______________" << std::endl;
//        for (int mm = 0; mm < (CV_intervals[j].size()/2); mm++){
//            std::cout << "Existing interval: " << CV_intervals[j][2*mm] << " " <<  CV_intervals[j][2*mm + 1] << " value: " << CV_values[j][mm] << std::endl;
//        }
//    }
                    
    for (int i = 0; i < CV_vector.size(); i++) {
        CV.row(i) = CV_vector[i];
    }
    for (int i = 0; i < CS_vector.size(); i++) {
        CS(i) = CS_vector[i];
        //CS(i) = std::min_element(CV_values[i].begin(), CV_values[i].end());
    }
    for (int i = 0; i < CI_vector.size(); i++) {
        CI.row(i) = CI_vector[i];
    }
    for (int i = 0; i < CV_vector.size(); i++) {
        std::cout << "1" << std::endl;
        double val = CV_values[i][0][0];
        Eigen::RowVectorXd argmin;
        argmin = CV_minima[i][0][0];
        std::cout << "2" << std::endl;
        for(int s = 0; s<CV_values[i].size(); s++){
        for (int mm = 0; mm<CV_values[i][s].size(); mm++) {
            if (CV_values[i][s][mm]<val) {
                std::cout << "3" << std::endl;
                val = CV_values[i][s][mm];
                std::cout << "4" << std::endl;
                argmin = CV_minima[i][s][mm];
            }
        }
        }
        std::cout << "5" << std::endl;
        std::cout << argmin << std::endl;
        CV_argmins_vector.row(i) = argmin;
        std::cout << "5b" << std::endl;
    }
    std::cout << "6" << std::endl;
}





