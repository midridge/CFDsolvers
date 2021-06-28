//
// Created by ennis on 2021/5/30.
//

#include <Eigen/Dense>
#include <string>

typedef Eigen::ArrayXXd Array;

bool exportResult(const std::string &name, unsigned time_step_count, Array &x, Array &y, Array &u, Array &v, Array &p);
bool readResult(const std::string &name, Array &u, Array &v, Array &p);