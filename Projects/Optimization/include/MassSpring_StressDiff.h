#ifndef MASS_SPRING_STRESS_DIFF_H
#define MASS_SPRING_STRESS_DIFF_H

#include "vector_manipulation.h"

Eigen::Matrix<AScalar, 3, 6> dtdx(AScalar Young, Vector6a X, Vector6a x, Vector3a normal){
    
    std::vector<Vector6a> dSdx(3);

    AScalar t1 = x[0] - x[3];
    AScalar t2 = x[1] - x[4];
    AScalar t3 = pow(t2, 0.2e1);
    AScalar t4 = pow(t1, 0.2e1);
    AScalar t5 = t3 + t4;
    AScalar t6 = pow(t5, -0.3e1 / 0.2e1);
    AScalar t7 = X[3] - X[0];
    AScalar t8 = X[4] - X[1];
    t7 = pow(t7, 0.2e1) + pow(t8, 0.2e1);
    t7 = 0.1e1 / t7;
    t8 = t5 * t6;
    AScalar t9 = t2 * normal[1];
    AScalar t10 = t1 * normal[0];
    AScalar t11 = t8 * (t10 + t9);
    AScalar t12 = t4 * t6;
    t9 = -t9 * t6 * t1 - t12 * normal[0] + t8 * normal[0];
    AScalar t13 = t8 * t9;
    t5 = (-t5 * t7 + 0.1e1) * Young;
    AScalar t14 = 0.1e1 / 0.2e1;
    t4 = t8 * t4 * Young * t7 * t11;
    AScalar t15 = t3 * t6;
    t10 = -t10 * t6 * t2 - t15 * normal[1] + t8 * normal[1];
    t6 = t6 * t11;
    AScalar t16 = t6 * t2;
    AScalar t17 = t8 * t10;
    AScalar t18 = t5 * t1;
    AScalar t19 = t8 * t1;
    AScalar t20 = t19 * Young * t2 * t7 * t11;
    t10 = t8 * t10;
    t6 = t6 * t1;
    AScalar t21 = t5 * t2;
    t3 = t8 * t3 * Young * t7 * t11;
    dSdx[0][0] = t14 * t5 * (t11 * (t12 - t8) - t13 * t1) + t4;
    dSdx[0][1] = -t14 * t18 * (-t16 + t17) + t20;
    dSdx[0][2] = 0;
    dSdx[0][3] = t14 * t5 * (t11 * (-t12 + t8) + t19 * t9) - t4;
    dSdx[0][4] = t14 * t18 * (t10 - t16) - t20;
    dSdx[0][5] = 0;
    dSdx[1][0] = -t14 * t21 * (t13 - t6) + t20;
    dSdx[1][1] = t14 * t5 * (t11 * (t15 - t8) - t17 * t2) + t3;
    dSdx[1][2] = 0;
    dSdx[1][3] = t14 * t21 * (t8 * t9 - t6) - t20;
    dSdx[1][4] = t14 * t5 * (t11 * (-t15 + t8) + t10 * t2) - t3;
    dSdx[1][5] = 0;
    dSdx[2][0] = 0;
    dSdx[2][1] = 0;
    dSdx[2][2] = 0;
    dSdx[2][3] = 0;
    dSdx[2][4] = 0;
    dSdx[2][5] = 0;

    Eigen::Matrix<AScalar, 3, 6> res;
    for(int i = 0; i < res.rows(); ++i){
        res.row(i) = dSdx[i].transpose();
    }
    return res;
}

Vector6a dbdx_inv(Vector6a x, Vector3a normal){
    
    Vector6a dbdx;
    AScalar t1 = x[0] - x[3];
    AScalar t2 = x[1] - x[4];
    AScalar t3 = pow(t2, 0.2e1);
    AScalar t4 = pow(t1, 0.2e1);
    AScalar t5 = t4 + t3;
    AScalar t6 = pow(t5, -0.3e1 / 0.2e1);
    t5 = t5 * t6;
    AScalar t7 = t2 * normal[1];
    AScalar t8 = t1 * normal[0];
    AScalar t9 = t5 * (t7 + t8);
    AScalar t10 = std::abs(t9);
    t1 = -t7 * t6 * t1 - t6 * t4 * normal[0] + t5 * normal[0];
    t4 = std::abs(t9) / t9;
    t2 = -t8 * t6 * t2 - t6 * t3 * normal[1] + t5 * normal[1];
    t3 = pow(t10, -0.2e1);
    dbdx[0] = -t3 * t1 * t4;
    dbdx[1] = -t3 * t2 * t4;
    dbdx[2] = 0;
    dbdx[3] = t3 * t1 * t4;
    dbdx[4] = t3 * t2 * t4;
    dbdx[5] = 0;
    return dbdx;
}

#endif