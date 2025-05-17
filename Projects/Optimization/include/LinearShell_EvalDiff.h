#ifndef LINEAR_SHELL_STRESS_DIFF_H
#define LINEAR_SHELL_STRESS_DIFF_H

#include "vector_manipulation.h"

Eigen::Matrix<AScalar, 4, 9> dSdx(Vector9a& q, Vector6a& p, AScalar lambda, AScalar mu){

    VectorXa dSdx(36);
    AScalar t1 = -q[3] + q[6];
    AScalar t2 = q[6] - q[0];
    AScalar t3 = q[3] - q[0];
    AScalar t4 = t1 * p[1] - t2 * p[3] + t3 * p[5];
    AScalar t5 = p[2] - p[4];
    AScalar t6 = -p[4] + p[0];
    AScalar t7 = p[2] - p[0];
    AScalar t8 = t5 * p[1] - t6 * p[3] - t7 * p[5];
    AScalar t9 = p[3] - p[5];
    t1 = t1 * p[0] - t2 * p[2] + t3 * p[4];
    t2 = -p[5] + p[1];
    t3 = p[3] - p[1];
    AScalar t10 = -t2 * p[2] - t3 * p[4] + t9 * p[0];
    t8 = 0.1e1 / t8;
    t10 = 0.1e1 / t10;
    AScalar t11 = pow(t8, 0.2e1);
    AScalar t12 = pow(t10, 0.2e1);
    AScalar t13 = t1 * t12;
    AScalar t14 = t13 * t5;
    AScalar t15 = t4 * t11;
    AScalar t16 = t15 * t9;
    AScalar t17 = lambda * (t14 + t16);
    t8 = t8 * t10;
    t10 = -q[4] + q[7];
    AScalar t18 = q[7] - q[1];
    AScalar t19 = q[4] - q[1];
    AScalar t20 = t10 * p[1] - t18 * p[3] + t19 * p[5];
    t10 = t10 * p[0] - t18 * p[2] + t19 * p[4];
    t18 = t10 * t12;
    t19 = t18 * t5;
    AScalar t21 = t20 * t11;
    AScalar t22 = t21 * t9;
    AScalar t23 = lambda * (t19 + t22);
    AScalar t24 = -q[5] + q[8];
    AScalar t25 = -q[8] + q[2];
    AScalar t26 = -q[5] + q[2];
    AScalar t27 = t24 * p[1] + t25 * p[3] - t26 * p[5];
    t24 = t24 * p[0] + t25 * p[2] - t26 * p[4];
    t12 = t24 * t12;
    t25 = t12 * t5;
    t11 = t27 * t11;
    t26 = t11 * t9;
    AScalar t28 = lambda * (t25 + t26);
    AScalar t29 = t13 * t6;
    AScalar t30 = t15 * t2;
    AScalar t31 = lambda * (t29 + t30);
    AScalar t32 = t18 * t6;
    AScalar t33 = t21 * t2;
    AScalar t34 = lambda * (t32 + t33);
    AScalar t35 = t12 * t6;
    AScalar t36 = t11 * t2;
    AScalar t37 = lambda * (t35 + t36);
    t13 = t13 * t7;
    t15 = t15 * t3;
    AScalar t38 = lambda * (t13 + t15);
    t18 = t18 * t7;
    t21 = t21 * t3;
    AScalar t39 = lambda * (t18 + t21);
    t12 = t12 * t7;
    t11 = t11 * t3;
    AScalar t40 = lambda * (t12 + t11);
    AScalar t41 = mu * t8 * (t2 * t24 + t27 * t6);
    AScalar t42 = mu * t8 * (t10 * t2 + t20 * t6);
    t2 = mu * t8 * (t1 * t2 + t4 * t6);
    t6 = mu * t8 * (t24 * t9 + t27 * t5);
    AScalar t43 = mu * t8 * (t10 * t9 + t20 * t5);
    t5 = mu * t8 * (t1 * t9 + t4 * t5);
    t9 = mu * t8 * (t24 * t3 + t27 * t7);
    t10 = mu * t8 * (t10 * t3 + t20 * t7);
    t1 = mu * t8 * (t1 * t3 + t4 * t7);
    dSdx[0] = 0.2e1 * t16 * mu + t17;
    dSdx[1] = t5;
    dSdx[2] = t5;
    dSdx[3] = 0.2e1 * t14 * mu + t17;
    dSdx[4] = 0.2e1 * t22 * mu + t23;
    dSdx[5] = t43;
    dSdx[6] = t43;
    dSdx[7] = 0.2e1 * t19 * mu + t23;
    dSdx[8] = 0.2e1 * t26 * mu + t28;
    dSdx[9] = t6;
    dSdx[10] = t6;
    dSdx[11] = 0.2e1 * t25 * mu + t28;
    dSdx[12] = -0.2e1 * t30 * mu - t31;
    dSdx[13] = -t2;
    dSdx[14] = -t2;
    dSdx[15] = -0.2e1 * t29 * mu - t31;
    dSdx[16] = -0.2e1 * t33 * mu - t34;
    dSdx[17] = -t42;
    dSdx[18] = -t42;
    dSdx[19] = -0.2e1 * t32 * mu - t34;
    dSdx[20] = -0.2e1 * t36 * mu - t37;
    dSdx[21] = -t41;
    dSdx[22] = -t41;
    dSdx[23] = -0.2e1 * t35 * mu - t37;
    dSdx[24] = -0.2e1 * t15 * mu - t38;
    dSdx[25] = -t1;
    dSdx[26] = -t1;
    dSdx[27] = -0.2e1 * t13 * mu - t38;
    dSdx[28] = -0.2e1 * t21 * mu - t39;
    dSdx[29] = -t10;
    dSdx[30] = -t10;
    dSdx[31] = -0.2e1 * t18 * mu - t39;
    dSdx[32] = -0.2e1 * t11 * mu - t40;
    dSdx[33] = -t9;
    dSdx[34] = -t9;
    dSdx[35] = -0.2e1 * t12 * mu - t40;

    Eigen::Matrix<AScalar, 4, 9> res;
    for(int i = 0; i < 9; ++i){
        res.col(i) = dSdx.segment(4*i, 4);
    }
    return res;
}

Eigen::Matrix<AScalar, 4, 9> dEdx(Vector9a& q, Vector6a& p){

    AScalar t1 = q[3] - q[6];
    AScalar t2 = -q[6] + q[0];
    AScalar t3 = -q[3] + q[0];
    AScalar t4 = t1 * p[1] - t2 * p[3] + t3 * p[5];
    AScalar t5 = p[2] - p[4];
    AScalar t6 = -p[4] + p[0];
    AScalar t7 = p[2] - p[0];
    AScalar t8 = t5 * p[1] - t6 * p[3] - t7 * p[5];
    AScalar t9 = p[3] - p[5];
    t1 = t1 * p[0] - t2 * p[2] + t3 * p[4];
    t2 = -p[5] + p[1];
    t3 = p[3] - p[1];
    AScalar t10 = t2 * p[2] + t3 * p[4] - t9 * p[0];
    t10 = 0.1e1 / t10;
    t8 = 0.1e1 / t8;
    AScalar t11 = t8 * t10;
    AScalar t12 = q[4] - q[7];
    AScalar t13 = -q[7] + q[1];
    AScalar t14 = -q[4] + q[1];
    AScalar t15 = t12 * p[1] - t13 * p[3] + t14 * p[5];
    t12 = t12 * p[0] - t13 * p[2] + t14 * p[4];
    t13 = q[5] - q[8];
    t14 = -q[8] + q[2];
    AScalar t16 = -q[5] + q[2];
    AScalar t17 = t13 * p[1] - t14 * p[3] + t16 * p[5];
    t13 = t13 * p[0] - t14 * p[2] + t16 * p[4];
    t10 = pow(t10, 0.2e1);
    t8 = pow(t8, 0.2e1);
    t14 = t11 * (t1 * t9 + t4 * t5) / 0.2e1;
    t16 = t11 * (t12 * t9 + t15 * t5) / 0.2e1;
    AScalar t18 = t11 * (t13 * t9 + t17 * t5) / 0.2e1;
    AScalar t19 = t11 * (t1 * t2 + t4 * t6) / 0.2e1;
    AScalar t20 = t11 * (t12 * t2 + t15 * t6) / 0.2e1;
    AScalar t21 = t11 * (t13 * t2 + t17 * t6) / 0.2e1;
    AScalar t22 = t11 * (t1 * t3 + t4 * t7) / 0.2e1;
    AScalar t23 = t11 * (t12 * t3 + t15 * t7) / 0.2e1;
    t11 = t11 * (t13 * t3 + t17 * t7) / 0.2e1;
    t17 = t17 * t8;
    t15 = t15 * t8;
    t4 = t4 * t8;
    t8 = t13 * t10;
    t12 = t12 * t10;
    t1 = t1 * t10;
    VectorXa dEdx(36);
    dEdx[0] = -t4 * t9;
    dEdx[1] = t14;
    dEdx[2] = t14;
    dEdx[3] = -t1 * t5;
    dEdx[4] = -t15 * t9;
    dEdx[5] = t16;
    dEdx[6] = t16;
    dEdx[7] = -t12 * t5;
    dEdx[8] = -t17 * t9;
    dEdx[9] = t18;
    dEdx[10] = t18;
    dEdx[11] = -t8 * t5;
    dEdx[12] = t4 * t2;
    dEdx[13] = -t19;
    dEdx[14] = -t19;
    dEdx[15] = t1 * t6;
    dEdx[16] = t15 * t2;
    dEdx[17] = -t20;
    dEdx[18] = -t20;
    dEdx[19] = t12 * t6;
    dEdx[20] = t17 * t2;
    dEdx[21] = -t21;
    dEdx[22] = -t21;
    dEdx[23] = t8 * t6;
    dEdx[24] = t4 * t3;
    dEdx[25] = -t22;
    dEdx[26] = -t22;
    dEdx[27] = t1 * t7;
    dEdx[28] = t15 * t3;
    dEdx[29] = -t23;
    dEdx[30] = -t23;
    dEdx[31] = t12 * t7;
    dEdx[32] = t17 * t3;
    dEdx[33] = -t11;
    dEdx[34] = -t11;
    dEdx[35] = t8 * t7;

    Eigen::Matrix<AScalar, 4, 9> res;
    for(int i = 0; i < 9; ++i){
        res.col(i) = dEdx.segment(4*i, 4);
    }

    return res;
}

#endif