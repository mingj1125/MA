#ifndef LINEAR_SHELL_STRESS_DIFF_H
#define LINEAR_SHELL_STRESS_DIFF_H

#include "vector_manipulation.h"

Eigen::Matrix<AScalar, 4, 9> dSdx(Vector9a& q, Vector6a& p, AScalar h, AScalar lambda, AScalar mu){

    AScalar t1 = p[2] - p[0];
    AScalar t2 = -p[5] + p[1];
    AScalar t3 = -p[4] + p[0];
    AScalar t4 = p[3] - p[1];
    AScalar t5 = q[3] - q[6];
    AScalar t6 = -q[6] + q[0];
    AScalar t7 = -q[3] + q[0];
    AScalar t8 = t5 * p[1] - t6 * p[3] + t7 * p[5];
    AScalar t9 = p[2] - p[4];
    AScalar t10 = -t1 * p[5] - t3 * p[3] + t9 * p[1];
    AScalar t11 = p[3] - p[5];
    t5 = t5 * p[0] - t6 * p[2] + t7 * p[4];
    t6 = t11 * p[0] - t2 * p[2] - t4 * p[4];
    t7 = 0.1e1 / t10;
    t6 = 0.1e1 / t6;
    t10 = pow(t6, 0.2e1);
    AScalar t12 = pow(t7, 0.2e1);
    AScalar t13 = t8 * t12;
    AScalar t14 = t13 * t11;
    AScalar t15 = t5 * t10;
    AScalar t16 = t15 * t9;
    AScalar t17 = 0.1e1 / 0.2e1;
    AScalar t18 = t17 * lambda;
    AScalar t19 = t18 * (t14 + t16);
    t6 = t7 * t6;
    t7 = q[4] - q[7];
    AScalar t20 = -q[7] + q[1];
    AScalar t21 = -q[4] + q[1];
    AScalar t22 = t20 * p[3] - t21 * p[5] - t7 * p[1];
    t7 = -t20 * p[2] + t21 * p[4] + t7 * p[0];
    t20 = t22 * t12;
    t21 = t20 * t11;
    AScalar t23 = t7 * t10;
    AScalar t24 = t23 * t9;
    AScalar t25 = t18 * (t21 - t24);
    AScalar t26 = q[5] - q[8];
    AScalar t27 = -q[8] + q[2];
    AScalar t28 = -q[5] + q[2];
    AScalar t29 = t26 * p[1] - t27 * p[3] + t28 * p[5];
    t26 = t26 * p[0] - t27 * p[2] + t28 * p[4];
    t12 = t29 * t12;
    t27 = t12 * t11;
    t10 = t26 * t10;
    t28 = t10 * t9;
    AScalar t30 = t18 * (t28 + t27);
    AScalar t31 = t13 * t2;
    AScalar t32 = t15 * t3;
    AScalar t33 = t18 * (t31 + t32);
    AScalar t34 = t20 * t2;
    AScalar t35 = t23 * t3;
    AScalar t36 = t18 * (t34 - t35);
    AScalar t37 = t12 * t2;
    AScalar t38 = t10 * t3;
    AScalar t39 = t18 * (t37 + t38);
    t13 = t13 * t4;
    t15 = t15 * t1;
    AScalar t40 = t18 * (t13 + t15);
    t20 = t20 * t4;
    t23 = t23 * t1;
    AScalar t41 = t18 * (-t23 + t20);
    t12 = t12 * t4;
    t10 = t10 * t1;
    t18 = t18 * (t12 + t10);
    t17 = t17 * (t1 * t2 - t3 * t4);
    AScalar t42 = t17 * mu;
    AScalar t43 = t42 * t6 * (t11 * t5 + t8 * t9);
    AScalar t44 = t42 * t6 * (t11 * t7 - t22 * t9);
    t9 = t42 * t6 * (t11 * t26 + t29 * t9);
    t11 = t42 * t6 * (t2 * t5 + t3 * t8);
    AScalar t45 = t42 * t6 * (-t2 * t7 + t22 * t3);
    t2 = t42 * t6 * (t2 * t26 + t29 * t3);
    t3 = t42 * t6 * (t1 * t8 + t4 * t5);
    t5 = t42 * t6 * (t1 * t22 - t4 * t7);
    t1 = t42 * t6 * (t1 * t29 + t26 * t4);
    VectorXa dSdx(36);
    dSdx[0] = -t17 * (-0.2e1 * t14 * mu - t19);
    dSdx[1] = t43;
    dSdx[2] = t43;
    dSdx[3] = -t17 * (-0.2e1 * t16 * mu - t19);
    dSdx[4] = -t17 * (0.2e1 * t21 * mu + t25);
    dSdx[5] = t44;
    dSdx[6] = t44;
    dSdx[7] = -t17 * (-0.2e1 * t24 * mu + t25);
    dSdx[8] = -t17 * (-0.2e1 * t27 * mu - t30);
    dSdx[9] = t9;
    dSdx[10] = t9;
    dSdx[11] = -t17 * (-0.2e1 * t28 * mu - t30);
    dSdx[12] = -t17 * (0.2e1 * t31 * mu + t33);
    dSdx[13] = -t11;
    dSdx[14] = -t11;
    dSdx[15] = -t17 * (0.2e1 * t32 * mu + t33);
    dSdx[16] = -t17 * (-0.2e1 * t34 * mu - t36);
    dSdx[17] = t45;
    dSdx[18] = t45;
    dSdx[19] = -t17 * (0.2e1 * t35 * mu - t36);
    dSdx[20] = -t17 * (0.2e1 * t37 * mu + t39);
    dSdx[21] = -t2;
    dSdx[22] = -t2;
    dSdx[23] = -t17 * (0.2e1 * t38 * mu + t39);
    dSdx[24] = -t17 * (0.2e1 * t13 * mu + t40);
    dSdx[25] = -t3;
    dSdx[26] = -t3;
    dSdx[27] = -t17 * (0.2e1 * t15 * mu + t40);
    dSdx[28] = -t17 * (-0.2e1 * t20 * mu - t41);
    dSdx[29] = t5;
    dSdx[30] = t5;
    dSdx[31] = -t17 * (0.2e1 * t23 * mu - t41);
    dSdx[32] = -t17 * (0.2e1 * t12 * mu + t18);
    dSdx[33] = -t1;
    dSdx[34] = -t1;
    dSdx[35] = -t17 * (0.2e1 * t10 * mu + t18);

    Eigen::Matrix<AScalar, 4, 9> res;
    for(int i = 0; i < 9; ++i){
        res.col(i) = dSdx.segment(4*i, 4);
    }
    return res*h;
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