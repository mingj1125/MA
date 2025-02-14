#include "Quadratic2DShell.h"
#include <vector>
#include <cmath>

// Define the quadrature points and weights for a triangle
struct QuadraturePoint {
    double xi, eta, weight;
};

std::vector<QuadraturePoint> get_6_point_gaussian_quadrature() {
    return {
        {0.0915762135, 0.0915762135, 0.1099517437},
        {0.8168475730, 0.0915762135, 0.1099517437},
        {0.0915762135, 0.8168475730, 0.1099517437},
        {0.4459484909, 0.1081030182, 0.2233815897},
        {0.1081030182, 0.4459484909, 0.2233815897},
        {0.4459484909, 0.4459484909, 0.2233815897}
    };
}

std::vector<Vector<T, 2>> get_shape_function_nodes(){
    return {
        {0., 0.},
        {1., 0.},
        {0., 1.},
        {0.5, 0.},
        {0., 0.5},
        {0.5, 0.5}
    };
}

Matrix<T, 2, 6> compute2DdNdX(const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, 
        const Matrix<T,3,1> & x3Undef, const Vector<T, 2> beta){

    T p[6];
    p[0] = x1Undef(0,0); p[1] = x1Undef(1,0);
    p[2] = x2Undef(0,0); p[3] = x2Undef(1,0);
    p[4] = x3Undef(0,0); p[5] = x3Undef(1,0);
    T t1 = p[0] + p[2];
    T t2 = 0.1e1 - beta[0] - beta[1];
    T t3 = p[0] + p[4];
    T t4 = p[2] + p[4];
    T t5 = beta[1] + beta[0];
    T t6 = 0.2e1;
    T t7 = -t5 * t6 + 0.1e1;
    T t8 = t6 * beta[0] - 0.1e1;
    T t9 = t7 * p[0];
    T t10 = t6 * (t2 * (-p[0] + t1) + (p[2] - t1) * beta[0] + (-t3 + t4) * beta[1]) + t8 * p[2] - t9;
    T t11 = p[1] + p[3];
    T t12 = p[1] + p[5];
    T t13 = p[3] + p[5];
    T t14 = t6 * beta[1] - 0.1e1;
    t7 = t7 * p[1];
    T t15 = -t6 * (t2 * (p[1] - t12) + (t11 - t13) * beta[0] + (-p[5] + t12) * beta[1]) + t14 * p[5] - t7;
    t1 = t6 * (t2 * (-p[0] + t3) + (-t1 + t4) * beta[0] + (p[4] - t3) * beta[1]) + t14 * p[4] - t9;
    t2 = t6 * (t2 * (-p[1] + t11) + (p[3] - t11) * beta[0] + (-t12 + t13) * beta[1]) + t8 * p[3] - t7;
    t3 = t1 * t2 - t10 * t15;
    t2 = -t2;
    t3 = 0.1e1 / t3;
    t4 = t3 * (0.4e1 * t5 - 0.3e1);
    t5 = 0.4e1 * beta[0];
    t6 = -0.1e1 + t5;
    t7 = 0.4e1 * beta[1];
    t8 = -0.1e1 + t7;
    t9 = -0.8e1 * beta[0] + 0.4e1 - 0.4e1 * beta[1];
    t11 = 0.4e1 - 0.4e1 * beta[0] - 0.8e1 * beta[1];
    t1 = -t1;
    Matrix<T, 2, 6> dNdX;
    dNdX(0,0) = -t4 * (t15 + t2);
    dNdX(0,1) = -t3 * t15 * t6;
    dNdX(0,2) = -t3 * t2 * t8;
    dNdX(0,3) = t3 * (-t15 * t9 + t5 * t2);
    dNdX(0,4) = t3 * (-t11 * t2 + t7 * t15);
    dNdX(0,5) = -0.4e1 * t3 * (t15 * beta[1] + t2 * beta[0]);
    dNdX(1,0) = -t4 * (t10 + t1);
    dNdX(1,1) = -t3 * t1 * t6;
    dNdX(1,2) = -t3 * t10 * t8;
    dNdX(1,3) = t3 * (-t1 * t9 + t5 * t10);
    dNdX(1,4) = t3 * (t7 * t1 - t10 * t11);
    dNdX(1,5) = -0.4e1 * t3 * (t1 * beta[1] + t10 * beta[0]);

    return dNdX;
}

Matrix<T, 2, 6> compute2DdNdx(const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, 
        const Matrix<T,3,1> & x3, const Vector<T, 2> beta){
    T q[6];
    q[0] = x1(0,0); q[1] = x1(1,0);
    q[2] = x2(0,0); q[3] = x2(1,0);
    q[4] = x3(0,0); q[5] = x3(1,0);        
    T t1 = beta[1] + beta[0];
    T t2 = 0.2e1;
    T t3 = -t1 * t2 + 0.1e1;
    T t4 = 0.1e1 - beta[0] - beta[1];
    T t5 = t2 * beta[0] - 0.1e1;
    T t6 = q[0] + q[2];
    T t7 = q[0] + q[4];
    T t8 = q[2] + q[4];
    T t9 = t3 * q[0];
    T t10 = t2 * (t4 * (-q[0] + t6) + (q[2] - t6) * beta[0] + (-t7 + t8) * beta[1]) + t5 * q[2] - t9;
    T t11 = t2 * beta[1] - 0.1e1;
    T t12 = q[1] + q[3];
    T t13 = q[1] + q[5];
    T t14 = q[3] + q[5];
    t3 = t3 * q[1];
    T t15 = t11 * q[5] - t2 * (t4 * (q[1] - t13) + (t12 - t14) * beta[0] + (-q[5] + t13) * beta[1]) - t3;
    t6 = t2 * (t4 * (-q[0] + t7) + (-t6 + t8) * beta[0] + (q[4] - t7) * beta[1]) + t11 * q[4] - t9;
    t2 = -t2 * (t4 * (q[1] - t12) + (-q[3] + t12) * beta[0] + (t13 - t14) * beta[1]) + t5 * q[3] - t3;
    t3 = t10 * t15 - t2 * t6;
    t2 = -t2;
    t3 = 0.1e1 / t3;
    t1 = t3 * (0.4e1 * t1 - 0.3e1);
    t4 = 0.4e1 * beta[0];
    t5 = -0.1e1 + t4;
    t7 = 0.4e1 * beta[1];
    t8 = -0.1e1 + t7;
    t9 = -0.8e1 * beta[0] + 0.4e1 - 0.4e1 * beta[1];
    t11 = 0.4e1 - 0.4e1 * beta[0] - 0.8e1 * beta[1];
    t6 = -t6;
    Matrix<T, 2, 6> dNdx;
    dNdx(0,0) = t1 * (t15 + t2);
    dNdx(0,1) = t3 * t15 * t5;
    dNdx(0,2) = t3 * t2 * t8;
    dNdx(0,3) = t3 * (t15 * t9 - t4 * t2);
    dNdx(0,4) = t3 * (t11 * t2 - t7 * t15);
    dNdx(0,5) = 0.4e1 * t3 * (t15 * beta[1] + t2 * beta[0]);
    dNdx(1,0) = t1 * (t10 + t6);
    dNdx(1,1) = t3 * t6 * t5;
    dNdx(1,2) = t3 * t10 * t8;
    dNdx(1,3) = t3 * (-t4 * t10 + t6 * t9);
    dNdx(1,4) = t3 * (t10 * t11 - t7 * t6);
    dNdx(1,5) = 0.4e1 * t3 * (t10 * beta[0] + t6 * beta[1]);

    return dNdx;
}

Matrix<T, 2, 2> compute2DDeformationGradient(const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, const Vector<T, 2> beta){
        
        Matrix<T, 2, 6> x;
        x << x1, x2, x3, 0.5*(x1+x2), 0.5*(x1+x3), 0.5*(x2+x3);
        return x*(compute2DdNdX(x1Undef, x2Undef, x3Undef, beta).transpose());
}

T computePointEnergyDensity(T lambda, T mu, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, const Vector<T, 2> beta){
        
        // Matrix<T, 2, 2> F = compute2DDeformationGradient(x1, x2, x3, x1Undef, x2Undef, x3Undef, beta);
        // Matrix<T, 2, 2> green_strain = 0.5*(F.transpose()*F-Matrix<T, 2, 2>::Identity());
        // T energy_density = 0.5*lambda*green_strain.trace()*green_strain.trace() + mu*green_strain.squaredNorm();
        T q[6];
        q[0] = x1(0,0); q[1] = x1(1,0);
        q[2] = x2(0,0); q[3] = x2(1,0);
        q[4] = x3(0,0); q[5] = x3(1,0);   
        T p[6];
        p[0] = x1Undef(0,0); p[1] = x1Undef(1,0);
        p[2] = x2Undef(0,0); p[3] = x2Undef(1,0);
        p[4] = x3Undef(0,0); p[5] = x3Undef(1,0);
        T t1 = q[0];
        T t2 = beta[0];
        T t3 = p[0];
        T t4 = 0.5e0 * t3;
        T t5 = p[2];
        T t6 = 0.5e0 * t5;
        T t7 = t4 + t6;
        T t9 = 0.4e1 * t7 * t2;
        T t10 = beta[1];
        T t11 = 0.1e1 - t2 - t10;
        T t14 = p[4];
        T t15 = 0.5e0 * t14;
        T t16 = t4 + t15;
        T t18 = 0.4e1 * t16 * t10;
        T t19 = t6 + t15;
        T t22 = 0.2e1 * t2;
        T t23 = 0.2e1 * t10;
        T t24 = 0.1e1 - t22 - t23;
        T t25 = t3 * t24;
        T t27 = 0.2e1 * t3 * t11;
        T t28 = t22 - 0.1e1;
        T t32 = 0.4e1 * t19 * t10 + 0.4e1 * t7 * t11 + 0.2e1 * t5 * t2 + t5 * t28 - t18 - t25 - t27 - t9;
        T t33 = t23 - 0.1e1;
        T t34 = p[5];
        T t38 = p[1];
        T t39 = 0.5e0 * t38;
        T t40 = p[3];
        T t41 = 0.5e0 * t40;
        T t42 = t39 + t41;
        T t44 = 0.4e1 * t42 * t2;
        T t45 = 0.5e0 * t34;
        T t46 = t39 + t45;
        T t50 = 0.4e1 * t46 * t10;
        T t51 = t41 + t45;
        T t54 = t38 * t24;
        T t56 = 0.2e1 * t38 * t11;
        T t57 = 0.2e1 * t34 * t10 + 0.4e1 * t46 * t11 + 0.4e1 * t51 * t2 + t34 * t33 - t44 - t50 - t54 - t56;
        T t66 = 0.2e1 * t14 * t10 + 0.4e1 * t16 * t11 + t14 * t33 + 0.4e1 * t19 * t2 - t18 - t25 - t27 - t9;
        T t74 = 0.4e1 * t51 * t10 + 0.4e1 * t42 * t11 + 0.2e1 * t40 * t2 + t40 * t28 - t44 - t50 - t54 - t56;
        T t77 = 0.1e1 / (t57 * t32 - t74 * t66);
        T t78 = t57 * t77;
        T t79 = 0.4e1 * t2;
        T t80 = 0.4e1 * t10;
        T t81 = -0.3e1 + t79 + t80;
        T t83 = -t74;
        T t84 = t83 * t77;
        T t86 = t81 * t78 + t81 * t84;
        T t88 = q[2];
        T t89 = t77 * t88;
        T t90 = t79 - 0.1e1;
        T t91 = t90 * t57;
        T t93 = q[4];
        T t94 = t77 * t93;
        T t95 = t80 - 0.1e1;
        T t96 = t95 * t83;
        T t98 = 0.5e0 * t1;
        T t99 = 0.5e0 * t88;
        T t100 = t98 + t99;
        T t102 = -0.8e1 * t2 + 0.4e1 - t80;
        T t104 = t2 * t84;
        T t106 = t102 * t78 - 0.4e1 * t104;
        T t108 = 0.5e0 * t93;
        T t109 = t98 + t108;
        T t110 = t10 * t78;
        T t113 = 0.4e1 - t79 - 0.8e1 * t10;
        T t115 = t113 * t84 - 0.4e1 * t110;
        T t117 = t99 + t108;
        T t119 = 0.4e1 * t110 + 0.4e1 * t104;
        T t121 = t86 * t1 + t106 * t100 + t115 * t109 + t119 * t117 + t91 * t89 + t96 * t94;
        T t122 = t121 * t121;
        T t123 = q[1];
        T t125 = q[3];
        T t126 = t77 * t125;
        T t128 = q[5];
        T t129 = t77 * t128;
        T t131 = 0.5e0 * t123;
        T t132 = 0.5e0 * t125;
        T t133 = t131 + t132;
        T t135 = 0.5e0 * t128;
        T t136 = t131 + t135;
        T t138 = t132 + t135;
        T t140 = t106 * t133 + t115 * t136 + t119 * t138 + t86 * t123 + t91 * t126 + t96 * t129;
        T t141 = t140 * t140;
        T t142 = -t66;
        T t143 = t142 * t77;
        T t145 = t32 * t77;
        T t147 = t81 * t143 + t81 * t145;
        T t149 = t90 * t142;
        T t151 = t95 * t32;
        T t154 = t2 * t145;
        T t156 = t102 * t143 - 0.4e1 * t154;
        T t158 = t10 * t143;
        T t161 = t113 * t145 - 0.4e1 * t158;
        T t164 = 0.4e1 * t158 + 0.4e1 * t154;
        T t166 = t147 * t1 + t156 * t100 + t161 * t109 + t164 * t117 + t149 * t89 + t151 * t94;
        T t167 = t166 * t166;
        T t174 = t147 * t123 + t149 * t126 + t151 * t129 + t156 * t133 + t161 * t136 + t164 * t138;
        T t175 = t174 * t174;
        T t177 = pow(t122 + t141 - 0.2e1 + t167 + t175, 0.2e1);
        T t181 = pow(t122 + t141 - 0.1e1, 0.2e1);
        T t185 = pow(t166 * t121 + t174 * t140, 0.2e1);
        T t188 = pow(t167 + t175 - 0.1e1, 0.2e1);
        T energy_density = 0.5e0 * t177 * lambda + (t181 + 0.2e1 * t185 + t188) * mu;
        return energy_density;
}

T compute2DQuadraticShellEnergy(T lambda, T mu, T thickness, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef){
        
        T area = ((x2Undef-x1Undef).cross(x3Undef-x1Undef)).norm();
        std::vector<QuadraturePoint> quadrature_points = get_6_point_gaussian_quadrature();
        double energy = 0.0;
        for(const auto& point: quadrature_points){
            energy += computePointEnergyDensity(lambda, mu, x1, x2, x3, x1Undef, x2Undef, x3Undef, {point.xi, point.eta}) * point.weight;
        }

        energy *= area*thickness;
        return energy;
}

Vector<T, 9> computePointEnergyDensityGradient(T lambda, T mu, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, const Vector<T, 2> beta){
        
        T q[6];
        q[0] = x1(0,0); q[1] = x1(1,0);
        q[2] = x2(0,0); q[3] = x2(1,0);
        q[4] = x3(0,0); q[5] = x3(1,0);   
        T p[6];
        p[0] = x1Undef(0,0); p[1] = x1Undef(1,0);
        p[2] = x2Undef(0,0); p[3] = x2Undef(1,0);
        p[4] = x3Undef(0,0); p[5] = x3Undef(1,0);
        T t1 = p[0] + p[2];
        T t2 = 0.1e1 - beta[0] - beta[1];
        T t3 = p[0] + p[4];
        T t4 = p[2] + p[4];
        T t5 = beta[0] + beta[1];
        T t6 = 0.2e1;
        T t7 = -t5 * t6 + 0.1e1;
        T t8 = t6 * beta[0];
        T t9 = -0.1e1 + t8;
        T t10 = t7 * p[0];
        T t11 = -t6 * (t2 * (p[0] - t1) + (-p[2] + t1) * beta[0] + (t3 - t4) * beta[1]) + t9 * p[2] - t10;
        T t12 = t6 * beta[1];
        T t13 = -0.1e1 + t12;
        T t14 = p[1] + p[3];
        T t15 = p[1] + p[5];
        T t16 = p[3] + p[5];
        t7 = t7 * p[1];
        T t17 = t13 * p[5] - t6 * (t2 * (p[1] - t15) + (t14 - t16) * beta[0] + (-p[5] + t15) * beta[1]) - t7;
        t1 = -t6 * (t2 * (p[0] - t3) + (t1 - t4) * beta[0] + (-p[4] + t3) * beta[1]) + t13 * p[4] - t10;
        t2 = -t6 * (t2 * (p[1] - t14) + (-p[3] + t14) * beta[0] + (t15 - t16) * beta[1]) + t9 * p[3] - t7;
        t3 = -t1 * t2 + t11 * t17;
        t2 = -t2;
        t3 = 0.1e1 / t3;
        t4 = t3 * (0.4e1 * t5 - 0.3e1);
        t5 = t4 * (t17 + t2);
        t7 = 0.4e1 * beta[0];
        t9 = -0.1e1 + t7;
        t10 = 0.4e1 * beta[1];
        t13 = -0.1e1 + t10;
        t14 = q[0] + q[2];
        t15 = -0.8e1 * beta[0] + 0.4e1 - 0.4e1 * beta[1];
        t16 = t17 * t15;
        T t18 = t3 * (-t7 * t2 + t16);
        T t19 = q[0] + q[4];
        T t20 = 0.4e1 - 0.4e1 * beta[0] - 0.8e1 * beta[1];
        T t21 = t2 * t20;
        T t22 = t3 * (-t10 * t17 + t21);
        T t23 = q[2] + q[4];
        T t24 = 0.1e1 / 0.2e1;
        T t25 = t6 * t3 * (t17 * beta[1] + t2 * beta[0]);
        T t26 = t24 * (t14 * t18 + t19 * t22) + t3 * (q[4] * t2 * t13 + q[2] * t17 * t9) + t5 * q[0] + t25 * t23;
        T t27 = q[1] + q[3];
        T t28 = q[1] + q[5];
        T t29 = q[3] + q[5];
        t18 = t24 * (t18 * t27 + t22 * t28) + t3 * (q[5] * t2 * t13 + q[3] * t17 * t9) + t5 * q[1] + t25 * t29;
        t1 = -t1;
        t4 = t4 * (t11 + t1);
        t22 = t1 * t15;
        t7 = t3 * (-t7 * t11 + t22);
        T t30 = t11 * t20;
        t10 = t3 * (-t10 * t1 + t30);
        T t31 = t6 * t3 * (t1 * beta[1] + t11 * beta[0]);
        t14 = t24 * (t10 * t19 + t14 * t7) + t3 * (q[2] * t9 * t1 + q[4] * t11 * t13) + t4 * q[0] + t31 * t23;
        t7 = t24 * (t10 * t28 + t27 * t7) + t3 * (q[3] * t9 * t1 + q[5] * t11 * t13) + t4 * q[1] + t31 * t29;
        t10 = std::pow(t14, 0.2e1);
        t19 = std::pow(t26, 0.2e1);
        t23 = std::pow(t18, 0.2e1);
        t27 = std::pow(t7, 0.2e1);
        t5 = t24 * t3 * (t16 + t21) - t25 + t5;
        t4 = t24 * t3 * (t30 + t22) - t31 + t4;
        t16 = t14 * t4;
        t21 = t26 * t5;
        t22 = 0.1e1 - t19 - t23;
        t25 = t14 * t26 + t18 * t7;
        t28 = 0.1e1 - t10 - t27;
        t6 = t6 * lambda * (-t6 + t10 + t19 + t23 + t27);
        t10 = 0.4e1 * mu;
        t19 = t7 * t4;
        t23 = t18 * t5;
        t9 = t15 * t24 + t12 + t9;
        t12 = t3 * t17 * t9;
        t1 = t3 * t1 * t9;
        t9 = t14 * t1;
        t15 = t26 * t12;
        t17 = t7 * t1;
        t27 = t18 * t12;
        t8 = t20 * t24 + t13 + t8;
        t2 = t3 * t2 * t8;
        t3 = t3 * t11 * t8;
        t8 = t14 * t3;
        t11 = t26 * t2;
        t13 = t7 * t3;
        t20 = t18 * t2;
        Vector<T, 9> gradient;
        gradient(0) = t6 * (t16 + t21) - t10 * (-t25 * (t14 * t5 + t26 * t4) + t16 * t28 + t21 * t22);
        gradient(1) = t6 * (t19 + t23) - t10 * (-t25 * (t18 * t4 + t5 * t7) + t19 * t28 + t23 * t22);
        gradient(2) = 0;
        gradient(3) = t6 * (t9 + t15) - t10 * (-t25 * (t1 * t26 + t12 * t14) + t9 * t28 + t15 * t22);
        gradient(4) = t6 * (t27 + t17) - t10 * (-t25 * (t1 * t18 + t12 * t7) + t17 * t28 + t27 * t22);
        gradient(5) = 0;
        gradient(6) = t6 * (t8 + t11) - t10 * (-t25 * (t14 * t2 + t26 * t3) + t8 * t28 + t11 * t22);
        gradient(7) = t6 * (t20 + t13) - t10 * (-t25 * (t18 * t3 + t7 * t2) + t13 * t28 + t20 * t22);
        gradient(8) = 0;

        return gradient;
}

Vector<T, 9> compute2DQuadraticShellEnergyGradient(T lambda, T mu, T thickness, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef){
        
        Vector<T, 9> gradient; gradient.setZero();
        T area = ((x2Undef-x1Undef).cross(x3Undef-x1Undef)).norm();
        std::vector<QuadraturePoint> quadrature_points = get_6_point_gaussian_quadrature();
        
        for(const auto& point: quadrature_points){
                gradient += computePointEnergyDensityGradient(lambda, mu, x1, x2, x3, x1Undef, x2Undef, x3Undef, {point.xi, point.eta}) * point.weight;
        }

        return gradient*area*thickness;
}

Matrix<T, 9, 9> computePointEnergyDensityHessian(T lambda, T mu, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, const Vector<T, 2> beta){
        
        T q[6];
        q[0] = x1(0,0); q[1] = x1(1,0);
        q[2] = x2(0,0); q[3] = x2(1,0);
        q[4] = x3(0,0); q[5] = x3(1,0);   
        T p[6];
        p[0] = x1Undef(0,0); p[1] = x1Undef(1,0);
        p[2] = x2Undef(0,0); p[3] = x2Undef(1,0);
        p[4] = x3Undef(0,0); p[5] = x3Undef(1,0);
        T t1 = p[0] + p[2];
        T t2 = 0.1e1 - beta[0] - beta[1];
        T t3 = p[0] + p[4];
        T t4 = p[2] + p[4];
        T t5 = beta[0] + beta[1];
        T t6 = 0.2e1;
        T t7 = -t5 * t6 + 0.1e1;
        T t8 = t6 * beta[0];
        T t9 = -0.1e1 + t8;
        T t10 = t7 * p[0];
        T t11 = -t6 * (t2 * (p[0] - t1) + (-p[2] + t1) * beta[0] + (t3 - t4) * beta[1]) + t9 * p[2] - t10;
        T t12 = t6 * beta[1];
        T t13 = -0.1e1 + t12;
        T t14 = p[1] + p[3];
        T t15 = p[1] + p[5];
        T t16 = p[3] + p[5];
        t7 = t7 * p[1];
        T t17 = t13 * p[5] - t6 * (t2 * (p[1] - t15) + (t14 - t16) * beta[0] + (-p[5] + t15) * beta[1]) - t7;
        t1 = -t6 * (t2 * (p[0] - t3) + (t1 - t4) * beta[0] + (-p[4] + t3) * beta[1]) + t13 * p[4] - t10;
        t2 = -t6 * (t2 * (p[1] - t14) + (-p[3] + t14) * beta[0] + (t15 - t16) * beta[1]) + t9 * p[3] - t7;
        t3 = -t1 * t2 + t11 * t17;
        t2 = -t2;
        t3 = 0.1e1 / t3;
        t4 = t3 * (0.4e1 * t5 - 0.3e1);
        t5 = t4 * (t17 + t2);
        t7 = 0.4e1 * beta[0];
        t9 = -0.1e1 + t7;
        t10 = 0.4e1 * beta[1];
        t13 = -0.1e1 + t10;
        t14 = q[0] + q[2];
        t15 = -0.8e1 * beta[0] + 0.4e1 - 0.4e1 * beta[1];
        t16 = t17 * t15;
        T t18 = t3 * (-t7 * t2 + t16);
        T t19 = q[0] + q[4];
        T t20 = 0.4e1 - 0.4e1 * beta[0] - 0.8e1 * beta[1];
        T t21 = t2 * t20;
        T t22 = t3 * (-t10 * t17 + t21);
        T t23 = q[2] + q[4];
        T t24 = 0.1e1 / 0.2e1;
        T t25 = t6 * t3 * (t17 * beta[1] + t2 * beta[0]);
        T t26 = t24 * (t14 * t18 + t19 * t22) + t3 * (q[4] * t2 * t13 + q[2] * t17 * t9) + t5 * q[0] + t25 * t23;
        t16 = t24 * t3 * (t16 + t21) - t25 + t5;
        t1 = -t1;
        t4 = t4 * (t11 + t1);
        t21 = t1 * t15;
        t7 = t3 * (-t7 * t11 + t21);
        T t27 = t11 * t20;
        t10 = t3 * (-t10 * t1 + t27);
        T t28 = t6 * t3 * (t1 * beta[1] + t11 * beta[0]);
        t14 = t24 * (t10 * t19 + t14 * t7) + t3 * (q[2] * t9 * t1 + q[4] * t11 * t13) + t4 * q[0] + t28 * t23;
        t19 = t24 * t3 * (t21 + t27) - t28 + t4;
        t21 = t14 * t19 + t16 * t26;
        t23 = q[1] + q[3];
        t27 = q[1] + q[5];
        T t29 = q[3] + q[5];
        t5 = t24 * (t18 * t23 + t22 * t27) + t3 * (q[5] * t2 * t13 + q[3] * t17 * t9) + t5 * q[1] + t25 * t29;
        t4 = t24 * (t10 * t27 + t23 * t7) + t3 * (q[3] * t9 * t1 + q[5] * t11 * t13) + t4 * q[1] + t28 * t29;
        t7 = std::pow(t26, 0.2e1);
        t10 = std::pow(t14, 0.2e1);
        t18 = std::pow(t5, 0.2e1);
        t22 = std::pow(t4, 0.2e1);
        t23 = -t6 + t7 + t10 + t18 + t22;
        t25 = std::pow(t16, 0.2e1);
        t27 = std::pow(t19, 0.2e1);
        t28 = 0.1e1 - t7 - t18;
        t29 = t14 * t16 + t19 * t26;
        T t30 = t14 * t26 + t4 * t5;
        T t31 = 0.1e1 - t10 - t22;
        T t32 = t16 * t30;
        T t33 = t31 * t27;
        T t34 = t28 * t25;
        T t35 = t23 * (t25 + t27) * t6;
        T t36 = t5 * t16 + t19 * t4;
        T t37 = t16 * t4 + t19 * t5;
        T t38 = 0.4e1 * t37;
        T t39 = 0.4e1 * lambda;
        T t40 = t39 * t36;
        t27 = mu * (0.8e1 * t4 * t27 * t14 + 0.8e1 * t5 * t25 * t26 + t38 * t29) + t40 * t21;
        t9 = t15 * t24 + t12 + t9;
        t12 = t3 * t17 * t9;
        t1 = t3 * t1 * t9;
        t9 = t1 * t14 + t12 * t26;
        t15 = t12 * t16;
        t17 = t1 * t19;
        T t41 = t1 * t26 + t12 * t14;
        T t42 = t10 * t1;
        T t43 = t17 * t31;
        T t44 = t15 * t28;
        T t45 = t30 * (t1 * t16 + t12 * t19);
        T t46 = t23 * (t15 + t17) * t6;
        T t47 = (0.4e1 * t21 * t9 + t46) * lambda + mu * (0.8e1 * t15 * t7 + 0.8e1 * t42 * t19 + 0.4e1 * t29 * t41 - 0.4e1 * t43 - 0.4e1 * t44 + 0.4e1 * t45);
        T t48 = t1 * t4 + t12 * t5;
        T t49 = t1 * t5 + t12 * t4;
        T t50 = 0.8e1 * t17 * t4 * t14 + 0.8e1 * t15 * t5 * t26;
        T t51 = 0.4e1 * t49;
        T t52 = t39 * t48;
        T t53 = mu * (t51 * t29 + t50) + t52 * t21;
        t8 = t20 * t24 + t13 + t8;
        t2 = t3 * t2 * t8;
        t3 = t3 * t11 * t8;
        t8 = t14 * t3 + t2 * t26;
        t11 = t2 * t16;
        t13 = t3 * t19;
        t20 = t14 * t2 + t26 * t3;
        t24 = t13 * t31;
        T t54 = t11 * t28;
        t16 = t30 * (t16 * t3 + t19 * t2);
        T t55 = t23 * (t13 + t11) * t6;
        T t56 = (0.4e1 * t21 * t8 + t55) * lambda + mu * (0.8e1 * t13 * t10 + 0.8e1 * t7 * t11 + 0.4e1 * t20 * t29 + 0.4e1 * t16 - 0.4e1 * t24 - 0.4e1 * t54);
        T t57 = t2 * t5 + t3 * t4;
        T t58 = t2 * t4 + t3 * t5;
        T t59 = 0.8e1 * t11 * t5 * t26 + 0.8e1 * t13 * t4 * t14;
        T t60 = 0.4e1 * t58;
        t39 = t39 * t57;
        T t61 = mu * (t60 * t29 + t59) + t39 * t21;
        t50 = mu * (t38 * t41 + t50) + t40 * t9;
        t15 = (0.4e1 * t36 * t48 + t46) * lambda + mu * (0.8e1 * t15 * t18 + 0.8e1 * t17 * t22 + 0.4e1 * t37 * t49 - 0.4e1 * t43 - 0.4e1 * t44 + 0.4e1 * t45);
        t17 = mu * (t38 * t20 + t59) + t40 * t8;
        t11 = (0.4e1 * t36 * t57 + t55) * lambda + mu * (0.8e1 * t11 * t18 + 0.8e1 * t13 * t22 + 0.4e1 * t37 * t58 + 0.4e1 * t16 - 0.4e1 * t24 - 0.4e1 * t54);
        t13 = std::pow(t1, 0.2e1);
        t16 = std::pow(t12, 0.2e1);
        t24 = t30 * t12;
        t38 = t13 * t31;
        t40 = t28 * t16;
        t43 = t23 * (t13 + t16) * t6;
        t13 = mu * (0.8e1 * t13 * t4 * t14 + 0.8e1 * t5 * t16 * t26 + t51 * t41) + t52 * t9;
        t44 = t3 * t1;
        t45 = t2 * t12;
        t46 = t45 * t28;
        t54 = t44 * t31;
        t12 = t30 * (t1 * t2 + t12 * t3);
        t55 = t23 * (t44 + t45) * t6;
        t59 = (0.4e1 * t9 * t8 + t55) * lambda + mu * (0.4e1 * t20 * t41 + 0.8e1 * t42 * t3 + 0.8e1 * t45 * t7 + 0.4e1 * t12 - 0.4e1 * t46 - 0.4e1 * t54);
        T t62 = 0.8e1 * t44 * t4 * t14 + 0.8e1 * t45 * t5 * t26;
        T t63 = mu * (t60 * t41 + t62) + t39 * t9;
        t51 = mu * (t51 * t20 + t62) + t52 * t8;
        t12 = (0.4e1 * t48 * t57 + t55) * lambda + mu * (0.8e1 * t45 * t18 + 0.8e1 * t44 * t22 + 0.4e1 * t49 * t58 + 0.4e1 * t12 - 0.4e1 * t46 - 0.4e1 * t54);
        t44 = std::pow(t3, 0.2e1);
        t45 = std::pow(t2, 0.2e1);
        t31 = t44 * t31;
        t28 = t45 * t28;
        t2 = t30 * t2;
        t6 = t23 * (t44 + t45) * t6;
        t4 = mu * (0.8e1 * t44 * t4 * t14 + 0.8e1 * t45 * t5 * t26 + t60 * t20) + t39 * t8;
        T hessian [6][6];
        hessian[0][0] = (0.4e1 * std::pow(t21, 0.2e1) + t35) * lambda + mu * (0.8e1 * t19 * (t10 * t19 + t32) + 0.8e1 * t25 * t7 + 0.4e1 * std::pow(t29, 0.2e1) - 0.4e1 * t33 - 0.4e1 * t34);
        hessian[0][1] = t27;
        hessian[0][2] = t47;
        hessian[0][3] = t53;
        hessian[0][4] = t56;
        hessian[0][5] = t61;
        hessian[1][0] = t27;
        hessian[1][1] = (0.4e1 * std::pow(t36, 0.2e1) + t35) * lambda + mu * (0.4e1 * std::pow(t37, 0.2e1) - 0.4e1 * t33 - 0.4e1 * t34 + 0.8e1 * t18 * t25 + 0.8e1 * t19 * (t19 * t22 + t32));
        hessian[1][2] = t50;
        hessian[1][3] = t15;
        hessian[1][4] = t17;
        hessian[1][5] = t11;
        hessian[2][0] = t47;
        hessian[2][1] = t50;
        hessian[2][2] = (0.4e1 * std::pow(t9, 0.2e1) + t43) * lambda + mu * (0.8e1 * t1 * (t42 + t24) + 0.8e1 * t16 * t7 + 0.4e1 * std::pow(t41, 0.2e1) - 0.4e1 * t38 - 0.4e1 * t40);
        hessian[2][3] = t13;
        hessian[2][4] = t59;
        hessian[2][5] = t63;
        hessian[3][0] = t53;
        hessian[3][1] = t15;
        hessian[3][2] = t13;
        hessian[3][3] = (0.4e1 * std::pow(t48, 0.2e1) + t43) * lambda + mu * (0.8e1 * t1 * (t1 * t22 + t24) + 0.8e1 * t16 * t18 + 0.4e1 * std::pow(t49, 0.2e1) - 0.4e1 * t38 - 0.4e1 * t40);
        hessian[3][4] = t51;
        hessian[3][5] = t12;
        hessian[4][0] = t56;
        hessian[4][1] = t17;
        hessian[4][2] = t59;
        hessian[4][3] = t51;
        hessian[4][4] = (0.4e1 * std::pow(t8, 0.2e1) + t6) * lambda + mu * (0.4e1 * std::pow(t20, 0.2e1) - 0.4e1 * t28 - 0.4e1 * t31 + 0.8e1 * t3 * (t10 * t3 + t2) + 0.8e1 * t45 * t7);
        hessian[4][5] = t4;
        hessian[5][0] = t61;
        hessian[5][1] = t11;
        hessian[5][2] = t63;
        hessian[5][3] = t12;
        hessian[5][4] = t4;
        hessian[5][5] = (0.4e1 * std::pow(t57, 0.2e1) + t6) * lambda + mu * (0.4e1 * std::pow(t58, 0.2e1) - 0.4e1 * t28 - 0.4e1 * t31 + 0.8e1 * t45 * t18 + 0.8e1 * t3 * (t22 * t3 + t2));

        Matrix<T, 9, 9> energy_hessian;
        for(int i = 0; i < 3; ++i){
            for(int j = 0; j < 3; ++j){
                Matrix<T, 3, 3> block; block.setZero();
                block(0,0) = hessian[i*2][j*2];
                block(1,0) = hessian[i*2+1][j*2];
                block(0,1) = hessian[i*2][j*2+1];
                block(1,1) = hessian[i*2+1][j*2+1];

                energy_hessian.block<3,3>(i*3, 3*j) = block;
            }
        }
        return energy_hessian;
}

Matrix<T, 9, 9> compute2DQuadraticShellEnergyHessian(T lambda, T mu, T thickness, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef){
        
        Matrix<T, 9, 9> hessian; hessian.setZero();
        T area = ((x2Undef-x1Undef).cross(x3Undef-x1Undef)).norm();
        std::vector<QuadraturePoint> quadrature_points = get_6_point_gaussian_quadrature();
        
        for(const auto& point: quadrature_points){
                hessian += computePointEnergyDensityHessian(lambda, mu, x1, x2, x3, x1Undef, x2Undef, x3Undef, {point.xi, point.eta}) * point.weight;
        }

        return hessian*area*thickness;
}