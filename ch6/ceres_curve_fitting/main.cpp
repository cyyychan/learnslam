#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

struct CURVE_FITTING_COST
{
    /* data */
    CURVE_FITTING_COST (double x, double y) : _x(x), _y(y) {}
    //残差计算
    template <typename T> 
    bool operator()(const T* abc, T* residual ) const
    {
        residual[0] = T(_y) - ceres::exp( abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); //y-exp(ax^2+bx+c)
        return true;
    }
    const double _x, _y;
};

int main(int argc, char** argv)
{
    double a = 1.0, b = 2.0, c = 1.0; // 真实参考值
    int N = 100; // 数据点
    cv::RNG rng; 
    double w_sigma = 1.0; // 噪声Sigma值 
    vector<double> x_data, y_data; 
    double abc[3] = {0, 0, 0}; // abc参数的估计值
    
    cout << "generate data" << endl;
    for(int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back(x);
        y_data.push_back(exp(a*x*x + b*x + c) + rng.gaussian(w_sigma));
        cout << x_data[i] << " " << y_data[i] << endl;
    }

    // 构建最小二乘问题
    ceres::Problem problem;
    for(int i=0; i<N; i++)
    {
        problem.AddResidualBlock
        (
            // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3> (new CURVE_FITTING_COST (x_data[i], y_data[i])),
            nullptr, // 核函数
            abc // 待估计数据
        ); 
    }

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary; //优化信息

    // 开始优化
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary); 
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "solve time cost = " << time_used.count() << "seconds. " << endl;

    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "estimated a,b,c = ";
    for (auto a:abc ) cout<<a<<" ";
    cout<<endl;

    return 0;
}
