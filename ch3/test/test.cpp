#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>


using namespace std;


int main(int argc, char**argv)
{
    Eigen::Quaterniond q1 = Eigen::Quaterniond(0.35, 0.2, 0.3, 0.1);
    Eigen::Vector3d t1 = Eigen::Vector3d(0.3, 0.1, 0.1);

    Eigen::Matrix3d rotation_matrix_1 = q1.toRotationMatrix();
    Eigen::Isometry3d Twq1 = Eigen::Isometry3d::Identity();
    Twq1.rotate(rotation_matrix_1);
    Twq1.pretranslate(t1);
    cout << "Transform matrix = \n" << Twq1.matrix() << endl;
    Eigen::Matrix4d Twq1_matrix = Twq1.matrix();

    Eigen::Quaterniond q2 = Eigen::Quaterniond(-0.5, 0.4, -0.1, 0.2);
    Eigen::Vector3d t2 = Eigen::Vector3d(-0.1, 0.5, 0.3);

    Eigen::Matrix3d rotation_matrix_2 = q2.toRotationMatrix();
    Eigen::Isometry3d Twq2 = Eigen::Isometry3d::Identity();
    Twq2.rotate(rotation_matrix_2);
    Twq2.pretranslate(t2);
    cout << "Transform matrix = \n" << Twq2.matrix() << endl;
    Eigen::Matrix4d Twq2_matrix = Twq2.matrix();

    Eigen::Vector4d pq1 = Eigen::Vector4d(0.5, 0, 0.2, 1);
    Eigen::Matrix4d Tq1q2 = Twq2_matrix * (Twq1_matrix.inverse());
    Eigen::Vector4d pq2 = Tq1q2 * pq1;
    cout << "position: " << pq2.block(0, 0, 3, 1) << endl;
    return 0;
}