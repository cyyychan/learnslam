#include <iostream>
using namespace std;
// Eigen
#include <Eigen/Core>
// Dense matrix computation
#include <Eigen/Dense>

#define MATRIX_SIZE 50

int main(int argc, char** argv)
{
	// Matrix 
	Eigen::Matrix<float, 2, 3> matrix_23;
	cout << matrix_23 << endl;

	// Vector3d
	Eigen::Vector3d v_3d;
	Eigen::Matrix<float, 3, 1> vd_3d;

	// Matrix3d
	Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();

	// Dynamic
	Eigen::MatrixXd matrix_x;
	Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > matrix_dynamic;

	// matrix computation
	matrix_33 = Eigen::Matrix3d::Random();
	cout << matrix_33 << endl;

	cout << matrix_33.transpose() << '\n' <<  endl;
	cout << matrix_33.sum() << '\n' << endl;
	cout << matrix_33.trace() << '\n' << endl;
	cout << 10*matrix_33 << '\n' << endl;
	cout << matrix_33.inverse() << '\n' << endl;
	cout << matrix_33.determinant() << '\n' << endl;

	Eigen::SelfAdjointEigenSolver< Eigen::Matrix3d > eigen_solver ( matrix_33.transpose() * matrix_33);
	cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
	cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

	// solve function
	Eigen::Matrix< double , MATRIX_SIZE, MATRIX_SIZE > matrix_NN;
	matrix_NN = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );
	Eigen::Matrix< double, MATRIX_SIZE, 1 > v_Nd;
	v_Nd = Eigen::MatrixXd::Random( MATRIX_SIZE, 1 );

	clock_t time_stt = clock();
	Eigen::Matrix< double, MATRIX_SIZE, 1 > x = matrix_NN.inverse() * v_Nd;
	cout << x << '\n' << endl;
	cout << "time use in normal inverse is: " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;

	time_stt = clock();
	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
	cout << x << '\n' << endl;
	cout << "time use in normal inverse is: " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;

	return 0; 
}