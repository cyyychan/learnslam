#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2, 
std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches);

void pose_estimation_2d2d(const std::vector<cv::KeyPoint>& keypoints_1, const std::vector<cv::KeyPoint>& keypoints_2,
const std::vector<cv::DMatch>& matches, Mat& R, Mat& t);

void triangulation(const std::vector<cv::KeyPoint>& keypoints_1, const std::vector<cv::KeyPoint>& keypoints_2,
const std::vector<cv::DMatch>& matches, const cv::Mat& R, const cv::Mat& t, std::vector<cv::Point3d>& points);

cv::Point2f pixel2cam(const cv::Point2d& p, const cv::Mat& K);

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cout << "usage: triangulation img1 img2" << endl;
        return 1;
    }
    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "found " << matches.size() << "matches" << endl;

    cv::Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    std::vector<cv::Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for(int i = 0; i < matches.size(); i++)
    {
        cv::Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::Point2d pt1_cam_3d(points[i].x/points[i].z, points[i].y/points[i].z);
        cout << "point in the first camera frame: " << pt1_cam << endl;
        cout << "point projected from 3D " << pt1_cam_3d;

        cv::Point2d pt2_cam = pixel2cam(keypoints_2[matches[i].queryIdx].pt, K);
        cv::Mat pt2_trans = R*(cv::Mat_<double>(3, 1)<<points[i].x, points[i].y, points[i].z) + t;
        pt2_trans /= pt2_trans.at<double>(2, 0);
        cout << "point in the second camera frame: " << pt2_cam << endl;
        cout << "point reprojected from second frame " << pt2_trans.t() << endl;
    }
}


void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3 
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2 
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
   // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

void pose_estimation_2d2d(const std::vector<cv::KeyPoint>& keypoints_1, const std::vector<cv::KeyPoint>& keypoints_2,
const std::vector<cv::DMatch>& matches, cv::Mat& R, cv::Mat& t)
{
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for(int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    cout << "fundamental matrix" << endl << fundamental_matrix << endl;

    cv::Point2d principal_point(325.1, 249.7);
    int focal_length = 521;
    cv::Mat essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential matrix" << endl << essential_matrix << endl;

    cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC, 3);
    cout << "homography matrix" << endl << homography << endl;

    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}

void triangulation(const std::vector<cv::KeyPoint>& keypoint_1, const std::vector<cv::KeyPoint>& keypoint_2,
const std::vector<cv::DMatch>& matches, const cv::Mat& R, const cv::Mat& t, std::vector<cv::Point3d>& points)
{
    cv::Mat T1 = (cv::Mat_<float>(3, 4)<<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<float>(3, 4)<<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    std::vector<cv::Point2f> pts_1, pts_2;
    for(cv::DMatch m:matches)
    {
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }
    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for(int i = 0; i < pts_4d.cols; i++)
    {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.push_back(p);
    }
}


cv::Point2f pixel2cam(const cv::Point2d& p, const cv::Mat& K)
{
    return cv::Point2f
    (
        (p.x - K.at<double>(0, 2) / K.at<double>(0, 0),
        p.y - K.at<double>(1, 2) / K.at<double>(1, 1))
    );
}