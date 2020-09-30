#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;


void find_feature_matches(const Mat& img_1, const Mat& img_2, std::vector<KeyPoint>& keypoints_1, std::vector<KeyPoint>& keypoints_2, std::vector<DMatch>& matches);

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2, std::vector<DMatch> matches, cv::Mat K, Mat& R, Mat& t);

Point2d pixel2cam(const Point2d& p, const Mat& K);

int main (int argc, char** argv)
{
    if(argc != 3)
    {
        cout << "usage: pose_estimation img1 img2" << endl;
        return 1;
    }

    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    
    // find feature matches
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "find " << matches.size() << " feature matches" << endl;
    
    // estimate
    cv::Mat R, t;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, K, R, t);
    
    // verify E = t^R*scale
    cv::Mat t_x = (cv::Mat_<double>(3, 3) << 
                    0, -t.at<double>(2, 0), t.at<double>(1, 0),
                    t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                    -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    cout << "t^R=" << endl << t_x*R << endl;

    // verify pole constraints
    for(cv::DMatch m: matches)
    {
        cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y);
        cv::Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
}

void find_feature_matches(const Mat& img_1, const Mat& img_2, std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches)
{
    cv::Mat descriptors_1, descriptors_2;
    Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    double min_dist=10000, max_dist=0;
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    printf("--Max dist : %f \n", max_dist);
    printf("--Min dist : %f \n", min_dist);

    for(int i = 0; i < descriptors_1.rows; i++)
    {
        if(match[i].distance <= max(2*min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d& p, const Mat& K)
{
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 2)
    );
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2, std::vector<DMatch> matches, cv::Mat K, cv::Mat& R, cv::Mat& t)
{
    std::vector<Point2f> points1;
    std::vector<Point2f> points2;

    for(int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    cout << "foudamental_matrix is " << endl << fundamental_matrix << endl;

    cv::Point2d principal_point(K.at<double>(0, 2), K.at<double>(1, 2));
    double focal_length = K.at<double>(1, 1);
    cv::Mat essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    cv::Mat homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}