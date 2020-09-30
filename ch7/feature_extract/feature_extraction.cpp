#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }

    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // 检测Oriented FAST角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // 根据角点位置 计算BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    Mat outimg1;
    cv::drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB feature points", outimg1);

    // 使用Hamming距离对两幅图像的描述子进行匹配
    vector<cv::DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    // 匹配点筛选
    double min_dist=10000, max_dist=0;
    for(int i=0; i<descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    // 匹配点合法判断
    std::vector<cv::DMatch> good_matches;
    for(int i=0; i<descriptors_1.rows; i++)
    {
        if(matches[i].distance <= max(2*min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    // 绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    cv::drawMatches(img_2, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    cv::namedWindow("match pts", 0);
    cv::namedWindow("good match pts", 0);
    cv::imshow("match pts", img_match);
    cv::imshow("good match pts", img_goodmatch);
    cv::waitKey(0);
    return 0;

}