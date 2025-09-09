/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/image.hpp>
// #include <sensor_msgs/image_encodings.h>
#include "image_encodings.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
// #include <ros/package.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include <sensor_msgs/msg/compressed_image.hpp>
// #include "camodocal/camera_models/CameraFactory.h"
#include "parameters.h"
#define SKIP_FIRST_CNT 10
using namespace std;

queue<sensor_msgs::msg::Image::ConstPtr> image_buf;
queue<sensor_msgs::msg::CompressedImage::ConstPtr> image_buf_c;



queue<sensor_msgs::msg::PointCloud::ConstPtr> point_buf;
// queue<sensor_msgs::msg::PointCloud2::ConstPtr> point_buf2;
queue<nav_msgs::msg::Odometry::ConstPtr> pose_buf;
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index  = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;
int USE_Compressed;

camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match_img;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry_rect;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud, pub_margin_cloud;

void new_sequence()
{
    printf("new sequence\n");
    sequence++;
    printf("sequence cnt %d \n", sequence);
    if (sequence > 9)
    {
        ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
        // ROS_BREAK();
    }
    posegraph.posegraph_visualization->reset();
    posegraph.publish();
    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pose_buf.empty())
        pose_buf.pop();
    while(!odometry_buf.empty())
        odometry_buf.pop();
    m_buf.unlock();
}


// void new_sequence_c()
// {
//     printf("new sequence\n");
//     sequence++;
//     printf("sequence cnt %d \n", sequence);
//     if (sequence > 5)
//     {
//         ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
//         // ROS_BREAK();
//     }
//     posegraph.posegraph_visualization->reset();
//     posegraph.publish();
//     m_buf.lock();
//     while(!image_buf_c.empty())
//         image_buf_c.pop();
//     while(!point_buf.empty())
//         point_buf.pop();
//     while(!pose_buf.empty())
//         pose_buf.pop();
//     while(!odometry_buf.empty())
//         odometry_buf.pop();
//     m_buf.unlock();
// }


void image_callback(const sensor_msgs::msg::Image::SharedPtr image_msg)
{
    // ROS_INFO("image_callback!");
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
    // printf(" image time %f \n", image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * (1e-9));

    // detect unstable camera stream
    if (last_image_time == -1)
        last_image_time = image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * (1e-9);
    else if (image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * (1e-9) - last_image_time > 1.0 
                    || image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * (1e-9) < last_image_time)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * (1e-9);
}



#include <memory>  // std::make_shared

void compressed_image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr cmsg)
{
    try
    {
        // 포맷 힌트 출력(디버그)
        RCLCPP_DEBUG(rclcpp::get_logger("loop_fusion"),
                     "Compressed format: %s", cmsg->format.c_str());

        sensor_msgs::msg::Image::SharedPtr image_msg;

        // 1) MONO8로 시도
        try {
            auto cv_ptr = cv_bridge::toCvCopy(cmsg, sensor_msgs::image_encodings::MONO8);
            image_msg = cv_ptr->toImageMsg();   // 이미 SharedPtr 반환
        } catch (const cv_bridge::Exception& e1) {
            // 2) 실패 시 BGR8로 디코딩 → GRAY 변환
            RCLCPP_WARN(rclcpp::get_logger("loop_fusion"),
                        "MONO8 decode failed, retry BGR8 then convert: %s", e1.what());

            auto cv_ptr_bgr = cv_bridge::toCvCopy(cmsg, sensor_msgs::image_encodings::BGR8);

            cv::Mat gray;
            cv::cvtColor(cv_ptr_bgr->image, gray, cv::COLOR_BGR2GRAY);

            // std::make_shared 사용 (boost 아님!)
            auto cv_ptr_gray = std::make_shared<cv_bridge::CvImage>();
            cv_ptr_gray->header   = cmsg->header;
            cv_ptr_gray->encoding = sensor_msgs::image_encodings::MONO8;
            cv_ptr_gray->image    = gray;

            image_msg = cv_ptr_gray->toImageMsg();
        }

        // header 보존
        image_msg->header = cmsg->header;

        {
            std::lock_guard<std::mutex> lock(m_buf);
            image_buf.push(image_msg);
        }

        // 불연속 감지
        const double t = image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * 1e-9;
        if (last_image_time == -1)
            last_image_time = t;
        else if (t - last_image_time > 1.0 || t < last_image_time)
        {
            ROS_WARN("image discontinue! detect a new sequence!");
            new_sequence();
        }
        last_image_time = t;
    }
    catch (const cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("loop_fusion"),
                     "cv_bridge decode failed: %s (format=%s)",
                     e.what(), cmsg->format.c_str());
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("loop_fusion"),
                     "compressed_image_callback std::exception: %s", e.what());
    }
    catch (...)
    {
        RCLCPP_ERROR(rclcpp::get_logger("loop_fusion"),
                     "compressed_image_callback unknown exception");
    }
}



void point_callback(const sensor_msgs::msg::PointCloud::SharedPtr point_msg)
{
    //ROS_INFO("point_callback!");
    {
        std::lock_guard<std::mutex> lk(m_buf);
        point_buf.push(point_msg);
        // 디버그: 들어올 땐 분명 늘어난다
        // RCLCPP_INFO(rclcpp::get_logger("loop_fusion"),
        //             "[point_cb] size=%zu t=%.9f", point_buf.size(),
        //             point_msg->header.stamp.sec + point_msg->header.stamp.nanosec*1e-9);
    }
    /*
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        printf("%d, 3D point: %f, %f, %f 2D point %f, %f \n",i , point_msg->points[i].x, 
                                                     point_msg->points[i].y,
                                                     point_msg->points[i].z,
                                                     point_msg->channels[i].values[0],
                                                     point_msg->channels[i].values[1]);
    }
    */
    // for visualization
    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::msg::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud->publish(point_cloud);
}

// only for visualization
void margin_point_callback(const sensor_msgs::msg::PointCloud::SharedPtr point_msg)
{
    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::msg::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_margin_cloud->publish(point_cloud);
}

void pose_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
{
    //ROS_INFO("pose_callback!");
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
    /*
    printf("pose t: %f, %f, %f   q: %f, %f, %f %f \n", pose_msg->pose.pose.position.x,
                                                       pose_msg->pose.pose.position.y,
                                                       pose_msg->pose.pose.position.z,
                                                       pose_msg->pose.pose.orientation.w,
                                                       pose_msg->pose.pose.orientation.x,
                                                       pose_msg->pose.pose.orientation.y,
                                                       pose_msg->pose.pose.orientation.z);
    */
}

void vio_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
{
    //ROS_INFO("vio_callback!");
    Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio *  vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;

    nav_msgs::msg::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = vio_t.x();
    odometry.pose.pose.position.y = vio_t.y();
    odometry.pose.pose.position.z = vio_t.z();
    odometry.pose.pose.orientation.x = vio_q.x();
    odometry.pose.pose.orientation.y = vio_q.y();
    odometry.pose.pose.orientation.z = vio_q.z();
    odometry.pose.pose.orientation.w = vio_q.w();
    pub_odometry_rect->publish(odometry);

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;        

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);


}

void extrinsic_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}

void process()
{
    const double EPS = 0.03;
    while (true)
    {
        sensor_msgs::msg::Image::ConstPtr image_msg = NULL;
        sensor_msgs::msg::PointCloud::ConstPtr point_msg = NULL;
        nav_msgs::msg::Odometry::ConstPtr pose_msg = NULL;

        // find out the messages with same time stamp
        if (!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            // front들의 시간
            auto tf = [&](auto& h) {
                return h.front()->header.stamp.sec + h.front()->header.stamp.nanosec * 1e-9;
            };

            // pop-while 대신, 최대 N회 시도 (무한루프 회피)
            int iter = 0;
            const int MAX_ITERS = 2000;
            while (iter++ < MAX_ITERS &&
                   !image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
            {
                double ti = tf(image_buf);
                double tp = tf(point_buf);
                double to = tf(pose_buf);

                double tmin = std::min({ti, tp, to});
                double tmax = std::max({ti, tp, to});

                if (tmax - tmin <= EPS)
                {
                    // 세 개가 충분히 가까움 → 동시에 pop해서 가져간다.
                    image_msg = image_buf.front(); image_buf.pop();
                    point_msg = point_buf.front(); point_buf.pop();
                    pose_msg  = pose_buf.front();  pose_buf.pop();
                    break;
                }

                // 가장 과거인 큐를 하나만 pop 해서 시간 맞추기
                if (ti == tmin)      image_buf.pop();
                else if (tp == tmin) point_buf.pop();
                else                 pose_buf.pop();
            }
        }
        else
        {
            // 참고용: 너무 자주 찍으면 성능에 영향 → 필요시 주석
            // printf("image=%d point=%d pose=%d\n", (int)image_buf.size(), (int)point_buf.size(), (int)pose_buf.size());
        }
        m_buf.unlock();

        if (pose_msg != NULL)
        {
            auto ts = [](const auto& h){
                return h.stamp.sec + h.stamp.nanosec * 1e-9;
            };
            // printf("pose  time %.9f\n",  ts(pose_msg->header));
            // printf("point time %.9f\n",  ts(point_msg->header));
            // printf("image time %.9f\n",  ts(image_msg->header));
            // skip fisrt few
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue;
            }

            if (skip_cnt < SKIP_CNT)
            {
                skip_cnt++;
                continue;
            }
            else
            {
                skip_cnt = 0;
            }

            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1")
            {
                sensor_msgs::msg::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat image = ptr->image;
            // build keyframe
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();


            if((T - last_t).norm() > SKIP_DIS)
            {
                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                vector<double> point_id;


            //     // COL=1224, ROW=384 (camera.yaml과 일치)
            //     const int border = 2; // 가장자리 여유
            //     const size_t Np = point_msg->points.size();

            //     // per-point 레이아웃 확인 (이미 로그상 true)
            //     bool per_point_layout = (point_msg->channels.size() == Np &&
            //                             Np > 0 &&
            //                             point_msg->channels[0].values.size() >= 5);

            //     if (!per_point_layout) {
            //         RCLCPP_ERROR(rclcpp::get_logger("loop_fusion"),
            //                     "Unexpected channel layout: channels.size()=%zu, points=%zu",
            //                     point_msg->channels.size(), Np);
            //         continue;
            //     }

            //     int kept=0, dropped_short=0, dropped_nan=0, dropped_oob=0;


            //     for (size_t i = 0; i < Np; ++i)
            //     {
            //         const auto& P = point_msg->points[i];
            //         if (!std::isfinite(P.x) || !std::isfinite(P.y) || !std::isfinite(P.z)) {
            //             ++dropped_nan; continue;
            //         }

            //         const auto& vals = point_msg->channels[i].values; // [un, vn, u, v, id]
            //         if (vals.size() < 5) { ++dropped_short; continue; }

            //         float un = vals[0];
            //         float vn = vals[1];
            //         float u  = vals[2];
            //         float v  = vals[3];
            //         double pid = vals[4];

            //         if (!std::isfinite(un) || !std::isfinite(vn) ||
            //             !std::isfinite(u)  || !std::isfinite(v)) {
            //             ++dropped_nan; continue;
            //         }

            //         // 경계 체크 (패치/디스크립터 추출을 고려해 여유 border 제외)
            //         if (u < border || v < border || u > (COL - 1 - border) || v > (ROW - 1 - border)) {
            //             ++dropped_oob; continue;
            //         }

            //         point_3d.emplace_back(P.x, P.y, P.z);
            //         point_2d_uv.emplace_back(u, v);
            //         point_2d_normal.emplace_back(un, vn);
            //         point_id.emplace_back(pid);
            //         ++kept;

            //         // 필요시 디버그
            //         // printf("u %.3f, v %.3f\n", u, v);
            //     }

            //     if (kept == 0) {
            //         RCLCPP_WARN(rclcpp::get_logger("loop_fusion"),
            //                     "No valid points (short=%d nan=%d oob=%d)", dropped_short, dropped_nan, dropped_oob);
            //         continue;
            //     } else {
            //         RCLCPP_INFO(rclcpp::get_logger("loop_fusion"),
            //                     "points kept=%d/%zu (short=%d nan=%d oob=%d)",
            //                     kept, Np, dropped_short, dropped_nan, dropped_oob);
            //     }

            //     // --- 키프레임 만들기 전, 안전성 체크(선택) ---
            //     if ( (int)image.cols != COL || (int)image.rows != ROW ) {
            //         RCLCPP_WARN(rclcpp::get_logger("loop_fusion"),
            //                     "image size %dx%d != expected %dx%d", image.cols, image.rows, COL, ROW);
            //     }
            //     if (image.type() != CV_8UC1) {
            //         RCLCPP_WARN(rclcpp::get_logger("loop_fusion"),
            //                     "image type=%d (expect CV_8UC1=0)", image.type());
            //     }
                
            //     // 모두 같은 길이인지 재확인
            //     if (!(point_3d.size()==point_2d_uv.size() &&
            //             point_3d.size()==point_2d_normal.size() &&
            //             point_3d.size()==point_id.size())) {
            //         RCLCPP_ERROR(rclcpp::get_logger("loop_fusion"),
            //                     "vector size mismatch: 3d=%zu uv=%zu norm=%zu id=%zu",
            //                     point_3d.size(), point_2d_uv.size(), point_2d_normal.size(), point_id.size());
            //         continue;
            //     }
                
            //     // --- KeyFrame / addKeyFrame 예외 포착 ---
            //     try {
            //         const double stamp = pose_msg->header.stamp.sec + pose_msg->header.stamp.nanosec * 1e-9;
                
            //         // 디버그: 값 범위 한번 찍기(첫 3개)
            //         auto pr = [&](const char* tag, const cv::Point2f& p){
            //         RCLCPP_INFO(rclcpp::get_logger("loop_fusion"), "%s=(%.3f, %.3f)", tag, p.x, p.y);
            //         };
            //         if (!point_2d_uv.empty()) {
            //         pr("uv0", point_2d_uv[0]);
            //         pr("n0 ", point_2d_normal[0]);
            //         }
                
            //         KeyFrame* keyframe = new KeyFrame(
            //             stamp, frame_index, T, R, image,
            //             point_3d, point_2d_uv, point_2d_normal, point_id, sequence);
                
            //         {
            //         std::lock_guard<std::mutex> _(m_process);
            //         start_flag = 1;
            //         posegraph.addKeyFrame(keyframe, 1);
            //         }
                
            //         ++frame_index;
            //         last_t = T;
            //     }
            //     catch (const std::string& e) {
            //         RCLCPP_ERROR(rclcpp::get_logger("loop_fusion"),
            //                     "KeyFrame/addKeyFrame threw std::string: %s", e.c_str());
            //         continue;
            //     }
            //     catch (const std::exception& e) {
            //         RCLCPP_ERROR(rclcpp::get_logger("loop_fusion"),
            //                     "KeyFrame/addKeyFrame std::exception: %s", e.what());
            //         continue;
            //     }
            //     catch (...) {
            //         RCLCPP_ERROR(rclcpp::get_logger("loop_fusion"),
            //                     "KeyFrame/addKeyFrame unknown exception");
            //         continue;
            //     }

                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x;
                    p_3d.y = point_msg->points[i].y;
                    p_3d.z = point_msg->points[i].z;
                    point_3d.push_back(p_3d);

                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    p_2d_normal.x = point_msg->channels[i].values[0];
                    p_2d_normal.y = point_msg->channels[i].values[1];
                    p_2d_uv.x = point_msg->channels[i].values[2];
                    p_2d_uv.y = point_msg->channels[i].values[3];
                    p_id = point_msg->channels[i].values[4];
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);

                    // printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
                }

                KeyFrame* keyframe = new KeyFrame(pose_msg->header.stamp.sec + pose_msg->header.stamp.nanosec * (1e-9), frame_index, T, R, image,
                                   point_3d, point_2d_uv, point_2d_normal, point_id, sequence);   
                m_process.lock();
                start_flag = 1;
                posegraph.addKeyFrame(keyframe, 1);
                m_process.unlock();
                frame_index++;
                last_t = T;
            }
        }
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

void command()
{
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            m_process.lock();
            posegraph.savePoseGraph();
            m_process.unlock();
            printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
            printf("program shutting down...\n");
            rclcpp::shutdown();
        }
        if (c == 'n')
            new_sequence();

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto n = rclcpp::Node::make_shared("loop_fusion");
    posegraph.registerPub(n);
    
    VISUALIZATION_SHIFT_X = 0;
    VISUALIZATION_SHIFT_Y = 0;
    SKIP_CNT = 0;
    SKIP_DIS = 0;

    if(argc != 2)
    {
        printf("please intput: rosrun loop_fusion loop_fusion_node [config file] \n"
               "for example: rosrun loop_fusion loop_fusion_node "
               "/home/tony-ws1/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 0;
    }
    
    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);

    std::string IMAGE_TOPIC;
    int LOAD_PREVIOUS_POSE_GRAPH;

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];

    // referred from: https://answers.ros.org/question/288501/ros2-equivalent-of-rospackagegetpath/
    std::string pkg_path = ament_index_cpp::get_package_share_directory("loop_fusion");
    // string vocabulary_file = pkg_path + "/../support_files/brief_k10L6.bin";
    string vocabulary_file = "/root/VINS-Fusion/support_files/brief_k10L6.bin";
    cout << "vocabulary_file" << vocabulary_file << endl;
    posegraph.loadVocabulary(vocabulary_file);

    // BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";
    BRIEF_PATTERN_FILE = "/root/VINS-Fusion/support_files/brief_pattern.yml";
    cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    printf("cam calib path: %s\n", cam0Path.c_str());
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());

    fsSettings["image0_topic"] >> IMAGE_TOPIC;        
    fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
    fsSettings["output_path"] >> VINS_RESULT_PATH;
    fsSettings["save_image"] >> DEBUG_IMAGE;
    fsSettings["compressed"] >> USE_Compressed;

    LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];
    VINS_RESULT_PATH = VINS_RESULT_PATH + "/vio_loop.csv";
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    int USE_IMU = fsSettings["imu"];
    posegraph.setIMUFlag(USE_IMU);
    fsSettings.release();

    if (LOAD_PREVIOUS_POSE_GRAPH)
    {
        printf("load pose graph\n");
        m_process.lock();
        posegraph.loadPoseGraph();
        m_process.unlock();
        printf("load pose graph finish\n");
        load_flag = 1;
    }
    else
    {
        printf("no previous pose graph\n");
        load_flag = 1;
    }

    auto sub_vio          = n->create_subscription<nav_msgs::msg::Odometry>("/vins_odometry", rclcpp::QoS(rclcpp::KeepLast(2000)), vio_callback);


    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_image_c = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image = nullptr;

    if (USE_Compressed) {
        sub_image_c       = n->create_subscription<sensor_msgs::msg::CompressedImage>(IMAGE_TOPIC, rclcpp::QoS(rclcpp::KeepLast(2000)), compressed_image_callback);
    }
    else{
        sub_image         = n->create_subscription<sensor_msgs::msg::Image>(IMAGE_TOPIC, rclcpp::QoS(rclcpp::KeepLast(2000)), image_callback);
    }




    auto q_reliable = rclcpp::QoS(rclcpp::KeepLast(2000)).reliable().durability_volatile();
    auto sub_pose         = n->create_subscription<nav_msgs::msg::Odometry>("/keyframe_pose", q_reliable, pose_callback);




    auto sub_extrinsic    = n->create_subscription<nav_msgs::msg::Odometry>("/extrinsic", rclcpp::QoS(rclcpp::KeepLast(2000)), extrinsic_callback);




    auto sub_point        = n->create_subscription<sensor_msgs::msg::PointCloud>("/keyframe_point", q_reliable, point_callback);





    auto sub_margin_point = n->create_subscription<sensor_msgs::msg::PointCloud>("/margin_cloud", rclcpp::QoS(rclcpp::KeepLast(2000)), margin_point_callback);


    pub_match_img          = n->create_publisher<sensor_msgs::msg::Image>("match_image", 1000);
    pub_camera_pose_visual = n->create_publisher<visualization_msgs::msg::MarkerArray>("camera_pose_visual", 1000);
    pub_point_cloud        = n->create_publisher<sensor_msgs::msg::PointCloud>("point_cloud_loop_rect", 1000);
    pub_margin_cloud       = n->create_publisher<sensor_msgs::msg::PointCloud>("margin_cloud_loop_rect", 1000);
    pub_odometry_rect      = n->create_publisher<nav_msgs::msg::Odometry>("odometry_rect", 1000);

    std::thread measurement_process;
    std::thread keyboard_command_process;

    measurement_process = std::thread(process);
    keyboard_command_process = std::thread(command);
    
    rclcpp::spin(n);

    return 0;
}
