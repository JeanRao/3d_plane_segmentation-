#include <iostream>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <eigen3/Eigen/Dense>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include <pcl/filters/crop_box.h>
#include <pcl/common/centroid.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Plane_3.h>
#include <CGAL/Line_3.h>
#include <CGAL/intersections.h>


// Define kernel for CGAL
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Plane_3 Plane_3;
typedef Kernel::Line_3 Line_3;

// 定义点云类型别名
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

// 函数原型声明
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> get_all_plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

//降样
pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf_size);

//bbox
pcl::PointCloud<pcl::PointXYZ>::Ptr extractPointsInBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                       Eigen::Vector4f min_pt,
                                                       Eigen::Vector4f max_pt);

// 分割函数声明
void segment_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  pcl::PointIndices::Ptr inliers,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud);
// 显示多个点云
void show_multiple_point_clouds(const std::vector<PointCloudXYZPtr>& cloud_list);

// 检查两个平面是否相交
// 1.检查平面夹角是否大于45 度，小于135 度 2.计算交线， 计算两个平面上点到交线的距离，在GAP范围内的点数分别有多少， 如果小于阈值，则判断为不相交
bool check_if_two_plane_connect(const std::vector<PointCloudXYZPtr> two_plane,
                                float gap);

// Sviz_Plane
class Sviz_Plane {
public:
    // Constructor
    Sviz_Plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<double>& plane_model);

    // Function to compute intersection with another plane
    boost::optional<boost::variant<CGAL::Line_3<CGAL::Epick>, CGAL::Plane_3<CGAL::Epick> > > intersect_plane(const Sviz_Plane& plane) const;

    // Function to visualize the plane
    void visualize(bool show_normal) const;


private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_pcd_;
    Plane_3 cgal_plane_;
    std::vector<double> plane_model_;
};


Sviz_Plane::Sviz_Plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<double>& plane_model) {
    plane_pcd_ = cloud;
    if (plane_model.size() != 4) {
        std::cerr << "Error: Plane equation coefficients must have size 4 (a, b, c, d)" << std::endl;
        return;
    }
    cgal_plane_ = Plane_3(plane_model[0], plane_model[1], plane_model[2], plane_model[3]);
}

// Function to compute intersection with another plane
boost::optional<boost::variant<CGAL::Line_3<CGAL::Epick>, CGAL::Plane_3<CGAL::Epick> > > Sviz_Plane::intersect_plane(const Sviz_Plane& plane) const {
    return CGAL::intersection(this->cgal_plane_, plane.cgal_plane_);
}

// Function to visualize the plane
void Sviz_Plane::visualize(bool show_normal) const {
    pcl::visualization::PCLVisualizer viewer("SVIZ Plane");
    viewer.setBackgroundColor(0, 0, 0);

    // Add point cloud
    viewer.addPointCloud<pcl::PointXYZ>(this->plane_pcd_, "point_cloud");


    // Add normal if required
    if (show_normal) {
        Point_3 origin = this->cgal_plane_.point();
        Point_3 normal_end = origin + this->cgal_plane_.orthogonal_vector();
        viewer.addLine<pcl::PointXYZ>(pcl::PointXYZ(origin.x(), origin.y(), origin.z()),
                                      pcl::PointXYZ(normal_end.x(), normal_end.y(), normal_end.z()),
                                      1.0, 0.0, 0.0, "normal");
    }

    viewer.spin();
}

//bool check_if_two_plane_connect(const std::vector<PointCloudXYZPtr> two_plane,
//                                float gap){
//    PointCloudXYZPtr plane1(new pcl::PointCloud<pcl::PointXYZ>(*two_plane[0]));
//    PointCloudXYZPtr plane2(new pcl::PointCloud<pcl::PointXYZ>(*two_plane[1]));
//    
//}

// 计算平面间焊缝

// 计算焊接角度



int main() {
    // 加载点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/td/project/svision_c/top.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return -1;
    }
    // bbox
    Eigen::Vector4f min_pt(-1.0, -1.0, -1.0, 1.0); // 包围框最小点的坐标
    Eigen::Vector4f max_pt(1.0, 1.0, 1.0, 1.0);    // 包围框最大点的坐标
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud = extractPointsInBox(cloud, min_pt, max_pt);

    //降样
    float leaf_size = 0.002; // 体素立方体的边长
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled = downsamplePointCloud(cropped_cloud, leaf_size);
    // 计算点云的质心
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cropped_cloud, centroid);
    std::cout << "Centroid: " << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << std::endl;

    // 调用获取所有平面函数
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> inlier_cloud_list = get_all_plane(cloud_downsampled);

    // 打印内点点云列表的大小
    std::cout << "Number of plane segments: " << inlier_cloud_list.size() << std::endl;
    // 显示多个点云
    show_multiple_point_clouds(inlier_cloud_list);

    return 0;
}

// 获取所有平面函数定义
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> get_all_plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> inlier_cloud_list;

    while (cloud->points.size() >= 2000) {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        segment_once(cloud, inliers, inlier_cloud);

        // 将内点点云保存到列表中
        if (inliers->indices.size ()>10000){
            inlier_cloud_list.push_back(inlier_cloud);
        }


        // 更新点云为剩余点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true); // 提取剩余点云
        extract.filter(*remaining_cloud);
        *cloud = *remaining_cloud;
    }

    return inlier_cloud_list;
}

// 分割函数定义
void segment_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  pcl::PointIndices::Ptr inliers,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud) {
    // 创建一个分割器
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.002);
    seg.setNumberOfThreads(4);
    seg.setInputCloud(cloud);
    seg.setMaxIterations(1000);
//    seg.setEpsAngle(0.174);
    seg.segment(*inliers, *coefficients);


    // 创建一个提取器
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false); // 提取内点
    extract.filter(*inlier_cloud);
}

// 显示多个点云
void show_multiple_point_clouds(const std::vector<PointCloudXYZPtr>& cloud_list) {
    // 创建可视化窗口
    pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
    viewer.setBackgroundColor(0, 0, 0);

    // 添加所有点云到窗口
    for (size_t i = 0; i < cloud_list.size(); ++i) {
        PointCloudXYZPtr cloud = cloud_list[i];
        std::string cloud_name = "cloud_" + std::to_string(i);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud, rand() % 256, rand() % 256, rand() % 256);
        viewer.addPointCloud<pcl::PointXYZ>(cloud, color_handler, cloud_name);
    }

    // 显示点云
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf_size) {
    // 创建 VoxelGrid 过滤器
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);

    // 执行滤波
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    voxel_grid.filter(*cloud_downsampled);

    return cloud_downsampled;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr extractPointsInBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                       Eigen::Vector4f min_pt,
                                                       Eigen::Vector4f max_pt) {
    // 创建 CropBox 滤波器
    pcl::CropBox<pcl::PointXYZ> crop_box_filter;
    crop_box_filter.setInputCloud(cloud);
    crop_box_filter.setMin(min_pt);
    crop_box_filter.setMax(max_pt);

    // 执行滤波
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    crop_box_filter.filter(*cropped_cloud);

    return cropped_cloud;
}
