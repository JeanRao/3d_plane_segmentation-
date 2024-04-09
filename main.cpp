#include <iostream>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/segmentation/extract_clusters.h>
#include <eigen3/Eigen/Dense>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include <pcl/filters/crop_box.h>
#include <pcl/common/centroid.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Plane_3.h>
#include <CGAL/Line_3.h>
#include <CGAL/intersections.h>
#include <cmath>
#include <boost/variant/variant.hpp>
#include <boost/optional/optional.hpp>
#include <CGAL/squared_distance_3_0.h>
#include <fstream>
#include <yaml-cpp/yaml.h>


// Define kernel for CGAL
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Plane_3 Plane_3;
typedef Kernel::Line_3 Line_3;
typedef Kernel::Vector_3 Vector_3;
typedef Kernel::Direction_3 Direction_3;

// 定义点云类型别名
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

// 函数原型声明


//降样
pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf_size);

//bbox
pcl::PointCloud<pcl::PointXYZ>::Ptr extractPointsInBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                       Eigen::Vector4f min_pt,
                                                       Eigen::Vector4f max_pt);
// calculate normal 
pcl::PointCloud<pcl::PointNormal>::Ptr computeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // 创建法线估计器对象
    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    ne.setInputCloud(cloud);

    // 创建一个 KdTree 对象（用于法线估计）
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);

    // 设置计算法线的半径
    ne.setRadiusSearch(0.05);

    // 计算法线
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    ne.compute(*cloud_with_normals);

    // 移除没有法线的点云
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    for (size_t i = 0; i < cloud_with_normals->size(); ++i) {
        if (!pcl::isFinite<pcl::PointNormal>((*cloud_with_normals)[i])) {
            indices->indices.push_back(i);
        }
    }
    pcl::ExtractIndices<pcl::PointNormal> extract;
    extract.setInputCloud(cloud_with_normals);
    extract.setIndices(indices);
    extract.setNegative(true);
    extract.filter(*cloud_with_normals);

    return cloud_with_normals;
}

// 分割函数声明
void segment_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  pcl::PointIndices::Ptr inliers,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud,
                  std::vector<float> &plane_model);
// 显示多个点云
void show_multiple_point_clouds(const std::vector<PointCloudXYZPtr>& cloud_list);

// 检查两个平面是否相交
// 1.检查平面夹角是否大于45 度，小于135 度 2.计算交线， 计算两个平面上点到交线的距离，在GAP范围内的点数分别有多少， 如果小于阈值，则判断为不相交


// 定义访问者类


// Sviz_Plane
class Sviz_Plane {
public:
    // Constructor
    Sviz_Plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<float>& plane_model);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_pcd_;
    Plane_3 cgal_plane_;
    std::vector<float> plane_model_;

    // Function to compute intersection with another plane
    boost::optional<boost::variant<CGAL::Line_3<CGAL::Epick>, CGAL::Plane_3<CGAL::Epick> > > intersect_plane(const Sviz_Plane& plane) const;


    // Function to visualize the plane
    void visualize(bool show_normal) const;


//private:
//    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_pcd_;
//    Plane_3 cgal_plane_;
//    std::vector<float> plane_model_;
};


Sviz_Plane::Sviz_Plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<float>& plane_model) {
    plane_pcd_ = cloud;
    if (plane_model.size() != 4) {
        std::cerr << "Error: Plane equation coefficients must have size 4 (a, b, c, d)" << std::endl;
        return;
    }
    cgal_plane_ = Plane_3(plane_model[0], plane_model[1], plane_model[2], plane_model[3]);
    plane_model_ = plane_model;
}

// Function to compute intersection with another plane
boost::optional<boost::variant<CGAL::Line_3<CGAL::Epick>, CGAL::Plane_3<CGAL::Epick> > > Sviz_Plane::intersect_plane(const Sviz_Plane& plane) const {
    return CGAL::intersection(this->cgal_plane_, plane.cgal_plane_);
}

// segment all plane
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> get_all_plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<Sviz_Plane> & all_plane);



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

//void CombineDuplicatePlane(std:: vector<Sviz_Plane> & all_plane, std:: vector<Sviz_Plane> & new_all_plane);
//void CombineDuplicatePlane(std:: vector<Sviz_Plane> & all_plane, std:: vector<Sviz_Plane> & new_all_plane){
//    // if the model of 2 plane is the same , combine the pcd of 2 planes and take average of 2 as plane model
//
//}
bool compare_function(const Eigen::Vector3d &v1, const Eigen:: Vector3d &v2, int axis, bool reverse);
bool compare_function(const Eigen::Vector3d &v1, const Eigen:: Vector3d &v2, int axis, bool reverse){
    bool result;
    if(axis==0){
        result=  v1.x()<v2.x();
    }
    else if (axis==1){
        result = v1.y()<v2.y();
    }
    else {
        result = v1.z()<v2.z();
    }
    if (reverse==true){
        return not result;
    }else{
        return result ;
    }
}


std::vector<PointCloudXYZ> EuclideanCluster(PointCloudXYZPtr &cloud);
std::vector<PointCloudXYZ> EuclideanCluster(PointCloudXYZPtr &cloud){
    // 创建点云聚类对象
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.02); // 设置聚类的容差
    ec.setMinClusterSize(100); // 设置聚类的最小大小
    ec.setMaxClusterSize(25000); // 设置聚类的最大大小

    ec.setInputCloud(cloud);

    // 执行聚类
    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);
    //
    std::vector<PointCloudXYZ> clusters ;
    clusters.push_back(*cloud);
    return clusters;
    }

std::vector<Eigen::Vector3d> GetFarthestPoints (std::vector<Eigen::Vector3d> &pointlist);
std::vector<Eigen::Vector3d> GetFarthestPoints (std::vector<Eigen::Vector3d>& pointlist){
    std::vector<Eigen::Vector3d> farthest;
    double distance_along_axis;
    double max_distance;
    max_distance=0;
    for (size_t i=0; i<3; ++i){
        std::vector<Eigen::Vector3d> farthest_along_axis;
        bool reverse ;
        reverse = false;
        std:: sort(pointlist.begin(),pointlist.end(), [& i, &reverse](const Eigen::Vector3d &v1, const Eigen:: Vector3d &v2){
            return compare_function(v1,v2,i,reverse);
        });
        farthest_along_axis.push_back(pointlist.front());
        farthest_along_axis.push_back(pointlist.back());
        distance_along_axis = (farthest_along_axis[0]-farthest_along_axis[1]).norm();
        if (max_distance< distance_along_axis){
            max_distance= distance_along_axis;
            size_t start;
            start = farthest_along_axis.size()-2;
            std::vector<Eigen::Vector3d> farthest_temp(farthest_along_axis.begin()+start,farthest_along_axis.end());
            farthest = farthest_temp;
            }
        }
    return farthest;
}


// 点云到直线的距离
void DistPtsToLine(const PointCloudXYZPtr cloud, Line_3 *line, std::vector<double> & distances);
void DistPtsToLine(const PointCloudXYZPtr cloud, Line_3 *line, std::vector<double> & distances){
    for (const auto & point : *cloud){
        Point_3 pcl_point(point.x,point.y,point.z);
        double distance;
        Point_3 line_point;
        line_point = line->point();
        distance = CGAL::squared_distance(pcl_point,line_point);
        Eigen::Vector3d a(pcl_point.x()-line_point.x(),pcl_point.y()-line_point.y(), pcl_point.z()-line_point.z());
        Vector_3 b;
        b= line->direction().vector() ;
        Eigen::Vector3d c(b[0], b[1],b[2]);
        double c_norm = c.norm();
        double projection = a.dot(c)/c_norm;
        double distance_fin = sqrt(distance-projection*projection);
        distances.push_back(distance_fin);
    }
};
std::vector<Eigen::Vector3d> projectPointsToLine(const std::vector<Eigen::Vector3d>& points, const Line_3& line);
std::vector<Eigen::Vector3d> projectPointsToLine(const std::vector<Eigen::Vector3d>& points, const Line_3& line) {
    std::vector<Eigen::Vector3d> projected_points;

    for (const auto& point : points) {
        // 将每个点投影到直线上
        Point_3 cgal_point(point.x(), point.y(), point.z());
        Point_3 projected_cgal_point = line.projection(cgal_point);

        // 将投影点转换为 Eigen 3D 向量并添加到输出向量中
        Eigen::Vector3d projected_point(projected_cgal_point.x(), projected_cgal_point.y(), projected_cgal_point.z());
        projected_points.push_back(projected_point);
    }

    return projected_points;
}

// dbscan
class DBSCAN {
public:
    DBSCAN(double eps, int minPts) : eps_(eps), minPts_(minPts) {}

    void setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        cloud_ = cloud;
    }

    void extract(std::vector<pcl::PointIndices>& cluster_indices) {
        std::vector<bool> visited(cloud_->size(), false);
        std::vector<int> cluster(cloud_->size(), -1);
        int cluster_idx = 0;

        for (size_t i = 0; i < cloud_->size(); ++i) {
            if (!visited[i]) {
                visited[i] = true;
                std::vector<int> neighbors;
                regionQuery(i, neighbors);
                if (neighbors.size() < minPts_) {
                    cluster[i] = -1; // 噪声点
                } else {
                    expandCluster(i, neighbors, cluster, visited, cluster_idx);
                    cluster_idx++;
                }
            }
        }

        cluster_indices.clear();
        cluster_indices.resize(cluster_idx);
        for (size_t i = 0; i < cloud_->size(); ++i) {
            if (cluster[i] != -1) {
                cluster_indices[cluster[i]].indices.push_back(i);
            }
        }
    }

private:
    double eps_;
    int minPts_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;

    void regionQuery(int idx, std::vector<int>& neighbors) {
        neighbors.clear();
        for (size_t i = 0; i < cloud_->size(); ++i) {
            if ((cloud_->points[i].getVector3fMap() - cloud_->points[idx].getVector3fMap()).norm() <= eps_) {
                neighbors.push_back(i);
            }
        }
    }

    void expandCluster(int idx, const std::vector<int>& neighbors, std::vector<int>& cluster,
                       std::vector<bool>& visited, int cluster_idx) {
        cluster[idx] = cluster_idx;
        for (size_t i = 0; i < neighbors.size(); ++i) {
            int neighbor_idx = neighbors[i];
            if (!visited[neighbor_idx]) {
                visited[neighbor_idx] = true;
                std::vector<int> neighbor_neighbors;
                regionQuery(neighbor_idx, neighbor_neighbors);
                if (neighbor_neighbors.size() >= minPts_) {
                    expandCluster(neighbor_idx, neighbor_neighbors, cluster, visited, cluster_idx);
                }
            }
            if (cluster[neighbor_idx] == -1) {
                cluster[neighbor_idx] = cluster_idx;
            }
        }
    }
};
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> findLargestCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
//    // 创建 DBSCAN 聚类对象
//    double eps = 0.1;
//    int minPts = 50;
//    DBSCAN dbscan(eps, minPts);
//    dbscan.setInputCloud(cloud);
//
//    // 执行聚类
//    std::vector<pcl::PointIndices> cluster_indices;
//    dbscan.extract(cluster_indices);
//
//    // 找到最大的聚类
//    size_t max_cluster_size = 0;
//    size_t max_cluster_index = 0;
//    for (size_t i = 0; i < cluster_indices.size(); ++i) {
//        if (cluster_indices[i].indices.size() > max_cluster_size) {
//            max_cluster_size = cluster_indices[i].indices.size();
//            max_cluster_index = i;
//        }
//    }
//
//    // 将最大的子点云放入 final_inlier 中，剩下的放入 remains 中
//    pcl::PointCloud<pcl::PointXYZ>::Ptr final_inlier(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr remains(new pcl::PointCloud<pcl::PointXYZ>);
//    if (max_cluster_index != -1) {
//        pcl::copyPointCloud(*cloud, cluster_indices[max_cluster_index].indices, *final_inlier);
//        // 从 new_inlier_cloud 中移除最大的子点云对应的点
//        pcl::ExtractIndices<pcl::PointXYZ> extract;
//        extract.setInputCloud(cloud);
//        extract.setIndices(boost::make_shared<const pcl::PointIndices>(cluster_indices[max_cluster_index]));
//        extract.setNegative(true);
//        extract.filter(*remains);
//    } else {
//        // 如果未找到聚类，则将所有点放入 remains 中
//        remains = cloud;
//    }
//    return {final_inlier,remains};

// EUCLIDIAN
    // 创建点云聚类对象
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.02); // 设置聚类的容差
    ec.setMinClusterSize(100); // 设置聚类的最小大小
    ec.setMaxClusterSize(25000); // 设置聚类的最大大小

    ec.setInputCloud(cloud);

    // 执行聚类
    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    // 找到聚类中点数最多的类的索引
    int max_cluster_index = 0;
    int max_cluster_size = 0;
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        if (cluster_indices[i].indices.size() > max_cluster_size) {
            max_cluster_size = cluster_indices[i].indices.size();
            max_cluster_index = i;
        }
    }
    // 创建最终的聚类点云和剩余点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr remains(new pcl::PointCloud<pcl::PointXYZ>);

    // 将最大的聚类放入 final_cluster，剩下的放入 remains
    if (max_cluster_index != -1)
    {
        for (size_t i = 0; i < cluster_indices.size(); ++i) {
        if (i == max_cluster_index) {
            pcl::copyPointCloud(*cloud, cluster_indices[i], *final_cluster);
        } else {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*cloud, cluster_indices[i], *cluster_cloud);
            *remains += *cluster_cloud;
        }
        }
    }
    else {
        // 如果未找到聚类，则将所有点放入 remains 中
        remains = cloud;
    }
    return {final_cluster,remains};
}

bool CheckIfTwoPlaneConnect(const std::vector<Sviz_Plane>& two_plane,
                                float gap, std::vector<Eigen::Vector3d> &weld_line);

bool CheckIfTwoPlaneConnect(const std::vector<Sviz_Plane>& two_plane,
                                float gap, std::vector<Eigen::Vector3d> &weld_line){
    Eigen::Vector3d v1(3);
    v1<< two_plane[0].plane_model_[0],two_plane[0].plane_model_[1],two_plane[0].plane_model_[2] ;
    v1.normalize();
    Eigen::Vector3d v2(3);
    v2<< two_plane[1].plane_model_[0],two_plane[1].plane_model_[1],two_plane[1].plane_model_[2];
    v2.normalize();
    double cosine_similarity = v1.dot(v2);
    if (fabs(cosine_similarity)>0.7){
        return false;
    }else{
        boost::optional<boost::variant<CGAL::Line_3<CGAL::Epick>, CGAL::Plane_3<CGAL::Epick> > > connect_line;
        connect_line = two_plane[0].intersect_plane(two_plane[1]);
        if (connect_line){
            if (auto line_ptr = boost::get<CGAL::Line_3<CGAL::Epick>>(&(*connect_line)))
            {
                    Direction_3 direction = line_ptr->direction();
//                    std::cout << 'inetrsect line direction'<< direction <<std::endl;
                    std::vector<PointCloudXYZPtr> point_clouds;
                    point_clouds.push_back(two_plane[0].plane_pcd_);
                    point_clouds.push_back(two_plane[1].plane_pcd_);
                    std::vector<double> distance1;
                    std::vector<double> distance2;
                    DistPtsToLine(two_plane[0].plane_pcd_,line_ptr,distance1);
                    DistPtsToLine(two_plane[1].plane_pcd_,line_ptr, distance2);
                    std::vector<Eigen::Vector3d> close_points1;
                    int count1 =0 ;
                    for (size_t i=0; i<distance1.size(); ++i){
                        double num ;
                        num= distance1[i];
                        if (abs(num)<gap){
                            Eigen::Vector3d close_point1 ;
                            close_point1.x() = two_plane[0].plane_pcd_->points[i].x;
                            close_point1.y() = two_plane[0].plane_pcd_->points[i].y;
                            close_point1.z() = two_plane[0].plane_pcd_->points[i].z;
                            close_points1.push_back(close_point1);
                            count1++;
                        }
                    }
                    std::vector<Eigen::Vector3d> close_points2;
                    int count2= 0;
                    for (size_t i=0; i<distance2.size(); ++i){
                        double num ;
                        num = distance2[i];
                        if (abs(num)<gap){
                            Eigen::Vector3d close_point2 ;
                            close_point2.x() = two_plane[1].plane_pcd_->points[i].x;
                            close_point2.y() = two_plane[1].plane_pcd_->points[i].y;
                            close_point2.z() = two_plane[1].plane_pcd_->points[i].z;
                            close_points2.push_back(close_point2);
                            count2++;
                        }
                    }
                    if (count1<100 or count2<100){
                        return false;
                    }
                    // compute weld line
                    // get the close point projected 1 , projected 2
                    std::vector<Eigen::Vector3d> projected_close_points1;
                    projected_close_points1 = projectPointsToLine(close_points1,*line_ptr);
                    std::vector<Eigen::Vector3d> projected_close_points2;
                    projected_close_points2 = projectPointsToLine(close_points2,*line_ptr);
                    // get farthest points 1 farthest points 2
                    std::vector<Eigen::Vector3d> farthest1;
                    std::vector<Eigen::Vector3d> farthest2;
                    farthest1 = GetFarthestPoints(projected_close_points1);
                    farthest2 = GetFarthestPoints(projected_close_points2);
                    // use the point pair with shortest length
                    double length1 ;
                    length1 = (farthest1[0]-farthest1[1]).norm();
                    double length2 ;
                    length2 = (farthest2[0]-farthest2[1]).norm();
                    if (length1>length2){
                        weld_line =  farthest2;
                    }else{
                            weld_line =  farthest1;
                            }
//                    show_multiple_point_clouds(point_clouds);
                    return true;
            } else{
//                std::cout << 'two_plane is parallel' <<std::endl;
                return false;
            }
        }
        else{
            return false;
        };
    }

}

// 计算平面间焊缝
void get_all_lines(const std::vector<Sviz_Plane>& all_plane,
        std::vector<Line_3> &weld_lines);
void get_all_lines(const std::vector<Sviz_Plane> & all_plane, std::vector<Line_3>& weld_lines){
    for (size_t i=0; i<all_plane.size(); ++i){
        for (size_t j= i+1; j< all_plane.size(); ++j){
            std::vector<Sviz_Plane> two_plane;
            two_plane = {all_plane[i],all_plane[j]};
//            if CheckIfTwoPlaneConnect(two_plane,0.01){
//
//
//            }
        }
    }
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

class WeldLineRec {
private:
    std::vector<Sviz_Plane> two_plane_;
    std::vector<Eigen::Vector3d> ends_;
    std::vector<std::vector<double>> welding_points_;
    void compute_welding_angle(){
        Eigen::Vector3d normal1, normal2;
        normal1[0] = two_plane_[0].plane_model_[0];
        normal1[1] = two_plane_[0].plane_model_[1];
        normal1[2] = two_plane_[0].plane_model_[2];
        normal2[0] = two_plane_[1].plane_model_[0];
        normal2[1] = two_plane_[1].plane_model_[1];
        normal2[2] = two_plane_[1].plane_model_[2];
        // vertical or horizontal
        if (fabs(normal1[2]) > 0.75 or fabs(normal2[2]) > 0.75) {
            // horizontal
            Eigen::Vector3d welding_vector;
            welding_vector = (normal1 + normal2) / 2;
            Eigen::Vector3d golden_vector2(0.00260663, 0.48484533, 0.51956243);
            // rotation
            Eigen::Matrix3d rotationMatrix;
            rotationMatrix = Eigen::Quaterniond::FromTwoVectors(golden_vector2, welding_vector).toRotationMatrix();

            // 已知的欧拉角（单位为度）
            double eulerAngleA_deg[3] = {175.0, 40.0, 50.0};
            double eulerAngleB_deg[3] = {175.0, 40.0, 130.0};

            // 将欧拉角转换为旋转矩阵
            Eigen::Matrix3d rotationMatrixA;
            Eigen::Matrix3d rotationMatrixB;
            rotationMatrixA = Eigen::AngleAxisd(eulerAngleA_deg[0] * M_PI / 180.0, Eigen::Vector3d::UnitX())
                              * Eigen::AngleAxisd(eulerAngleA_deg[1] * M_PI / 180.0, Eigen::Vector3d::UnitY())
                              * Eigen::AngleAxisd(eulerAngleA_deg[2] * M_PI / 180.0, Eigen::Vector3d::UnitZ());
            rotationMatrixB = Eigen::AngleAxisd(eulerAngleB_deg[0] * M_PI / 180.0, Eigen::Vector3d::UnitX())
                              * Eigen::AngleAxisd(eulerAngleB_deg[1] * M_PI / 180.0, Eigen::Vector3d::UnitY())
                              * Eigen::AngleAxisd(eulerAngleB_deg[2] * M_PI / 180.0, Eigen::Vector3d::UnitZ());

            // 旋转欧拉角并输出结果
            Eigen::Vector3d eulerAngleA_rotated =
                    (rotationMatrixA * rotationMatrix).eulerAngles(0, 1, 2) * 180.0 / M_PI;
            Eigen::Vector3d eulerAngleB_rotated =
                    (rotationMatrixB * rotationMatrix).eulerAngles(0, 1, 2) * 180.0 / M_PI;
            if (fabs(normal1[2]) > fabs(normal2[2])) {
                std::vector<double> pt1;
                pt1 = {ends_[0][0], ends_[0][1], ends_[0][2], eulerAngleA_rotated[0], eulerAngleA_rotated[1],
                       eulerAngleA_rotated[2]};
                std::vector<double> pt2;
                pt2 = {ends_[1][0], ends_[1][1], ends_[1][2], eulerAngleB_rotated[0], eulerAngleB_rotated[1],
                       eulerAngleB_rotated[2]};
                welding_points_ = {pt1, pt2};
            } else {
                std::vector<double> pt1;
                std::vector<double> pt2;
                pt1 = {ends_[1][0], ends_[1][1], ends_[1][2], eulerAngleA_rotated[0], eulerAngleA_rotated[1],
                       eulerAngleA_rotated[2]};
                pt2 = {ends_[0][0], ends_[0][1], ends_[0][2], eulerAngleB_rotated[0], eulerAngleB_rotated[1],
                       eulerAngleB_rotated[2]};
                welding_points_ = {pt1, pt2};
            }

        } else {
            //vertical
            Eigen::Vector3d welding_vector;
            welding_vector = (normal1 + normal2) / 2;
            Eigen::Vector3d golden_vector2(0.00260663, 0.48484533, 0.51956243);
            // rotation
            Eigen::Matrix3d rotationMatrix;
            rotationMatrix = Eigen::Quaterniond::FromTwoVectors(golden_vector2, welding_vector).toRotationMatrix();

            // 已知的欧拉角（单位为度）
            double eulerAngleA_deg[3] = {175.0, 40.0, 50.0};
            double eulerAngleB_deg[3] = {175.0, 40.0, 130.0};

            // 将欧拉角转换为旋转矩阵
            Eigen::Matrix3d rotationMatrixA;
            Eigen::Matrix3d rotationMatrixB;
            rotationMatrixA = Eigen::AngleAxisd(eulerAngleA_deg[0] * M_PI / 180.0, Eigen::Vector3d::UnitX())
                              * Eigen::AngleAxisd(eulerAngleA_deg[1] * M_PI / 180.0, Eigen::Vector3d::UnitY())
                              * Eigen::AngleAxisd(eulerAngleA_deg[2] * M_PI / 180.0, Eigen::Vector3d::UnitZ());
            rotationMatrixB = Eigen::AngleAxisd(eulerAngleB_deg[0] * M_PI / 180.0, Eigen::Vector3d::UnitX())
                              * Eigen::AngleAxisd(eulerAngleB_deg[1] * M_PI / 180.0, Eigen::Vector3d::UnitY())
                              * Eigen::AngleAxisd(eulerAngleB_deg[2] * M_PI / 180.0, Eigen::Vector3d::UnitZ());

            // 旋转欧拉角并输出结果
            Eigen::Vector3d eulerAngleA_rotated =
                    (rotationMatrixA * rotationMatrix).eulerAngles(0, 1, 2) * 180.0 / M_PI;
            Eigen::Vector3d eulerAngleB_rotated =
                    (rotationMatrixB * rotationMatrix).eulerAngles(0, 1, 2) * 180.0 / M_PI;
            if (fabs(normal1[2]) > fabs(normal2[2])) {
                std::vector<double> pt1;
                pt1 = {ends_[0][0], ends_[0][1], ends_[0][2], eulerAngleA_rotated[0], eulerAngleA_rotated[1],
                       eulerAngleA_rotated[2]};
                std::vector<double> pt2;
                pt2 = {ends_[1][0], ends_[1][1], ends_[1][2], eulerAngleB_rotated[0], eulerAngleB_rotated[1],
                       eulerAngleB_rotated[2]};
                welding_points_ = {pt1, pt2};
            } else {
                std::vector<double> pt1;
                std::vector<double> pt2;
                pt1 = {ends_[1][0], ends_[1][1], ends_[1][2], eulerAngleA_rotated[0], eulerAngleA_rotated[1],
                       eulerAngleA_rotated[2]};
                pt2 = {ends_[0][0], ends_[0][1], ends_[0][2], eulerAngleB_rotated[0], eulerAngleB_rotated[1],
                       eulerAngleB_rotated[2]};
                welding_points_ = {pt1, pt2};
            }
        }
    }
public:

    WeldLineRec(std::vector<Sviz_Plane> two_plane, std::vector<Eigen::Vector3d> ends){
        two_plane_ = two_plane;
        ends_ = ends;
        compute_welding_angle();
    }
    std::vector<Sviz_Plane> get_two_plane() const {
            return two_plane_;
    }
    std::vector<Eigen::Vector3d> get_ends() const{
        return ends_;
    }
    std::vector<std::vector<double>> get_welding_points() const{
        return welding_points_;
    }


};
// 点云到平面的距离
// 点云到直线的距离
// 点云到直线的投影
// 计算焊接角度
void printVector(const std::vector<std::vector<double>>& data) {
    for (const auto& row : data) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}


int main() {
    // 加载点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/td/project/svision_c/pcd/非标2-一体.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return -1;
    }
    // bbox
    Eigen::Vector4f min_pt(-1.0, -1.0, -1.0, 1.0); // 包围框最小点的坐标
    Eigen::Vector4f max_pt(1.0, 1.0, 1.0, 1.0);    // 包围框最大点的坐标
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud = extractPointsInBox(cloud, min_pt, max_pt);

    //降样
    float leaf_size = 0.003; // 体素立方体的边长
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled = downsamplePointCloud(cropped_cloud, leaf_size);
    // 计算点云的质心
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cropped_cloud, centroid);
    std::cout << "Centroid: " << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << std::endl;


    // 调用获取所有平面函数
    std::vector<Sviz_Plane> all_plane;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> inlier_cloud_list = get_all_plane(cloud_downsampled, all_plane);
//    for (size_t i=0; i<all_plane.size(); ++i){
//        show_multiple_point_clouds({all_plane[i].plane_pcd_});
//    }
    // 找到并显示相交的平面
    // weld lines
    std::vector<std::vector<Eigen::Vector3d>> welding_point_list;
    std::vector<std::vector<std::vector<double>>> welding_point_list_angle;
    for (size_t i=0; i<all_plane.size(); ++i){
        for (size_t j= i+1; j< all_plane.size(); ++j){
            std::vector<Sviz_Plane> two_plane;
            two_plane = {all_plane[i],all_plane[j]};
            std::vector<Eigen::Vector3d> weld_line ={};
//            show_multiple_point_clouds({all_plane[i].plane_pcd_,all_plane[j].plane_pcd_});
            if(CheckIfTwoPlaneConnect(two_plane,0.01, weld_line)){
            welding_point_list.push_back(weld_line);
            WeldLineRec wl(two_plane,weld_line);
            std::vector<std::vector<double>> wl_with_angle;
            wl_with_angle=wl.get_welding_points();
            printVector(wl_with_angle);
            welding_point_list_angle.push_back(wl_with_angle);
            }
        }
    }
    //print welding line


    // 打印内点点云列表的大小
    std::cout << "Number of plane segments: " << inlier_cloud_list.size() << std::endl;
    // 显示多个点云
    show_multiple_point_clouds(inlier_cloud_list);

    return 0;
}

// 获取所有平面函数定义
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> get_all_plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<Sviz_Plane>& all_plane) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> inlier_cloud_list;
    while (cloud->points.size() >= 1000) {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<float> plane_model;
        segment_once(cloud, inliers, inlier_cloud, plane_model);

        // cluster
        std::vector<PointCloudXYZPtr> clusters ;
        clusters = findLargestCluster(inlier_cloud);
        inlier_cloud = clusters[0];
        PointCloudXYZPtr remains(new pcl::PointCloud<pcl::PointXYZ>);
        remains = clusters[1];

        // 将内点点云保存到列表中
        if (inlier_cloud->points.size ()>1000){
            inlier_cloud_list.push_back(inlier_cloud);
            //test
            Sviz_Plane plane_temp(inlier_cloud,plane_model);
            all_plane.push_back(plane_temp);
//            plane_temp.visualize(true);
        }else {
            break;
        }
        // 更新点云为剩余点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true); // 提取剩余点云
        extract.filter(*remaining_cloud);
        *cloud = *remaining_cloud;
        *cloud += *remains;

    }

    return inlier_cloud_list;
}

// 分割函数定义
void segment_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  pcl::PointIndices::Ptr inliers,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud,
                  std::vector<float> &plane_model) {
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

    plane_model=coefficients->values;
    // filter with plane normal
    // 计算法线
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(0.01); // 设置法线估计的搜索半径
    ne.compute(*cloud_normals);
    // 获取平面法线
    Eigen::Vector3f plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);

    // 比较每个内点的法线与平面法线的相似度，移除不相似的点
    std::vector<int> new_inliers;
    for (size_t i = 0; i < inliers->indices.size(); ++i) {
        int idx = inliers->indices[i];
        Eigen::Vector3f cloud_normal(cloud_normals->points[idx].normal_x,
                                     cloud_normals->points[idx].normal_y,
                                     cloud_normals->points[idx].normal_z);

        // 计算余弦相似度
        float cosine_similarity = plane_normal.dot(cloud_normal) / (plane_normal.norm() * cloud_normal.norm());

        // 如果余弦相似度的绝对值大于等于0.85，则将该点添加到新的内点列表中
        if (std::abs(cosine_similarity) >= 0.85) {
            new_inliers.push_back(idx);
        }
    }

    // 将新的内点列表赋值给 inliers
    inliers->indices = new_inliers;


    // 创建一个提取器
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false); // 提取内点
    extract.filter(*inlier_cloud);
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
