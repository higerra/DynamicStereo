#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/impl/radius_outlier_removal.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/registration/icp.h>
#include <pcl/registration/impl/icp.hpp>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/impl/icp_nl.hpp>
//#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/features/fpfh.h>
#include <pcl/features/impl/fpfh.hpp>
#include <pcl/io/ply_io.h>

#include <Eigen/Core>

struct PointXYZRGBNormalTime{
	PCL_ADD_POINT4D;
	PCL_ADD_NORMAL4D;
	PCL_ADD_RGB;
	float curvature;
	float time;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBNormalTime,
								  (float, x, x)
										  (float, y, y)
										  (float, z, z)
										  (float, normal_x, normal_x)
										  (float, normal_y, normal_y)
										  (float, normal_z, normal_z)
										  (float, rgb, rgb)
										  (float, curvature, curvature)
										  (float, time, time))

typedef PointXYZRGBNormalTime PointT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimation<PointT,PointT, FeatureT> FeatureEstimationT;

namespace dynamic_rendering{
	class Frame;

	struct OFFfile{
		std::vector<Eigen::Vector3d>points;
		std::vector<Eigen::Vector3d>colors;
		std::vector<Eigen::Vector3i>faces;
		inline void addPoint(const Eigen::Vector3d& pt){
			points.push_back(pt);
			colors.push_back(Eigen::Vector3d(200,200,200));
		}
		inline void addPoint(const Eigen::Vector3d& pt, const Eigen::Vector3d& color){
			points.push_back(pt);
			colors.push_back(color);
		}
		inline void addFace (const Eigen::Vector3d&pt1,
							 const Eigen::Vector3d&pt2,
							 const Eigen::Vector3d&pt3){
			addPoint(pt1); addPoint(pt2); addPoint(pt3);
			faces.push_back(Eigen::Vector3i(points.size()-3, points.size()-2, points.size()-1));
		}
		void saveOFF(const std::string &filename){
			std::ofstream fout(filename.c_str());
			fout<<"COFF"<<std::endl;
			fout<<points.size()<<' '<<faces.size()<<" 0"<<std::endl;
			for (int i=0; i<points.size(); i++){
				fout<<points[i][0]<<' '<<points[i][1]<<' '<<points[i][2]<<' ';
				fout<<colors[i][0]<<' '<<colors[i][1]<<' '<<colors[i][2]<<std::endl;
			}
			for (int i=0; i<faces.size(); i++)
				fout<<"3 "<<faces[i][0]<<' '<<faces[i][1]<<' '<<faces[i][2]<<std::endl;
			fout.close();
		}
	};

	class MyPointRepresentation: public pcl::PointRepresentation<PointT>{
		using pcl::PointRepresentation<PointT>::nr_dimensions_;
	public:
		MyPointRepresentation(){
			nr_dimensions_ = 4;
		}
		virtual void copyToFloatArray(const PointT &p, float *out)const{
			out[0] = p.x;
			out[1] = p.y;
			out[2] = p.z;
			out[3] = p.curvature;
		}
	};


	class PointCloud{
	public:
		PointCloud(const std::string &filename);
		PointCloud(): has_normal(true){
			cloud.width = 0;
			cloud.height = 1;
			cloud.points.clear();
		}

		void setTime(float timestamp);
		void setTime(int ind, float timestamp);

		bool initFromFile(const std::string &filename);
		void savePLYFile(const std::string &filename);
		inline void setColor(int id, const Eigen::Vector3d& color){
			assert(id < cloud.points.size());
			cloud.points[id].r = (uint8_t)color[0];
			cloud.points[id].g = (uint8_t)color[1];
			cloud.points[id].b = (uint8_t)color[2];
		}
		inline Eigen::Vector3d getColor(int id) const{
			assert(id < cloud.points.size());
			return Eigen::Vector3d((double)cloud.points[id].r, (double)cloud.points[id].g, (double)cloud.points[id].b);
		}
		inline Eigen::Vector3d getPosition(int id) const{
			assert(id < cloud.points.size());
			return Eigen::Vector3d(cloud.points[id].x, cloud.points[id].y, cloud.points[id].z);
		}
		inline Eigen::Vector3d getNormal(int id)const{
			assert(id < cloud.points.size() && has_normal);
			return Eigen::Vector3d(cloud.points[id].normal_x, cloud.points[id].normal_y, cloud.points[id].normal_z);
		}

		inline float getTime(int id) const{
			assert(id < cloud.points.size());
			return cloud.points[id].time;
		}

		inline bool hasNormal() const{
			return has_normal;
		}

		inline pcl::PointCloud<PointT>& getPointCloud_nonConst(){
			return cloud;
		}
		inline const pcl::PointCloud<PointT>& getPointCloud()const{
			return cloud;
		}


		inline void addPoint(const Eigen::Vector3d& pos, const Eigen::Vector3d& color, const Eigen::Vector3d& normal, const float timestamp = 0.0){
			PointT newpt;
			newpt.x = pos[0]; newpt.y = pos[1]; newpt.z = pos[2];
			newpt.r = color[0]; newpt.g = color[1]; newpt.b = color[2];
			newpt.normal_x = normal[0]; newpt.normal_y = normal[1]; newpt.normal_z = normal[2];
			newpt.time = timestamp;
			cloud.points.push_back(newpt);
			cloud.width++;
		}

		void mergePointCloud(const PointCloud& newpc);

		inline size_t getPointSize() const{
			return cloud.points.size();
		}

		void estimateNormal();

		//remove outlier with radius outlier remover from pcl
		void removeNoise(const double radius, const int min_num);
		void downSample(const double radius);

		Eigen::Matrix4d registerTo(const PointCloud&target, const int max_iter = 50);
		Eigen::Matrix4d robustAlignTo(const PointCloud& target);
		void transform(const Eigen::Matrix4d& trans);

	private:
		pcl::PointCloud<PointT> cloud;
		bool has_normal;
	};

	void assignColorToPointCloud(const std::vector<Frame>&frames, PointCloud& pc);
	void createDepth(const std::vector<PointCloud>& pc, std::vector<Frame>&frames,
			 const std::vector<float>& timeline_PC,
			 bool is_fillhole = false);
	void findCorrespondences(const PointCloud& pc1,const PointCloud& pc2,
							 std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> >&cor,
							 const int corres_num = 100);
}


#endif






