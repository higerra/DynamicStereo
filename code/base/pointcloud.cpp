#include "pointcloud.h"

#include <pcl/features/normal_3d.h>
#include <fstream>
#include <algorithm>
#include "frame.h"

using namespace std;
using namespace Eigen;

namespace dynamic_rendering{
    bool PointCloud::initFromFile(const string &filename){
	ifstream plyin(filename.c_str());
	if(!plyin.is_open()){
	    plyin.close();
	    return false;
	}
	string temp;
	float v1,v2,v3;
	int ptnum;
	bool hascolor = false;
	has_normal = false;
	while(plyin >> temp){
	    if(temp == "vertex")
		break;
	}
	plyin >> ptnum;
	while(true){
	    plyin >> temp;
	    if(temp=="red")
		hascolor = true;
	    if(temp=="nx")
		has_normal = true;
	    if(temp=="end_header")
		break;
	}
	
	cloud.width = ptnum;
	cloud.height = 1;
	cloud.is_dense = false;
	cloud.points.resize(ptnum);
	assert((hascolor&&has_normal) || (!hascolor && !has_normal));
	for(size_t i=0; i<cloud.points.size(); i++){
	    plyin >> v1 >> v2 >> v3;
	    cloud.points[i].x = v1;
	    cloud.points[i].y = v2;
	    cloud.points[i].z = v3;
	    if(hascolor && has_normal){
		int c1,c2,c3;
		float nx, ny, nz, cur;
		float time;
		plyin >> nx >> ny >> nz >> c1 >> c2 >> c3 >> cur >> time;
		cloud.points[i].r = (uchar)c1;
		cloud.points[i].g = (uchar)c2;
		cloud.points[i].b = (uchar)c3;
		cloud.points[i].normal_x = nx;
		cloud.points[i].normal_y = ny;
		cloud.points[i].normal_z = nz;
		cloud.points[i].curvature = cur;
		cloud.points[i].time = time;
		
	    }else{
		cloud.points[i].r = 200;
		cloud.points[i].g = 200;
		cloud.points[i].b = 200;
		cloud.points[i].normal_x = 0;
		cloud.points[i].normal_y = 0;
		cloud.points[i].normal_z = 0;
		cloud.points[i].curvature = 0;
		cloud.points[i].time = 0.0;
	    }
	    
	}
	plyin.close();

	if(!has_normal){
	    estimateNormal();
	}
	return true;
    }

    PointCloud::PointCloud(const string &filename){
	initFromFile(filename);
    }

    void PointCloud::savePLYFile(const string &filename){
	pcl::io::savePLYFileASCII(filename.c_str(), cloud);
    }

    void PointCloud::mergePointCloud(const PointCloud& newpc){
	for(int i=0; i<newpc.getPointSize(); i++){
	    addPoint(newpc.getPosition(i), newpc.getColor(i), newpc.getNormal(i), newpc.getTime(i));
	}
    }

    void PointCloud::estimateNormal(){
	pcl::NormalEstimation<PointT, PointT> norm_est;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	norm_est.setSearchMethod(tree);
	norm_est.setKSearch(30);

	norm_est.setInputCloud(cloud.makeShared());
//	norm_est.compute(*points_with_normals_src);
	norm_est.compute(cloud);
//	pcl::copyPointCloud(*points_with_normals_src, *curcloudptr);
	has_normal = true;
    }

    void PointCloud::setTime(float timestamp){
	for(int i=0; i<getPointSize(); i++){
	    cloud.points[i].time = timestamp;
	}
    }

    void PointCloud::setTime(int ind, float timestamp){
	assert(ind < getPointSize());
	cloud.points[ind].time = timestamp;
    }
  
    void PointCloud::removeNoise(const double radius, const int min_num){
	pcl::PointCloud<PointT>::Ptr cloudptr = cloud.makeShared();
	pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    
	pcl::RadiusOutlierRemoval<PointT> outrem;
	outrem.setInputCloud(cloudptr);
	outrem.setRadiusSearch(radius);
	outrem.setMinNeighborsInRadius (min_num);
	outrem.filter(*filtered);
	cloud.swap(*filtered);
    }

    void PointCloud::downSample(const double radius){
	pcl::PointCloud<PointT>::Ptr cloudptr = cloud.makeShared();
	pcl::PointCloud<PointT>::Ptr downsampled(new pcl::PointCloud<PointT>);

	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(cloudptr);
	sor.setLeafSize(radius, radius, radius);
	sor.filter(*downsampled);
	cloud.swap(*downsampled);
    }

    //non linear ICP, code mostly from http://pointclouds.org/documentation/tutorials/pairwise_incremental_registration.php
    Matrix4d PointCloud::registerTo(const PointCloud&target, const int max_iter){
	assert(this->hasNormal() && target.hasNormal());
	pcl::PointCloud<PointT>::Ptr curcloudptr = cloud.makeShared();
	pcl::PointCloud<PointT>::Ptr targetptr = target.getPointCloud().makeShared();
	pcl::PointCloud<PointT> result;

	// pcl::IterativeClosestPoint<PointT, PointT> icp;
	// icp.setInputCloud(curcloudptr);
	// icp.setInputTarget(targetptr);
	// icp.align(result);
	

//	pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_src (new pcl::PointCloud<pcl::PointNormal>);
//	pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_tgt (new pcl::PointCloud<pcl::PointNormal>);

	MyPointRepresentation point_representation;
	float alpha[4] = {1.0,1.0,1.0,1.0};
	point_representation.setRescaleValues(alpha);

	pcl::IterativeClosestPointNonLinear<PointT, PointT> reg;
	reg.setTransformationEpsilon(1e-6);
	reg.setMaxCorrespondenceDistance(0.1);
	reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
	reg.setUseReciprocalCorrespondences(true);
	reg.setInputSource(curcloudptr);
	reg.setInputTarget(targetptr);

	Matrix4f Ti = Matrix4f::Identity(), prev;
	pcl::PointCloud<PointT>::Ptr reg_result = curcloudptr;
	reg.setMaximumIterations(10);
	for(int i=0; i<max_iter; i++){
	    curcloudptr = reg_result;
	    reg.setInputSource(curcloudptr);
	    reg.align(*reg_result);
	    
	    Ti = reg.getFinalTransformation() * Ti;
	    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
		reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
    
	    prev = reg.getLastIncrementalTransformation ();
	}

	pcl::transformPointCloud( cloud, result, Ti);
	cloud.swap(result);

	Matrix4d transform;
	for (int i=0; i<16; i++)
	    transform(i) = Ti(i);
	return transform;
    }

    Matrix4d PointCloud::robustAlignTo(const PointCloud& target){
	// pcl::PointCloud<PointT>::Ptr curcloudptr = cloud.makeShared();
	// pcl::PointCloud<PointT>::Ptr targetptr = target.getPointCloud().makeShared();

	// pcl::PointCloud<PointNormalT>::Ptr src_with_normals (new pcl::PointCloud<PointNormalT>);
	// pcl::PointCloud<PointNormalT>::Ptr tgt_with_normals (new pcl::PointCloud<PointNormalT>);
	// pcl::PointCloud<PointNormalT> result;
	
	// pcl::NormalEstimation<PointT, PointNormalT> norm_est;
	// pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	// norm_est.setSearchMethod(tree);
	// norm_est.setKSearch(30);

	// norm_est.setInputCloud(curcloudptr);
	// norm_est.compute(*src_with_normals);
	// norm_est.setInputCloud(targetptr);
	// norm_est.compute(*tgt_with_normals);

	// pcl::copyPointCloud(*curcloudptr, *src_with_normals);
	// pcl::copyPointCloud(*targetptr, *tgt_with_normals);

	// //Estimate Features
	// pcl::PointCloud<FeatureT>::Ptr src_features(new pcl::PointCloud<FeatureT>);
	// pcl::PointCloud<FeatureT>::Ptr tgt_features(new pcl::PointCloud<FeatureT>);
	// FeatureEstimationT fest;
	// fest.setRadiusSearch(0.025);
	// fest.setInputCloud(src_with_normals);
	// fest.setInputNormals(src_with_normals);
	// fest.compute(*src_features);
	// fest.setInputCloud(tgt_with_normals);
	// fest.setInputNormals(tgt_with_normals);
	// fest.compute(*tgt_features);

	// //Perform alignment
	// pcl::SampleConsensusPrerejective<PointNormalT, PointNormalT, FeatureT> align;
	// align.setInputSource(src_with_normals);
	// align.setSourceFeatures(src_features);
	// align.setInputTarget(tgt_with_normals);
	// align.setTargetFeatures(tgt_features);
	// align.setMaximumIterations(50000);
	// align.setNumberOfSamples(3);
	// align.setCorrespondenceRandomness(5);
	// align.setSimilarityThreshold(0.8f);
	// align.setMaxCorrespondenceDistance(0.01);
	// align.setInlierFraction(0.25f);

	// align.align(result);

	 Matrix4d transform = Matrix4d::Identity();
	// if(true){
	//     transform = align.getFinalTransformation().cast<double>();
	//     pcl::copyPointCloud(result, cloud);
	// }
	 return transform;
    }


  void PointCloud::transform(const Matrix4d &trans){
    Matrix4f trans_f;
    trans_f<<trans(0,0), trans(0,1), trans(0,2), trans(0,3),
      trans(1,0), trans(1,1), trans(1,2), trans(1,3),
      trans(2,0), trans(2,1), trans(2,2), trans(2,3),
      trans(3,0), trans(3,1), trans(3,2), trans(3,3);
    pcl::PointCloud<PointT> transformed_cloud;
    pcl::transformPointCloud(cloud, transformed_cloud, trans_f);
    cloud.swap(transformed_cloud);
  }


    void assignColorToPointCloud(const std::vector<Frame>&frames, PointCloud& pc){
	const int frame_margin = 15;
	PointCloud newpc;
	int bestind = -1;
	float pttime_pre = 0.0;
	for(size_t i=0; i<pc.getPointSize(); i++){
	    pc.setColor(i,Vector3d(200,200,200));
	    const Vector3d& pos = pc.getPosition(i);
	    float curpttime = pc.getTime(i);
	    if(curpttime != pttime_pre){
	    	float mindiff = 99999;
	    	for(int frameid=0; frameid < frames.size(); frameid++){
	    	    if(abs(frames[frameid].getTime() - curpttime) < mindiff){
	    		bestind = frameid;
	    		mindiff = abs(frames[frameid].getTime() - curpttime);
	    	    }
	    	}
	    }
	    int lower = bestind - frame_margin >= 0 ? bestind - frame_margin : 0;
	    int upper = bestind + frame_margin < frames.size() ? bestind + frame_margin : frames.size() - 1;
	    bool isassigned = false;
	    for(int frameid=lower; frameid <= upper; frameid++){
		Vector2d imgpt = frames[frameid].getCamera().projectToImage(pos);
		if(frames[frameid].isVisible(pos) && frames[frameid].isValidRGB(imgpt)){
		    Vector3d ptcolor = frames[frameid].getColor(imgpt);
		    swap(ptcolor[0], ptcolor[2]);
		    newpc.addPoint(pos, ptcolor, pc.getNormal(i));
		    isassigned = true;
		    break;
		}
	    }
	    // if(isassigned)
	    // 	continue;
	    // for(int frameid=bestind+1; frameid<=upper; frameid++){
	    // 	Vector2d imgpt = frames[frameid].getCamera().projectToImage(pos);
	    // 	if(frames[frameid].isVisible(pos) && frames[frameid].isValidRGB(imgpt)){
	    // 	    newpc.addPoint(pos, frames[frameid].getColor(imgpt));
	    // 	    break;
	    // 	}
	    // }
	}
	pc.getPointCloud_nonConst().swap(newpc.getPointCloud_nonConst());
    }
    
    void createDepth(const vector<PointCloud>& pc,
		     std::vector<Frame>&frames,
		     const vector<float>& timeline_PC,
		     bool is_fillhole){
	if(frames.size() == 0)
	    return;
	Vector2d depthDim = frames[0].RGBToDepth(Vector2d(frames[0].getWidth(), frames[0].getHeight()));
	const int pc_margin = 5;
	
	cout<<"Depth size:"<<depthDim[0]<<' '<<depthDim[1]<<endl;
	for(int i=0; i<frames.size(); i++){
	    cout<<"Computing depth for frame "<<i<<endl<<flush;
	    frames[i].getDepth_nonConst().initialize(depthDim[0], depthDim[1]);
	    int bestind = 0;
	    float min_timediff = 999999;
	    for(int pcid=0; pcid<pc.size(); pcid++){
		float curtime_diff = std::abs(frames[i].getTime() - timeline_PC[pcid]);
		if(curtime_diff < min_timediff){
		    min_timediff = curtime_diff;
		    bestind = pcid;
		}
	    }

	    int startid = bestind - pc_margin >= 0? bestind - pc_margin : 0;
	    int endid = bestind + pc_margin < pc.size() ? bestind + pc_margin : pc.size()-1;
	    int count = 0;

	    for(int pcid=startid; pcid <= endid; pcid++){
		for(int ptid=0; ptid<pc[pcid].getPointSize(); ptid++){
		    Vector3d curpos = pc[pcid].getPosition(ptid);
		    Vector3d curlocalpos = frames[i].getCamera().transformToLocal(curpos);
		    Vector4d imgpt = frames[i].getCamera().getIntrinsic() * Vector4d(curlocalpos[0], curlocalpos[1], curlocalpos[2], 1.0);
		    if(imgpt[2] != 0){
			imgpt[0] /= imgpt[2];
			imgpt[1] /= imgpt[2];
		    }
		    Vector2d depthpix = frames[i].RGBToDepth(Vector2d(imgpt[0], imgpt[1]));
		    
		    if(frames[i].isValidDepth(depthpix)){
			count++;
			double curdepth = curlocalpos[2];
			double exdepth = frames[i].getDepth().getDepthAtInt(round(depthpix[0]), round(depthpix[1]));
			if(curdepth > 0){
			    if(exdepth < 0 || (curdepth < exdepth)){
				frames[i].getDepth_nonConst().setDepthAtInt(round(depthpix[0]), round(depthpix[1]), curdepth);
			    }
			}
		    }
		}
	    }
	    
	    frames[i].getDepth_nonConst().updateStatics();
	    cout<<"Frame "<<i<<' '<<"valid count:"<<count<<flush<<endl;
	    if(is_fillhole)
		frames[i].getDepth_nonConst().fillhole();
	}
    }

    void findCorrespondences(const PointCloud& pc1,const PointCloud& pc2,
			     vector<pair<Vector3d, Vector3d> >&cor,
			     const int corres_num){
	cor.clear();
	PointCloud pc1_2 = pc1;
//	PointCloud pc2_2 = pc2;
	pc1_2.registerTo(pc2,10);

	pcl::PointCloud<PointT>::Ptr pc1_ptr = pc1_2.getPointCloud().makeShared();
	pcl::PointCloud<PointT>::Ptr pc2_ptr = pc2.getPointCloud().makeShared();
	pcl::registration::CorrespondenceEstimation<PointT, PointT> est;
	est.setInputSource(pc1_ptr);
	est.setInputTarget(pc2_ptr);

	pcl::Correspondences all_correspondences;
	est.determineReciprocalCorrespondences(all_correspondences);
	vector<pair<float, int> >sort_array;
	for(int i=0; i<all_correspondences.size(); i++)
	    sort_array.push_back(pair<float,int>(all_correspondences[i].distance, i));
	std::sort(sort_array.begin(), sort_array.end());
	
	for(int i=0; i<corres_num && i<all_correspondences.size(); i++){
	    int ind1 = all_correspondences[sort_array[i].second].index_query;
	    int ind2 = all_correspondences[sort_array[i].second].index_match;
//	    pc1_2.setColor(ind1, Vector3d(255,0,0));
//	    pc2_2.setColor(ind2, Vector3d(255,0,0));
	    pair<Vector3d, Vector3d> curpair = pair<Vector3d, Vector3d>(pc2.getPosition(ind2), pc1.getPosition(ind1));
	    cor.push_back(curpair);
	}
	// string path1 = "temp1.ply";
	// string path2 = "temp2.ply";
	// pc1_2.savePLYFile(path1);
	// pc2_2.savePLYFile(path2);
    }

}










