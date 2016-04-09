//
// Created by yanhang on 4/7/16.
//

#include "dynamic_confidence.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {
	void DynamicConfidence::init() {
		cout << "Reading reconstruction" << endl;
		CHECK(theia::ReadReconstruction(file_io.getReconstruction(), &reconstruction))
		<< "Can not open reconstruction file";
		CHECK_EQ(reconstruction.NumViews(), file_io.getTotalNum());
		const vector<theia::ViewId> &vids = reconstruction.ViewIds();
		orderedId.resize(vids.size());
		for (auto i = 0; i < vids.size(); ++i) {
			const theia::View *v = reconstruction.View(vids[i]);
			std::string nstr = v->Name().substr(5, 5);
			int idx = atoi(nstr.c_str());
			orderedId[i] = IdPair(idx, vids[i]);
		}
		std::sort(orderedId.begin(), orderedId.end(),
		          [](const std::pair<int, theia::ViewId> &v1, const std::pair<int, theia::ViewId> &v2) {
			          return v1.first < v2.first;
		          });
	}

	void DynamicConfidence::run(const int anchor, Depth &confidence) {
		const int framenum = file_io.getTotalNum();
		CHECK_LT(anchor, framenum);
		const int startid = anchor - max_tWindow / 2 >= 0 ? anchor - max_tWindow / 2 : 0;
		const int endid = anchor + max_tWindow / 2 < framenum ? anchor + max_tWindow / 2 : framenum - 1;

		char buffer[1024] = {};

		vector<FlowFrame> flow_forward((size_t) endid - startid + 1);
		vector<FlowFrame> flow_backward((size_t) endid - startid + 1);
		printf("=====================\nFrame: %d\n", anchor);
		cout << "Reading optical flow..." << endl;
		for (auto i = startid; i <= endid; ++i) {
			cout << '.' << flush;
			if (i < file_io.getTotalNum() - 1)
				flow_forward[i - startid].readFlowFile(file_io.getOpticalFlow_forward(i));
			if (i > 0)
				flow_backward[i - startid].readFlowFile(file_io.getOpticalFlow_backward(i));
		}
		cout << endl;
		const int width = (int)(flow_forward[0].width() / downsample);
		const int height = (int)(flow_forward[0].height() / downsample);
		const int widthFull = flow_forward[0].width();
		const int heightFull = flow_forward[0].height();

		Depth confidence_down(width, height, -1);
		confidence.initialize(widthFull, heightFull, 0);

		const int min_interval = 0;
		const double kth_ratio = 0.8;
		const size_t min_length = 5;
		double min_depth, max_depth;
		cout << "Computing min-max depth" << endl;
		computeMinMaxDepth(anchor, min_depth, max_depth);

		const int id = anchor - startid;
		const theia::Camera& refCam = reconstruction.View(orderedId[anchor].second)->Camera();

		cout << "Computing confidence..." << endl;
		const int unit = width * height / 10;
		const int testx = 1596 / downsample;
		const int testy = 472 / downsample;

		int startx = 0, endx = width-1, starty = 0, endy = height-1;
		if(testx >=0 && testy>=0){
			printf("Debug mode: %d, %d\n", testx, testy);
			startx = testx;
			endx = testx;
			starty = testy;
			endy = testy;
		}

		double max_line_length = 100;
		for (auto y = starty; y<= endy; ++y) {
			for (auto x = startx; x <= endx; ++x) {
				if((y*width+x) % unit == 0)
					cout << '.' << flush;
				vector<double> epiErr;
				Vector2d locL((double) x, (double) y);
				Vector3d ray = refCam.PixelToUnitDepthRay(locL * downsample);
				Vector3d minpt = refCam.GetPosition() + ray * min_depth;
				Vector3d maxpt = refCam.GetPosition() + ray * max_depth;
				if(testx >= 0 && testy >= 0){
					printf("min depth: %.3f, max depth: %.3f\n", min_depth, max_depth);
				}

				for (auto i = 0; i < flow_forward.size(); ++i) {
					const theia::Camera& cam2 = reconstruction.View(orderedId[i+startid].second)->Camera();
					if(i == id){
						Mat img = imread(file_io.getImage(i+startid));
						cv::circle(img, cv::Point(locL[0], locL[1]), 2, cv::Scalar(0,0,255), 2);
						sprintf(buffer, "%s/temp/conf_ref%05d_%05d.jpg", file_io.getDirectory().c_str(), anchor, i+startid);
						imwrite(buffer, img);
						continue;
					}
					if (std::abs(id - i) < min_interval)
						continue;

					Vector2d locR;
					if (id < i) {
						if (!flow_util::trackPoint(locL * downsample, flow_forward, id, i, locR))
							continue;
					} else {
						if (!flow_util::trackPoint(locL * downsample, flow_backward, id, i, locR))
							continue;
					}

					Vector2d spt, ept;
					cam2.ProjectPoint(minpt.homogeneous(), &spt);
					cam2.ProjectPoint(maxpt.homogeneous(), &ept);
//					Vector2d dir = spt - ept;
//					dir.normalize();
//					spt = ept + dir * max_line_length;

					if(x == testx && y == testy){
						Mat img = imread(file_io.getImage(i+startid));
						cv::circle(img, cv::Point(locR[0], locR[1]), 2, cv::Scalar(0,0,255), 2);
						printf("---------------------\nFrame %d, spt:(%.2f,%.2f), ept:(%.2f,%.2f)\n", i+startid, spt[0], spt[1], ept[0], ept[1]);
						cv::line(img, cv::Point(spt[0], spt[1]), cv::Point(ept[0], ept[1]), cv::Scalar(255,0,0), 2);
						sprintf(buffer, "%s/temp/conf_ref%05d_%05d.jpg", file_io.getDirectory().c_str(), anchor, i+startid);
						imwrite(buffer, img);

						theia::Matrix3x4d pMatrix;
						cam2.GetProjectionMatrix(&pMatrix);
						cout << "Projection matrix:" << endl << pMatrix << endl;
					}

					epiErr.push_back(geometry_util::distanceToLineSegment<2>(locR, spt, ept));
				}
				if (epiErr.size() < min_length) {
					confidence(x, y) =  0.0;
					continue;
				}
				const size_t kth = (size_t) (epiErr.size() * kth_ratio);
				nth_element(epiErr.begin(), epiErr.begin() + kth, epiErr.end());
				confidence_down.setDepthAtInt(x, y, epiErr[kth]);
			}
		}
		cout << endl;

		//upsample to original resolution
		for(auto x=0; x<widthFull-downsample; ++x){
			for(auto y=0; y<heightFull-downsample; ++y)
				confidence(x,y) = confidence_down.getDepthAt(Vector2d((double)x/downsample, (double)y/downsample));
		}

		confidence.updateStatics();
	}


	void DynamicConfidence::computeMinMaxDepth(const int anchor, double& min_depth, double& max_depth) const{
		const theia::View *anchorView = reconstruction.View(orderedId[anchor].second);
		const theia::Camera cam = anchorView->Camera();
		vector<theia::TrackId> trackIds = anchorView->TrackIds();
		printf("number of tracks:%lu\n", trackIds.size());
		vector<double> depths;
		for (const auto tid: trackIds) {
			const theia::Track *t = reconstruction.Track(tid);
			Vector4d spacePt = t->Point();
			Vector2d imgpt;
			double curdepth = cam.ProjectPoint(spacePt, &imgpt);
			if (curdepth > 0)
				depths.push_back(curdepth);
		}
		//ignore furthest 1% and nearest 1% points
		const double lowRatio = 0.01;
		const double highRatio = 0.99;
		const size_t lowKth = (size_t) (lowRatio * depths.size());
		const size_t highKth = (size_t) (highRatio * depths.size());
		//min_disp should be correspond to high depth
		nth_element(depths.begin(), depths.begin() + lowKth, depths.end());
		min_depth = depths[lowKth];
		nth_element(depths.begin(), depths.begin() + highKth, depths.end());
		max_depth = depths[highKth];
	}
}//namespace dynamic_stereo