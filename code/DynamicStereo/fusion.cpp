//
// Created by Yan Hang on 3/2/16.
//

#include "dynamicstereo.h"
#include "external/QPBO1.4/ELC.h"
#include "external/QPBO1.4/QPBO.h"
using namespace std;
using namespace Eigen;
using namespace cv;

namespace dynamic_stereo{

	void DynamicStereo::fusionMove(Depth &p1, const Depth &p2) {
		//create problem
		ELCReduce::PBF<EnergyType> pbf;
		//formulate
		//unary term
		for(auto i=0; i<width*height; ++i){
			int disp1 = (int)p1.getDepthAtInd(i);
			int disp2 = (int)p2.getDepthAtInd(i);
			CHECK_GE(disp1, 0);
			CHECK_LT(disp1, dispResolution);
			CHECK_GE(disp2, 0);
			CHECK_LT(disp2, dispResolution);
			EnergyType ue1 = MRF_data[dispResolution * i + disp1];
			EnergyType ue2 = MRF_data[dispResolution * i + disp2];
			pbf.AddUnaryTerm(i, ue1, ue2);
		}

		//third order smoothness term
		const EnergyType lamh = (EnergyType)(108 * images.size());
		const EnergyType laml = (EnergyType)(9 * images.size());

		vector<ELCReduce::VID> indices(3);
		vector<EnergyType> SE(8);
		for(auto x=1; x<width-1; ++x) {
			for (auto y = 1; y < height - 1; ++y) {
				//horizontal
				int p = y * width + x - 1;
				int q = y * width + x;
				int r = y * width + x + 1;
				EnergyType lam;
				if(refSeg[p] == refSeg[q] && refSeg[p] == refSeg[r])
					lam = lamh;
				else
					lam = laml;

				SE[0] = (EnergyType)(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
				SE[1] = (EnergyType)(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
				SE[2] = (EnergyType)(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
				SE[3] = (EnergyType)(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
				SE[4] = (EnergyType)(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
				SE[5] = (EnergyType)(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
				SE[6] = (EnergyType)(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
				SE[7] = (EnergyType)(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
				for(auto &S: SE)
					S = S * lam;

				indices[0] = p; indices[1] = q; indices[2] = r;
				pbf.AddHigherTerm(3, indices.data(), SE.data());

				//vertical
				p = (y - 1) * width + x;
				r = (y + 1) * width + x;
				SE[0] = (EnergyType)(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
				SE[1] = (EnergyType)(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
				SE[2] = (EnergyType)(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
				SE[3] = (EnergyType)(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
				SE[4] = (EnergyType)(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
				SE[5] = (EnergyType)(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
				SE[6] = (EnergyType)(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
				SE[7] = (EnergyType)(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
				for(auto &S: SE)
					S = S * lam;

				indices[0] = p; indices[1] = q; indices[2] = r;
				pbf.AddHigherTerm(3, indices.data(), SE.data());
			}
		}

		//reduce
		cout << "Reducing with ELC..." << endl;
		ELCReduce::PBF<EnergyType> qpbf;
		printf("Number of variables: %d\n", pbf.maxID());
		pbf.reduceHigher();
		pbf.toQuadratic(qpbf, width * height);
		int numVar = qpbf.maxID();
		printf("Done. number of variables:%d (ori %d)\n", numVar, width * height);
		printf("Convering to QPBO object...\n");

		kolmogorov::qpbo::QPBO<EnergyType> qpbo(numVar, numVar * 4);
		qpbf.convert(qpbo, numVar);

		printf("Number of nodes in qpbo: %d\n", qpbo.GetNodeNum());

		printf("Done\n");
		//solve
		cout << "Solving..." << endl << flush;
		float t = (float)getTickCount();
		qpbo.MergeParallelEdges();
		qpbo.Solve();
		qpbo.ComputeWeakPersistencies();
		t = ((float)getTickCount() - t) / (float)getTickFrequency();
		printf("Done. Time usage:%.3f\n", t);


		//fusion
		float unlabeled = 0.0;
		for(auto i=0; i<width * height; ++i){
			int l = qpbo.GetLabel(i);
			int disp1 = (int)p1.getDepthAtInd(i);
			int disp2 = (int)p2.getDepthAtInd(i);
			if(l == 0)
				p1.setDepthAtInd(i, disp1);
			else if(l < 0) {
				p1.setDepthAtInd(i, disp1);
				unlabeled += 1.0;
			}
			else
				p1.setDepthAtInd(i, disp2);
		}

		printf("Unlabeled pixels: %.2f, Ratio: %.2f\n", unlabeled, unlabeled / (float)(width * height));
	}

}//namespace dynamic_stereo

