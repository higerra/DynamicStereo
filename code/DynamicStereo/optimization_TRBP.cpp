//
// Created by yanhang on 3/3/16.
//
#include <map>
#include "optimization.h"
#include "external/segment_ms/msImageProcessor.h"
#include <opengm/functions/sparsemarray.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

    SecondOrderOptimizeTRBP::SecondOrderOptimizeTRBP(const FileIO& file_io_, const int kFrames_, shared_ptr<StereoModel<EnergyType> > model_):
	StereoOptimization(file_io_, kFrames_, model_), trun(4){
//        segment_ms::msImageProcessor ms_segmentator;
//	    const Mat& image = model->image;
//        ms_segmentator.DefineBgImage(image.data, segment_ms::COLOR, image.rows, image.cols);
//        const int hs = 4;
//        const float hr = 5.0f;
//        const int min_a = 40;
//        ms_segmentator.Segment(hs, hr, min_a, meanshift::SpeedUpLevel::MED_SPEEDUP);
//        refSeg.resize((size_t)image.cols * image.rows);
//        const int * labels = ms_segmentator.GetLabels();
//        for(auto i=0; i<image.cols * image.rows; ++i)
//            refSeg[i] = labels[i];
//
//        laml = 9 * kFrames;
//        lamh = 108 * kFrames;
    }

    void SecondOrderOptimizeTRBP::optimize(Depth &result, const int max_iter) const {
        char buffer[1024] = {};
	Vector4i ROI(0,0,width, height);
	ROI[0] = 1451 / (int)model->downsample;
	ROI[1] = 337 / (int)model->downsample;
	ROI[2] = 1904 / (int)model->downsample;
	ROI[3] = 479 / (int)model->downsample;

	const int roiW = ROI[2] - ROI[0];
	const int roiH = ROI[3] - ROI[1];

        //formulate problem with OpenGM
        typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
        typedef opengm::SparseFunction<double, size_t, size_t> SparseFunction;
//        typedef opengm::GraphicalModel<double, opengm::Adder,
//                opengm::ExplicitFunction<double>, Space> GraphicalModel;
	typedef opengm::GraphicalModel<double, opengm::Adder,
				       OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>, SparseFunction), Space> GraphicalModel;
        typedef opengm::TrbpUpdateRules<GraphicalModel, opengm::Minimizer> UpdateRules;
        typedef opengm::MessagePassing<GraphicalModel, opengm::Minimizer, UpdateRules, opengm::MaxDistance> TRBP;
        //typedef opengm::GraphCut<GraphicalModel, opengm::Minimizer

	const int& nLabel = model->nLabel;
	const int kVar = roiW * roiH;
	printf("Number of variables: %d\n", kVar);

        Space space((size_t)(kVar), (size_t)nLabel);
        GraphicalModel gm(space);
        //add unary terms
        size_t shape[] = {(size_t) nLabel, (size_t) nLabel, (size_t)nLabel};
	int varIdx = 0;

	vector<int> idxMap(width * height);

	for(auto y=ROI[1]; y<ROI[3]; ++y){
	    for(auto x=ROI[0]; x<ROI[2]; ++x, ++varIdx){
		const int& i = y * width + x;
		opengm::ExplicitFunction<double> f(shape, shape + 1);
		for (auto l = 0; l < nLabel; ++l)
		    f(l) = model->operator()(i, l) / model->MRFRatio;
		GraphicalModel::FunctionIdentifier fid = gm.addFunction(f);
		size_t vid[] = {(size_t) varIdx};
		gm.addFactor(fid, vid, vid + 1);
		idxMap[i] = varIdx;
	    }
	}

        //add triple terms
        cout << "Adding triple terms" << endl << flush;

	const double ws = 0.01;

        SparseFunction fv(shape, shape+3, 0.0);
        int count = 0;
        for (int l0 = 0; l0 < nLabel; ++l0) {
            for (int l1 = 0; l1 < nLabel; ++l1) {
                for (int l2 = 0; l2 < nLabel; ++l2) {
                    int labelDiff = std::abs(l0 + l2 - 2 * l1);
                    size_t coord[] = {(size_t)l0, (size_t)l1, (size_t)l2};
                    if (labelDiff <= 2000){
			fv.insert(coord, (double)labelDiff * ws);
                        count++;
                    }
                }
            }
        }
        GraphicalModel::FunctionIdentifier fidtriple = gm.addFunction(fv);
        cout << "Non zero count: " << count << endl << flush;

	for(auto y=ROI[1]; y<ROI[3]; ++y){
	    for(auto x=ROI[0]; x<ROI[2]; ++x){
		if(x > ROI[0] && x < ROI[2] - 1) {
		    size_t vIndxH[] = {(size_t) idxMap[y * width + x - 1], (size_t)idxMap[y * width + x], (size_t)idxMap[y * width + x + 1]};
		    gm.addFactor(fidtriple, vIndxH, vIndxH+3);
		}
		if(y>ROI[1] && y < ROI[3] - 1) {
		    size_t vIndxV[] = {(size_t) idxMap[(y - 1) * width + x], (size_t) idxMap[y * width + x],
				       (size_t) idxMap[(y + 1) * width + x]};
		    gm.addFactor(fidtriple, vIndxV, vIndxV+3);
		}
	    }
	}


        //solve
        const double converge_bound = 1e-5;
        const double damping = 0.0;
        TRBP::Parameter parameter(max_iter);
        TRBP trbp(gm, parameter);
        cout << "Solving with TRBP..." << endl << flush;
        float t = (float)getTickCount();
        trbp.infer();
        t = ((float)getTickCount() - t) / (float)getTickFrequency();
        double finalEnergy = trbp.value();
        printf("Done. Final energy: %.3f, Time usage: %.2fs\n", finalEnergy, t);

        vector<size_t> labels;
        trbp.arg(labels);
        CHECK_EQ(labels.size(), kVar);
        result.initialize(width, height, -1);
        for(auto y=ROI[1]; y<ROI[3]; ++y){
	    for(auto x=ROI[0]; x<ROI[2]; ++x){
		CHECK_LT(idxMap[y*width+x], labels.size());
		result(x,y) = labels[idxMap[y*width+x]];
	    }
	}
    }

    double SecondOrderOptimizeTRBP::evaluateEnergy(const Depth& disp) const {
        return 0.0;
    }

    void toyTripleTRBP(){
	printf("Running toy TRBP with triple term...\n");
	const int dim = 3;
	const int nLabel = 3;
	typedef double EnergyTypeT;
	//vector<int> vars(dim * dim,0.0);
	vector<int>vars{0,2,2,0,2,2,0,2,2};
	vector<EnergyTypeT> dCost{0,0.5,0.5, 0.2,0.2,0, 0.5,0.5,0,
		0,0.5,0.5, 0.2,0.2,0, 0.5,0.5,0,
		0,0.5,0.5, 0.2,0.2,0, 0.5,0.5,0};
//        vector<EnergyTypeT> dCost{0,0.1,0.1, 0.1,0,0.1, 0.1,0.1,0,
//                                  0,0.1,0.1, 0.1,0,0.1, 0.1,0.1,0,
//                                  0,0.1,0.1, 0.1,0,0.1, 0.1,0.1,0};

//        vector<EnergyTypeT> dCost(dim*dim*nLabel);
//        std::default_random_engine generator;
//        std::uniform_real_distribution<double> distribution;
//        for(auto i=0; i<dCost.size(); ++i)
//            dCost[i] = distribution(generator);

	CHECK_EQ(dCost.size(), dim*dim*nLabel);

	auto lapE = [](double l1, double l2, double l3){
	    return std::abs(l2 * 2 - l1 - l3);
	};

	CHECK_EQ(lapE(0,1,2), 0.0);

	CHECK_EQ(lapE(0,2,2), 2.0);

	const double ws = 1;

	auto evaluateEnergy = [&](const std::vector<int>& result){
	    CHECK_EQ(result.size(), dim*dim);
	    double e = 0.0;
	    for(auto i=0; i<dim*dim; ++i)
		e += dCost[i*nLabel+result[i]];
	    for(auto y=0; y<dim; ++y){
		for(auto x=0; x<dim; ++x){
		    if(x > 0 && x < dim-1)
			e += lapE(result[y*dim+x-1], result[y*dim+x], result[y*dim+x+1]) * ws;
		    if(y > 0 && y < dim-1)
			e += lapE(result[(y-1)*dim+x], result[y*dim+x], result[(y+1)*dim+x]) * ws;
		}
	    }
	    return e;
	};

	typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
	typedef opengm::SparseFunction<EnergyTypeT, size_t, size_t> SparseFunction;
	typedef opengm::GraphicalModel<EnergyTypeT, opengm::Adder,
				       opengm::ExplicitFunction<EnergyTypeT>, Space> GraphicalModel;
	typedef opengm::TrbpUpdateRules<GraphicalModel, opengm::Minimizer> UpdateRules;
	typedef opengm::MessagePassing<GraphicalModel, opengm::Minimizer, UpdateRules, opengm::MaxDistance> TRBP;

	//typedef opengm::GraphCut<GraphicalModel, opengm::Minimizer
	Space space((size_t)(dim * dim), (size_t)nLabel);
	GraphicalModel gm(space);

	size_t shape[] = {(size_t) nLabel, (size_t) nLabel, (size_t)nLabel};
	opengm::ExplicitFunction<EnergyTypeT> fv(shape, shape+3);
	int count = 0;

	for (int l0 = 0; l0 < nLabel; ++l0) {
	    for (int l1 = 0; l1 < nLabel; ++l1) {
		for (int l2 = 0; l2 < nLabel; ++l2) {
		    size_t coord[] = {(size_t)l0, (size_t)l1, (size_t)l2};
		    fv((size_t)l0,(size_t)l1,(size_t)l2) = lapE(l0,l1,l2) * ws;
		    count++;
		}
	    }
	}
	GraphicalModel::FunctionIdentifier ftriple = gm.addFunction(fv);

	//add unary terms
	for (auto i = 0; i < dim * dim; ++i) {
	    opengm::ExplicitFunction<EnergyTypeT> f(shape, shape + 1);
	    for (auto l = 0; l < nLabel; ++l)
		f(l) = dCost[i*nLabel+l];
	    GraphicalModel::FunctionIdentifier fid = gm.addFunction(f);
	    size_t vid[] = {(size_t) i};
	    gm.addFactor(fid, vid, vid + 1);
	}

	//add triple terms
	for(auto y=0; y<dim; ++y){
	    for(auto x=0; x<dim; ++x){
		size_t vIndxV[] = {(size_t) (y - 1) * dim + x, (size_t) y * dim + x, (size_t) (y + 1) * dim + x};
		size_t vIndxH[] = {(size_t) y * dim + x - 1, (size_t) y * dim + x, (size_t) y * dim + x + 1};
		if(x > 0 && x < dim - 1) {
		    gm.addFactor(ftriple, vIndxH, vIndxH + 3);
		}
		if(y > 0 && y < dim - 1) {
		    gm.addFactor(ftriple, vIndxV, vIndxV + 3);
		}
	    }
	}

	const double converge_bound = 1e-7;
	const double damping = 0.0;
	TRBP::Parameter parameter(2000);
	TRBP trbp(gm, parameter);
	cout << "Solving with TRBP..." << endl << flush;
	float t = (float)getTickCount();
	trbp.infer();
	t = ((float)getTickCount() - t) / (float)getTickFrequency();
	EnergyTypeT finalEnergy = trbp.value();


	vector<size_t> result;
	trbp.arg(result);
	for(auto i=0; i<dim*dim; ++i)
	    vars[i] = (int)result[i];

	printf("Done. Final energy: %.3f (%.3f by TRBP), Time usage: %.2fs\n", evaluateEnergy(vars), finalEnergy, t);

	printf("Optimized result:\n");
	for(auto y=0; y<dim; ++y){
	    for(auto x=0; x<dim; ++x)
		cout << vars[y*dim+x] << ' ';
	    cout << endl;
	}

	vector<int> gtvars{0,1,2,0,1,2,0,1,2};
	printf("Ground true energy: %.2f\n", evaluateEnergy(gtvars));
    }


}//namespace dynamic_stereo
