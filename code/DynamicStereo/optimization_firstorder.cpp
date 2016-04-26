//
// Created by yanhang on 3/3/16.
//

#include "optimization.h"
#include "external/MRF2.2/mrf.h"
#include "external/MRF2.2/GCoptimization.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {

	FirstOrderOptimize::FirstOrderOptimize(const FileIO &file_io_, const int kFrames_,
	                                       shared_ptr<StereoModel<EnergyType> > model_) :
			StereoOptimization(file_io_, kFrames_, model_) { }


	void FirstOrderOptimize::optimize(Depth &result, const int max_iter) const {
		const int nLabel = model->nLabel;
		DataCost *dataCost = new DataCost(const_cast<EnergyType *>(model->unary.data()));
		SmoothnessCost *smoothnessCost = new SmoothnessCost(1, 10, model->weight_smooth * model->MRFRatio,
		                                                    const_cast<EnergyType *>(model->hCue.data()),
		                                                    const_cast<EnergyType *>(model->vCue.data()));
		EnergyFunction *energy_function = new EnergyFunction(dataCost, smoothnessCost);
		shared_ptr<MRF> mrf(new Expansion(width, height, nLabel, energy_function));
		mrf->initialize();

		//randomly initialize
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(0, nLabel - 1);
		for (auto i = 0; i < width * height; ++i)
			mrf->setLabel(i, distribution(generator));

		float initDataEnergy = (float) mrf->dataEnergy() / model->MRFRatio;
		float initSmoothEnergy = (float) mrf->smoothnessEnergy() / model->MRFRatio;
		float t;
		mrf->optimize(max_iter, t);
		float finalDataEnergy = (float) mrf->dataEnergy() / model->MRFRatio;
		float finalSmoothEnergy = (float) mrf->smoothnessEnergy() / model->MRFRatio;

		printf("Graph cut finished.\nInitial energy: (%.3f, %.3f, %.3f)\nFinal energy: (%.3f,%.3f,%.3f)\nTime usage: %.2fs\n",
		       initDataEnergy, initSmoothEnergy, initDataEnergy + initSmoothEnergy,
		       finalDataEnergy, finalSmoothEnergy, finalDataEnergy + finalSmoothEnergy, t);

		result.initialize(width, height, -1);
		for (auto i = 0; i < width * height; ++i)
			result.setDepthAtInd(i, mrf->getLabel(i));

		delete dataCost;
		delete smoothnessCost;
		delete energy_function;
	}

	double FirstOrderOptimize::evaluateEnergy(const Depth &disp) const {
		return 0.0;
	}

}//namespace dynamic_stereo