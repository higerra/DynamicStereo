//
// Created by yanhang on 1/18/16.
//

#include "meanshift.h"

using namespace std;
double GaussianKernel::evaluate(double distance2) {
	double dis2 = distance2 / h2;
	return std::exp(-0.5 * dis2);
}

double EpanechnikovKernel::evaluate(double distance2) {
	double dis2 = distance2 / h2;
	if(dis2 >= 0 && dis2 <= 1)
		return 1 - dis2;
	else
		return 0;
}

double MeanShift::double_max = 999999.9;
double MeanShift::epsilon = 0.0000001;
double MeanShift::stopCriteria = 0.0001;

void MeanShift::sanityCheck(){
    if(sampleSet.empty())
        throw runtime_error("MeanShift::MeanShift(): sampleSet is empty");

    dim = sampleSet[0].size();
    if(dim == 0)
        throw runtime_error("MeanShift::MeanShift(): Data dimension cannot be 0");

    for(const auto& sample: sampleSet)
        if(sample.size() != (size_t)dim)
            throw runtime_error("MeanShift::MeanShift(): samples have different dimensions");
}

MeanShift::MeanShift(const ::MeanShift::SampleSet & sampleSet_):
        sampleSet(sampleSet_), kNum(sampleSet.size()){
	setKernelType(GAUSSIAN);
    sanityCheck();
}


MeanShift::MeanShift(const SampleSet &sampleSet_, const double h_, const KernelType kernelType_):
        sampleSet(sampleSet_), h(h_), kNum(sampleSet.size()), kernelType(kernelType_){
	setKernelType(kernelType_);
    sanityCheck();
}

int MeanShift::getMode(const int id) const{
    if(id >= (int)modes.size())
        throw runtime_error("MeanShift::getMode(): id out of range");

    return modes[id];
}

double MeanShift::euclideanDistance(const Sample &pt1, const Sample &pt2)const {
    if(pt1.size() != pt2.size())
        throw runtime_error("MeanShift::euclideanDistance(): dimensions don't match");
    double res = 0.0;
    for(size_t i=0; i<pt1.size(); ++i)
        res += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);
    return res;
}

void MeanShift::setKernelType(KernelType newType){
	kernelType = newType;
	kernel.reset();
	switch(kernelType){
		case GAUSSIAN:
			kernel = shared_ptr<KernelBase>(new GaussianKernel(h));
			break;
		case EPANECHNIKOV:
			kernel = shared_ptr<KernelBase>(new EpanechnikovKernel(h));
			break;
		default:
			throw runtime_error("Unrecognized kernel type");
	}
}

bool MeanShift::shiftCenter(Sample& pt) {
	double total_weight = 0.0;
	Sample pt_ori = pt;
	for(auto& p: pt)
		p = 0.0;
	for(const auto& s: sampleSet){
		double weight = kernel->evaluate(euclideanDistance(s, pt_ori));
		for(auto i=0; i<dim; ++i)
			pt[i] += weight * s[i];
		total_weight += weight;
	}

	if(total_weight <= epsilon)
		return true;

	//converge?
	double offset_norm = 0.0;
	for(auto i=0; i<dim; ++i)
		offset_norm += pt[i] * pt_ori[i];
	offset_norm = std::sqrt(offset_norm);
	if(offset_norm <= stopCriteria)
		return true;
	return false;
}

void MeanShift::cluster() {
	centers.clear();
	centers = sampleSet;
	for (auto &center: centers) {
		while (true) {
			if (shiftCenter(center))
				break;
		}
	}

	//find mode
	modes.resize((size_t) kNum);
	for (auto &m: modes)
		m = -1;
	int idx = 0;

	SampleSet centers_;
	double radius2 = 0.01 * h * h;
	for (size_t i = 0; i < kNum; ++i) {
		if (modes[i] >= 0)
			continue;
		Sample curcenter(dim, 0.0);
		for(auto k=0; k<dim; ++k)
			curcenter[k] += centers[i][k];
		double weight = 1.0;
		modes[i] = idx++;
		for (size_t j = 0; j < kNum; ++j) {
			if (modes[j] >= 0)
				continue;
			double dis = std::sqrt(euclideanDistance(centers[i], centers[j]));
			if(dis <= radius2) {
				modes[j] = idx;
				for(auto k=0; k<dim; ++k)
					curcenter[k] += centers[j][k];
				weight += 1.0;
			}
		}
		for(auto& v: curcenter)
			v /= weight;
		centers_.push_back(curcenter);
	}
	centers.swap(centers_);
}
