//
// Created by yanhang on 1/18/16.
//

#ifndef RENDERPROJECT_MEANSHIFT_H
#define RENDERPROJECT_MEANSHIFT_H

#include <vector>
#include <memory>
#include <iostream>
#include <cmath>
class KernelBase{
public:
    virtual double evaluate(double distance2) = 0;
};

class GaussianKernel: public KernelBase{
public:
	GaussianKernel(const double h_): h2(h_ * h_){}
    virtual double evaluate(double distance2);
private:
	const double h2;
};

class EpanechnikovKernel: public KernelBase{
public:
	EpanechnikovKernel(const double h_): h2(h_ * h_){}
    virtual double evaluate(double distance2);
private:
	const double h2;
};

class MeanShift {
public:
    enum KernelType{
        GAUSSIAN,
        EPANECHNIKOV
    };
    typedef std::vector<double> Sample;
    typedef std::vector<std::vector<double> > SampleSet;

    MeanShift(const SampleSet&, const double, const KernelType );
    MeanShift(const SampleSet&);

    double euclideanDistance(const Sample& pt1, const Sample& pt2) const;

    void setKernelType(KernelType newType);

    inline void setBandWidth(double newH){h = newH;}
    inline const SampleSet& getCluterCenters() const {return centers;}
    inline const std::vector<int>& getModes() const { return modes;}
	inline size_t getClusterNum()const {return centers.size();}
    int getMode(const int id) const;

    void cluster();
private:
    void sanityCheck();
    bool shiftCenter(Sample& pt);

    const SampleSet& sampleSet;
    double h;
    size_t dim;
	const size_t kNum;

	std::shared_ptr<KernelBase> kernel;
    KernelType kernelType;
    SampleSet centers;
    std::vector<int> modes;

    static double double_max;
    static double epsilon;
	static double stopCriteria;
};


#endif //RENDERPROJECT_MEANSHIFT_H
