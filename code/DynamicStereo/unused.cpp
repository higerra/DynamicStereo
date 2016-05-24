//
// Created by yanhang on 4/4/16.
//


//dynamicstereo.cpp
//runstereo()
//		{
//			//test for GridWarpping
//			vector<Mat> fullImg(images.size());
//			for (auto i = 0; i < fullImg.size(); ++i)
//				fullImg[i] = imread(file_io.getImage(i + offset));
//			GridWarpping gridWarpping(file_io, anchor, fullImg, *model, reconstruction, orderedId,
//			                          depth_firstOrder_filtered, downsample, offset);
//
//			printf("================\nTesting for bilinear coefficience\n");
//			Vector2d testP(500, 500);
//			Vector4i testInd;
//			Vector4d testW;
//			gridWarpping.getGridIndAndWeight(testP, testInd, testW);
//			printf("(%d,%d,%d,%d), (%.2f,%.2f,%.2f,%.2f)\n", testInd[0], testInd[1], testInd[2], testInd[3],
//			       testW[0], testW[1], testW[2], testW[3]);
//
//			Mat mask;
//			sprintf(buffer, "%s/mask%05d.jpg", file_io.getDirectory().c_str(), anchor);
//			mask = imread(buffer);
//			CHECK(mask.data);
//			CHECK_EQ(mask.cols, fullImg[0].cols);
//			CHECK_EQ(mask.rows, fullImg[0].rows);
//			cvtColor(mask, mask, CV_RGB2GRAY);
//
//			for (auto i = 0; i < fullImg.size(); ++i) {
//				printf("=================\nWarpping frame %d\n", i);
//				vector<Vector2d> refPt, srcPt;
//				const int testF = i;
//				printf("Computing point correspondence...\n");
//				//gridWarpping.computePointCorrespondence(testF, refPt, srcPt);
//				gridWarpping.computePointCorrespondenceNoWarp(testF, refPt, srcPt);
//				CHECK_EQ(refPt.size(), srcPt.size());
//
//				printf("Done, correspondence: %d\n", (int) refPt.size());
//
//
//
//				Mat warpped = fullImg[anchor-offset].clone();
//				const theia::Camera &refCam = reconstruction.View(orderedId[anchor].second)->Camera();
//				const theia::Camera &srcCam = reconstruction.View(orderedId[testF + offset].second)->Camera();
//				for (auto y = downsample; y < warpped.rows - downsample; ++y) {
//					for (auto x = downsample; x < warpped.cols - downsample; ++x) {
//						if(mask.at<uchar>(y,x) < 200)
//							continue;
//						double d = depth_firstOrder_filtered.getDepthAt(
//								Vector2d((double) x / (double) downsample, (double) y / (double) downsample));
//						Vector3d ray = refCam.PixelToUnitDepthRay(Vector2d(x, y));
//						Vector3d spt = refCam.GetPosition() + d * ray;
//						Vector2d imgpt;
//						srcCam.ProjectPoint(spt.homogeneous(), &imgpt);
//						if (imgpt[0] < 0 || imgpt[0] > warpped.cols - 1 || imgpt[1] < 0 || imgpt[1] > warpped.rows - 1)
//							continue;
//						Vector3d pix = interpolation_util::bilinear<uchar, 3>(fullImg[testF].data, warpped.cols,
//						                                                      warpped.rows, imgpt);
//						warpped.at<Vec3b>(y, x) = Vec3b((uchar) pix[0], (uchar) pix[1], (uchar) pix[2]);
//					}
//				}
//				sprintf(buffer, "%s/temp/stereo%05d.jpg", file_io.getDirectory().c_str(), testF);
//				imwrite(buffer, warpped);
//				Mat grayRef, graySrc, colorRef, colorSrc;
//				colorSrc = fullImg[testF].clone();
//				colorRef = fullImg[anchor-offset].clone();
////				cvtColor(fullImg[anchor - offset], grayRef, CV_RGB2GRAY);
////				cvtColor(warpped, graySrc, CV_RGB2GRAY);
////
////				cvtColor(grayRef, colorRef, CV_GRAY2RGB);
////				cvtColor(graySrc, colorSrc, CV_GRAY2RGB);
//
////				for (auto x = 0; x < fullImg[0].cols; x += gridWarpping.getBlockW()) {
////					cv::line_util(colorRef, cv::Point(x, 0), cv::Point(x, fullImg[0].rows - 1), Scalar(255, 255, 255), 1);
////					cv::line_util(colorSrc, cv::Point(x, 0), cv::Point(x, fullImg[0].rows - 1), Scalar(255, 255, 255), 1);
////				}
////				for (auto y = 0; y < fullImg[0].rows; y += gridWarpping.getBlockH()) {
////					cv::line_util(colorRef, cv::Point(0, y), cv::Point(fullImg[0].cols - 1, y), Scalar(255, 255, 255), 1);
////					cv::line_util(colorSrc, cv::Point(0, y), cv::Point(fullImg[0].cols - 1, y), Scalar(255, 255, 255), 1);
////				}
//				Mat tgtImg = fullImg[testF].clone();
//				drawKeyPoints(tgtImg, srcPt);
//				sprintf(buffer, "%s/temp/trackOnTgt%05d.jpg", file_io.getDirectory().c_str(), testF);
//				imwrite(buffer, tgtImg);
//
//				Mat refImg = fullImg[anchor-offset].clone();
//				drawKeyPoints(refImg, refPt);
//				sprintf(buffer, "%s/temp/trackOnRef%05d.jpg", file_io.getDirectory().c_str(), anchor-offset);
//				imwrite(buffer, refImg);
//
//				drawKeyPoints(colorRef, refPt);
//
//				Mat stabled, vis;
//				Mat comb;
//				gridWarpping.computeWarppingField(testF, refPt, srcPt, fullImg[testF], stabled, vis, true);
//
////				hconcat(stabled, vis, comb);
////				sprintf(buffer, "%s/temp/sta_%05dimg1.jpg", file_io.getDirectory().c_str(), testF);
////				imwrite(buffer, colorRef);
////				sprintf(buffer, "%s/temp/sta_%05dimg2.jpg", file_io.getDirectory().c_str(), testF + offset);
////				imwrite(buffer, colorSrc);
////				sprintf(buffer, "%s/temp/sta_%05dimg3.jpg", file_io.getDirectory().c_str(), testF + offset);
////				imwrite(buffer, colorRef);
////				sprintf(buffer, "%s/temp/sta_%05dimg3.jpg", file_io.getDirectory().c_str(), testF);
////				imwrite(buffer, colorRef);
//
//
//				for(auto y=0; y<stabled.rows; ++y){
//					for(auto x=0; x<stabled.cols; ++x){
//						if(mask.at<uchar>(y,x) < 200)
//							stabled.at<Vec3b>(y,x) = fullImg[anchor-offset].at<Vec3b>(y,x);
//					}
//				}
//
//				sprintf(buffer, "%s/temp/sta_%05dimg4.jpg", file_io.getDirectory().c_str(), testF);
//				imwrite(buffer, stabled);
//
////				sprintf(buffer, "%s/temp/unstabled%05d.jpg", file_io.getDirectory().c_str(), testF + offset);
////				imwrite(buffer, warpped);
////
////				sprintf(buffer, "%s/temp/stabbled%05d.jpg", file_io.getDirectory().c_str(), testF + offset);
////				imwrite(buffer, stabled);
////			sprintf(buffer, "%s/temp/sta_gri%05d.jpg", file_io.getDirectory().c_str(), testF+offset);
////			imwrite(buffer, vis);
////			sprintf(buffer, "%s/temp/sta_com%05d.jpg", file_io.getDirectory().c_str(), testF+offset);
////			imwrite(buffer, comb);
//			}
//		}



//{
////test for PCA
//const int dim = 3;
//vector<double> reprojeEs(dispResolution);
//double minreproE = numeric_limits<double>::max();
//int minreproDisp = 0;
//for (auto testd = 0; testd < dispResolution; ++testd) {
//printf("===============================\nDisparity %d\n", testd);
//vector<vector<double> > patches;
//const theia::Camera &refCam = reconstruction.View(orderedId[anchor].second)->Camera();
//getPatchArray(dbtx / downsample, dbty / downsample, testd, 0, refCam, 0, (int)images.size()-1, patches);
//vector<VectorXd> patch_reduce;
//int fid = anchor - tWindowStereo / 2;
//for (const auto &p: patches) {
//CHECK_EQ(p.size() % dim, 0);
//if (*min_element(p.begin(), p.end()) < 0)
//continue;
//for(auto j=0; j<p.size() / dim; ++j){
//VectorXd pt(dim);
//for(auto k=0; k<dim; ++k)
//pt[k] = p[j*dim+k];
//patch_reduce.push_back(pt);
//}
//}
//
//sprintf(buffer, "%s/temp/matrix%05d_%03d.txt", file_io.getDirectory().c_str(), anchor, testd);
//ofstream fout(buffer);
//CHECK(fout.is_open());
//for(auto i=0; i<patch_reduce.size(); ++i){
//for(auto j=0; j<dim; ++j)
//fout << patch_reduce[i][j] << ' ';
//fout << endl;
//}
//fout.close();
//
//Mat Dm((int) patch_reduce.size(), dim, CV_64FC1);
//for (auto i = 0; i < patch_reduce.size(); ++i) {
//for (auto j = 0; j < dim; ++j)
//Dm.at<double>(i, j) = patch_reduce[i][j];
//}
//cv::PCA pca(Dm, Mat(), CV_PCA_DATA_AS_ROW, 0);
//Mat eigenv = pca.eigenvalues;
//vector<double> ev(dim);
//for (auto i = 0; i < dim; ++i)
//ev[i] = eigenv.at<double>(i, 0);
//
//double ratio = ev[0] / (ev[0] + ev[1] + ev[2]);
//printf("Eigen values: %.3f,%.3f,%.3f. Ratio: %.3f\n", ev[0], ev[1], ev[2], ratio);
//
////compute reprojection error
//cv::PCA pca2(Dm, Mat(), CV_PCA_DATA_AS_ROW, 1);
//double reproE = 0.0;
//for(auto i=0; i<Dm.rows; ++i){
//Mat reprojected = pca2.backProject(pca2.project(Dm.row(i)));
//Mat absd;
//cv::absdiff(Dm.row(i), reprojected, absd);
//const double* pAbsd = (double*)absd.data;
//reproE += sqrt(pAbsd[0]*pAbsd[0]+pAbsd[1]*pAbsd[1]+pAbsd[2]*pAbsd[2]);
//}
//reproE = reproE / (double)Dm.rows;
//printf("Reprojection error: %.3f\n", reproE);
//reprojeEs[testd] = reproE;
//if(reproE < minreproE){
//minreproE = reproE;
//minreproDisp = testd;
//}
//}
//printf("Minimum reprojection error: %.3f, disp: %d\n", minreproE, minreproDisp);
//}
//
//{
////plot the matching cost
//for (auto testd = 0; testd < dispResolution; ++testd) {
//sprintf(buffer, "%s/temp/costpattern%05d_%03d.txt", file_io.getDirectory().c_str(), anchor, testd);
//ofstream fout(buffer);
//CHECK(fout.is_open());
////				printf("===============================\nDisparity %d\n", testd);
//vector<vector<double> > patches;
//const theia::Camera &refCam = reconstruction.View(orderedId[anchor].second)->Camera();
//getPatchArray(dbtx / downsample, dbty / downsample, testd, pR, refCam, 0, (int) images.size() - 1,
//patches);
//vector<double> mCost;
//int startid = 999, endid = 0;
//int refId = (int) patches.size() / 2;
//for (auto i = 0; i < patches.size(); ++i) {
//if (*min_element(patches[i].begin(), patches[i].end()) < 0)
//continue;
//startid = std::min(startid, i);
//endid = std::max(endid, i);
//}
//for (auto i = startid; i <= endid; ++i) {
//double ssd = 0.0;
//for (auto j = 0; j < patches[i].size(); ++j) {
//ssd += (patches[refId][j] - patches[i][j]) * (patches[refId][j] - patches[i][j]);
//}
//mCost.push_back(ssd);
//}
//for (auto v: mCost)
//fout << v << ' ';
//fout << endl;
//fout.close();
//}
//}







/////////////////////////////////////////////////////////////////////////////
//Different inference
/////////////////////////////////////////////////////////////////////////////
//		cout << "Solving with second order smoothness (trbp)..." << endl;
//		SecondOrderOptimizeTRBP optimizer_trbp(file_io, (int)images.size(), model);
//		Depth result_trbp, depth_trbp;
//		optimizer_trbp.optimize(result_trbp, 10);
//		disparityToDepth(result_trbp, depth_trbp);
//
//		sprintf(buffer, "%s/temp/result%05d_trbp_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
//		result_trbp.saveImage(buffer, 255.0 / (double)dispResolution);
//		sprintf(buffer, "%s/temp/mesh%05d_trbp.ply", file_io.getDirectory().c_str(), anchor);
//		utility::saveDepthAsPly(string(buffer), depth_trbp, images[anchor-offset], reconstruction.View(orderedId[anchor].second)->Camera(), downsample);


//		cout << "Solving with second order smoothness (fusion move)..." << endl;
//		SecondOrderOptimizeFusionMove optimizer_fusion(file_io, images.size(), model, dispUnary);
//		const vector<int>& refSeg = optimizer_fusion.getRefSeg();
//		Mat segImg;
//		utility::visualizeSegmentation(refSeg, width, height, segImg);
//		sprintf(buffer, "%s/temp/refSeg%.5d.jpg", file_io.getDirectory().c_str(), anchor);
//		imwrite(buffer, segImg);
//		Depth result_fusion;
//		optimizer_fusion.optimize(result_fusion, 300);
//
//		sprintf(buffer, "%s/temp/result%05d_fusionmove_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
//		result_fusion.saveImage(buffer, 255.0 / (double)dispResolution);
//		printf("Saving depth to point cloud...\n");
//		Depth depth_fusion;
//		disparityToDepth(result_fusion, depth_fusion);
//		sprintf(buffer, "%s/temp/mesh_fusion_b%05d.ply", file_io.getDirectory().c_str(), anchor);
//		utility::saveDepthAsPly(string(buffer), depth_fusion, images[anchor-offset], reconstruction.View(anchor)->Camera(), downsample);
//		warpToAnchor(result_fusion, "fusion");

//		cout << "Solving with second order smoothness (TRWS)..." << endl;
//		SecondOrderOptimizeTRWS optimizer_TRWS(file_io, (int)images.size(), model);
//		Depth result_TRWS;
//		optimizer_TRWS.optimize(result_TRWS, 1);
//
//		sprintf(buffer, "%s/temp/result%05d_TRWS_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
//		result_TRWS.saveImage(buffer, 255.0 / (double)dispResolution);
//		warpToAnchor(result_TRWS, "TRWS");


//////////////////////////////////////////////////////
//Used in DynamicSegment
//////////////////////////////////////////////////////
//		vector<Mat> intensityRaw(warppedImg.size());
//		for(auto i=0; i<warppedImg.size(); ++i)
//			cvtColor(warppedImg[i], intensityRaw[i], CV_BGR2GRAY);
//
//		auto isInside = [&](int x, int y){
//			return x>=0 && y >= 0 && x < width && y < height;
//		};
//
//		vector<vector<double> > intensity((size_t)width * height);
//		for(auto & i: intensity)
//			i.resize(warppedImg.size(), 0.0);
//
//		const int pR = 0;
//
//		//box filter with invalid pixel handling
//		for(auto i=0; i<warppedImg.size(); ++i) {
//			printf("frame %d\n", i+offset);
//			for (auto y = 0; y < height; ++y) {
//				for (auto x = 0; x < width; ++x) {
//					double curI = 0.0;
//					double count = 0.0;
//					for (auto dx = -1 * pR; dx <= pR; ++dx) {
//						for (auto dy = -1 * pR; dy <= pR; ++dy) {
//							const int curx = x + dx;
//							const int cury = y + dy;
//							if(isInside(curx, cury)){
//								uchar gv = intensityRaw[i].at<uchar>(cury,curx);
//								if(gv == (uchar)0)
//									continue;
//								count = count + 1.0;
//								curI += (double)gv;
//							}
//						}
//					}
//					if(count < 1)
//						continue;
//					intensity[y*width+x][i] = curI / count;
//				}
//			}
//		}
//
////		for(auto i=0; i<warppedImg.size(); ++i){
////			sprintf(buffer, "%s/temp/patternb%05d_%05d.txt", file_io.getDirectory().c_str(), anchor, i+offset);
////			ofstream fout(buffer);
////			CHECK(fout.is_open());
////			for(auto y=0; y<height; ++y){
////				for(auto x=0; x<width; ++x)
////					fout << intensity[y*width+x][i] << ' ';
////				//fout << colorDiff[y*width+x][i] << ' ';
////				fout << endl;
////			}
////			fout.close();
////		}
//
//
//		//brightness confidence dynamicness confidence
//		Depth brightness(width, height, 0.0);
//		for(auto y=0; y<height; ++y){
//			for(auto x=0; x<width; ++x){
//				vector<double>& pixIntensity = intensity[y*width+x];
//				CHECK_GT(pixIntensity.size(), 0);
//				double count = 0.0;
//				//take median as brightness
//				const size_t kth = pixIntensity.size() / 2;
//				nth_element(pixIntensity.begin(), pixIntensity.begin()+kth, pixIntensity.end());
//				brightness(x,y) = pixIntensity[kth];
//
//				double averageIntensity = 0.0;
//				for(auto i=0; i<pixIntensity.size(); ++i){
//					if(pixIntensity[i] > 0){
//						//brightness(x,y) += pixIntensity[i];
//						averageIntensity += pixIntensity[i];
//						count += 1.0;
//					}
//				}
//				if(count < 2){
//					continue;
//				}
////				averageIntensity /= count;
////				for(auto i=0; i<pixIntensity.size(); ++i){
////					if(pixIntensity[i] > 0)
////						dynamicness(x,y) += (pixIntensity[i] - averageIntensity) * (pixIntensity[i] - averageIntensity);
////				}
////				if(dynamicness(x,y) > 0)
////					dynamicness(x,y) = std::sqrt(dynamicness(x,y)/(count - 1));
//			}
//		}
//






///////////////////////////////////////////////////////////////////////////////////
//gaussian model
//int br = std::max(stats.at<int>(l,CC_STAT_WIDTH)/2, stats.at<int>(l,CC_STAT_HEIGHT)/2);
//printf("Init br: %d\n", br);
//while(br < 500){
//double nStatic = 0.0;
//for(auto x=cx-br; x<=cx+br; ++x){
//for(auto y=cy-br; y<=cy+br; ++y){
//if(x >= 0 && x < width && y >= 0 && y < height) {
//if (staticCan.at<uchar>(x, y) > 200)
//nStatic += 1.0;
//}
//}
//}
//if(nStatic > min_multi * area)
//break;
//br += 5;
//}
//printf("br:%d\n", br);
//
////estimate foreground histogram and background histogram
//vector<Vec3b> nsample;
//vector<vector<Vec3b> > psample(input.size());
//
//for(auto x=cx-br; x<=cx+br; ++x){
//for(auto y=cy-br; y<=cy+br; ++y) {
//if (x >= 0 && y >= 0 && x < width && y < height) {
//if (labels.at<int>(y,x) == l) {
//for (auto v = 0; v < input.size(); ++v) {
//Vec3b pix = input[v].at<Vec3b>(y,x);
//if(pix != Vec3b(0,0,0))
//psample[v].push_back(pix);
//}
//}
//if (staticCan.at<uchar>(y, x) > 200) {
////for (auto v = 0; v < input.size(); ++v)
//for(auto v=-1*fR; v<=fR; ++v) {
//Vec3b pix = input[anchor - offset + v].at<Vec3b>(y, x);
//if(pix != Vec3b(0,0,0))
//nsample.push_back(pix);
//}
//}
//}
//}
//}
//if(nsample.size() < min_nSample)
//continue;
//
//if(l == testL){
//for(auto v=-1*fR; v<=fR; ++v) {
//Mat tempMat = input[anchor - offset+v].clone();
//for (auto y = 0; y < height; ++y) {
//for (auto x = 0; x < width; ++x) {
//if (labels.at<int>(y, x) == l)
//tempMat.at<Vec3b>(y, x) = tempMat.at<Vec3b>(y, x) * 0.4 + Vec3b(0, 0, 255) * 0.6;
//}
//}
//cv::rectangle(tempMat, cv::Point(cx - br, cy - br), cv::Point(cx + br, cy + br),
//        cv::Scalar(0, 0, 255));
//sprintf(buffer, "%s/temp/sample_region%05d_com%d_f%05d.jpg", file_io.getDirectory().c_str(), anchor, l, anchor-offset+v);
//imwrite(buffer, tempMat);
//}
//
//printf("Dummping out samples...\n");
//sprintf(buffer, "%s/temp/sample_train%05d_com%05d.txt", file_io.getDirectory().c_str(), anchor, l);
//ofstream fout(buffer);
//CHECK(fout.is_open());
//for(auto i=0; i<nsample.size(); ++i)
//fout << (int)nsample[i][0] << ' ' << (int)nsample[i][1] << ' ' << (int)nsample[i][2] << endl;
//fout.close();
//sprintf(buffer, "%s/temp/sample_test%05d_com%05d.txt", file_io.getDirectory().c_str(), anchor, l);
//fout.open(buffer);
//CHECK(fout.is_open());
//for(auto i=0; i<psample.size(); ++i)
//for(auto j=0; j<psample[i].size(); ++j)
//fout << (int)psample[i][j][0] << ' ' << (int)psample[i][j][1] << ' ' << (int)psample[i][j][2] << endl;
//fout.close();
//}
//
//Ptr<cv::ml::EM> gmmbg = cv::ml::EM::create();
//gmmbg->setClustersNumber(kComponent);
//Mat nsampleMat((int)nsample.size(), 3, CV_64F);
//for(auto i=0; i<nsample.size(); ++i){
//nsampleMat.at<double>(i,0) = nsample[i][0];
//nsampleMat.at<double>(i,1) = nsample[i][1];
//nsampleMat.at<double>(i,2) = nsample[i][2];
//}
//printf("Training local background color model, number of samples:%d...\n", nsampleMat.rows);
//gmmbg->trainEM(nsampleMat);
//printf("Done.\n");
//Mat means = gmmbg->getMeans();
//Mat weights = gmmbg->getWeights();
//const double* pGmmWeights = (double*) weights.data;
//printf("Means of component gaussian models:\n");
//for(auto i=0; i<means.rows; ++i){
//printf("(%.2f,%.2f,%.2f)\n", means.at<double>(i,0), means.at<double>(i,1), means.at<double>(i,2));
//}
//printf("Weights of components:\n");
//for(auto i=0; i<gmmbg->getClustersNumber(); ++i)
//printf("%.2f ", pGmmWeights[i]);
//printf("\n");
//
//if(l == testL){
//Mat toysample(1,3,CV_64F, cv::Scalar::all((uchar)100));
//Mat toyProb(1,gmmbg->getClustersNumber(), CV_64F, cv::Scalar::all(0));
//Vec2d toyRes = gmmbg->predict2(toysample, toyProb);
//printf("nLog for toy example: %.3f\n", -1 * toyRes[0]);
//}
//
//vector<double> pnLogs(psample.size());
//for(auto i=0; i<psample.size(); ++i){
//double pbg = 0.0;
//for(auto j=0; j<psample[i].size(); ++j) {
//Mat sample(1, 3, CV_64F);
//Mat prob(1, gmmbg->getClustersNumber(), CV_64F);
//sample.at<double>(0, 0) = (double) psample[i][j][0];
//sample.at<double>(0, 1) = (double) psample[i][j][1];
//sample.at<double>(0, 2) = (double) psample[i][j][2];
//Vec2d pre = gmmbg->predict2(sample, prob);
////                    double curProb = 0.0;
////                    for(auto clu=0; clu<gmmbg->getClustersNumber(); ++clu){
////                        curProb += prob.at<double>(0, clu) * pGmmWeights[clu];
////                    }
//pbg -= pre[0];
////                    pbg += curProb;
//}
//pnLogs[i] = pbg / (double)psample[i].size();
//if(l == testL){
//printf("frame %d(%d), num: %d, value: %.3f\n", i+offset, i, (int)psample[i].size(), pnLogs[i]);
//}
//}
//const size_t pProbth = pnLogs.size() * 0.9;
//nth_element(pnLogs.begin(), pnLogs.begin()+pProbth, pnLogs.end());
////const double res = *max_element(pnLogs.begin(), pnLogs.end());
//const double res = pnLogs[pProbth];
//printf("result: %.3f\n", res);
//if(res > nLogThres ){
//for(auto i=0; i<width * height; ++i){
//if(pLabel[i] == l)
//result.data[i] = 255;
//}
//}
//}





//////////////////////////////////////////////////////////////////////////
//From: dynamicSegment.h
//unsed function for segmenting displays
//05/21/2016
//////////////////////////////////////////////////////////////////////////
void computeColorConfidence(const std::vector<cv::Mat>& input, Depth& result) const;
//compute threshold for nlog
double computeNlogThreshold(const std::vector<cv::Mat>& input, const cv::Mat& inputMask, const int K) const;

void getHistogram(const std::vector<cv::Vec3b>& samples, std::vector<double>& hist, const int nBin) const;
void assignColorTerm(const std::vector<cv::Mat>& warped, const cv::Ptr<cv::ml::EM> fgModel, const cv::Ptr<cv::ml::EM> bgModel,
                     std::vector<double>& colorTerm) const;

//		void solveMRF(const std::vector<double>& unary,
//					  const std::vector<double>& vCue, const std::vector<double>& hCue,
//					  const cv::Mat& img, const double weight_smooth,
//					  cv::Mat& result) const;

void DynamicSegment::computeColorConfidence(const std::vector<cv::Mat> &input, Depth &result) const {
    //compute color difference pattern
    cout << "Computing dynamic confidence..." << endl;
    const int width = input[0].cols;
    const int height = input[0].rows;
    result.initialize(width, height, 0.0);

    //search in a 7 by 7 window
    const int pR = 2;
    auto threadFunc = [&](int tid, int numt){
        for(auto y=tid; y<height; y+=numt) {
            for (auto x = 0; x < width; ++x) {
                double count = 0.0;
                vector<double> colorDiff;
                colorDiff.reserve(input.size());

                for (auto i = 0; i < input.size(); ++i) {
                    if (i == anchor - offset)
                        continue;
                    Vec3b curPix = input[i].at<Vec3b>(y, x);
                    if (curPix == Vec3b(0, 0, 0)) {
                        continue;
                    }
                    double min_dis = numeric_limits<double>::max();
                    for (auto dx = -1 * pR; dx <= pR; ++dx) {
                        for (auto dy = -1 * pR; dy <= pR; ++dy) {
                            const int curx = x + dx;
                            const int cury = y + dy;
                            if (curx < 0 || cury < 0 || curx >= width || cury >= height)
                                continue;
                            Vec3b refPix = input[anchor - offset].at<Vec3b>(cury, curx);
                            min_dis = std::min(cv::norm(refPix - curPix), min_dis);
                        }
                    }
                    colorDiff.push_back(min_dis);
                    count += 1.0;
                }
                if (count < 1)
                    continue;
                const size_t kth = colorDiff.size() * 0.8;
//				sort(colorDiff.begin(), colorDiff.end(), std::less<double>());
                nth_element(colorDiff.begin(), colorDiff.begin() + kth, colorDiff.end());
//				dynamicness(x,y) = accumulate(colorDiff.begin(), colorDiff.end(), 0.0) / count;
                result(x, y) = colorDiff[kth];
            }
        }
    };


    const int num_thread = 6;
    vector<thread_guard> threads((size_t)num_thread);
    for(auto tid=0; tid<num_thread; ++tid){
        std::thread t(threadFunc, tid, num_thread);
        threads[tid].bind(t);
    }
    for(auto &t: threads)
        t.join();
}

void DynamicSegment::getHistogram(const std::vector<cv::Vec3b> &samples, std::vector<double> &hist,
                                  const int nBin) const {
    CHECK_EQ(256%nBin, 0);
    hist.resize((size_t)nBin*3, 0.0);
    const int rBin = 256 / nBin;
    for(auto& s: samples){
        vector<int> ind{(int)s[0]/rBin, (int)s[1]/rBin, (int)s[2]/rBin};
        hist[ind[0]] += 1.0;
        hist[ind[1]+nBin] += 1.0;
        hist[ind[2]+2*nBin] += 1.0;
    }
    //normalize
    const double sum = std::accumulate(hist.begin(), hist.end(), 0.0);
    //const double sum = *max_element(hist.begin(), hist.end());
    CHECK_GT(sum, 0);
    for(auto &h: hist)
        h /= sum;
}

//estimate nLog value based on masked region (pedestrian...)
double DynamicSegment::computeNlogThreshold(const std::vector<cv::Mat> &input, const cv::Mat &inputMask, const int K) const {
    CHECK(!input.empty());
    const int width = input[0].cols;
    const int height = input[0].rows;
    CHECK_EQ(inputMask.cols, width);
    CHECK_EQ(inputMask.rows, height);
    CHECK_EQ(inputMask.channels(), 1);

    Mat mask = 255 - inputMask;
    const uchar* pMask = mask.data;
    //process inside patches. For each patch, randomly choose half pixels for negative samples
    const int pR = 50;
    const int fR = 5;
    const double nRatio = 0.5;

    double threshold = 0.0;
    double count = 0.0;
    Mat labels, stats, centroids;
    const int nLabel = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    const int min_area = 150;
    const int min_kSample = 1000;

    for(auto l=1; l<nLabel; ++l){
        printf("Component %d/%d\n", l, nLabel-1);
        const int area = stats.at<int>(l,CC_STAT_AREA);
        if(area < min_area)
            continue;
        const int left = stats.at<int>(l,CC_STAT_LEFT);
        const int top = stats.at<int>(l, CC_STAT_TOP);
        const int comW = stats.at<int>(l, CC_STAT_WIDTH);
        const int comH = stats.at<int>(l, CC_STAT_HEIGHT);

        for(auto cy=top+pR; cy<top+comH-pR; cy+=pR){
            for(auto cx=left+pR; cx<left+comW-pR; cx+=pR){
                vector<Vec3b> samples;

                for(auto x=cx-pR; x<=cx+pR; ++x){
                    for(auto y=cy-pR; y<=cy+pR; ++y){
                        if(x<0 || y<0 || x >=width || y>=height)
                            continue;
                        if(pMask[y*width+x] > 200){
                            for(auto v=-1*fR; v<=fR; ++v)
                                samples.push_back(input[anchor-offset+v].at<Vec3b>(y,x));
                        }
                    }
                }
                if(samples.size() < min_kSample)
                    continue;
                std::random_shuffle(samples.begin(), samples.end());
                const int boundry = (int)(samples.size() * nRatio);
                cv::Ptr<cv::ml::EM> gmmbg = cv::ml::EM::create();
                CHECK(gmmbg.get());
                gmmbg->setClustersNumber(K);
                Mat nSamples(boundry, 3, CV_64F);
                for(auto i=0; i<boundry; ++i){
                    nSamples.at<double>(i,0) = (double)samples[i][0];
                    nSamples.at<double>(i,1) = (double)samples[i][1];
                    nSamples.at<double>(i,2) = (double)samples[i][2];
                }
                printf("Training at (%d,%d)... Number of samples: %d\n", cx, cy, boundry);
                gmmbg->trainEM(nSamples);

                double curnLog = 0.0;
                double curCount = 0.0;
                for(auto i=boundry; i<samples.size(); ++i){
                    Mat s(1,3,CV_64F);
                    for(auto j=0; j<3; ++j)
                        s.at<double>(0,j) = (double)samples[i][j];
                    Mat prob(1, gmmbg->getClustersNumber(), CV_64F);
                    Vec2d pre = gmmbg->predict2(s, prob);
                    curnLog -= pre[0];
                    curCount += 1.0;
                }
                printf("Done, nLog: %.3f\n", curnLog/curCount);
                threshold += curnLog / curCount;
                count += 1.0;
            }
        }
    }
    CHECK_GT(count,0.9);
    return threshold / count;
}

void DynamicSegment::assignColorTerm(const std::vector<cv::Mat> &warped, const Ptr<cv::ml::EM> fgModel,
                                     const cv::Ptr<cv::ml::EM> bgModel, std::vector<double> &colorTerm)const {
    CHECK(!warped.empty());
    const int width = warped[0].cols;
    const int height = warped[0].rows;
    colorTerm.resize((size_t)width * height * 2);
    for(auto v=0; v<warped.size(); ++v){
        const uchar* pImg = warped[v].data;
        for(auto i=0; i<width * height; ++i){
            Mat x(3,1,CV_64F);
            double* pX = (double*) x.data;
            pX[0] = pImg[3*i];
            pX[1] = pImg[3*i+1];
            pX[2] = pImg[3*i+2];
            Vec2d predfg = fgModel->predict2(x, Mat());
            Vec2d predbg = bgModel->predict2(x, Mat());
            //use negative log likelihood for energy
            colorTerm[2*i] = -1 * predbg[0];
            colorTerm[2*i+1] = -1 * predfg[0];
        }
    }
}

