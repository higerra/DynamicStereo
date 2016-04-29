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
////					cv::line(colorRef, cv::Point(x, 0), cv::Point(x, fullImg[0].rows - 1), Scalar(255, 255, 255), 1);
////					cv::line(colorSrc, cv::Point(x, 0), cv::Point(x, fullImg[0].rows - 1), Scalar(255, 255, 255), 1);
////				}
////				for (auto y = 0; y < fullImg[0].rows; y += gridWarpping.getBlockH()) {
////					cv::line(colorRef, cv::Point(0, y), cv::Point(fullImg[0].cols - 1, y), Scalar(255, 255, 255), 1);
////					cv::line(colorSrc, cv::Point(0, y), cv::Point(fullImg[0].cols - 1, y), Scalar(255, 255, 255), 1);
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