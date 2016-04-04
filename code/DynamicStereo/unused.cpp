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

