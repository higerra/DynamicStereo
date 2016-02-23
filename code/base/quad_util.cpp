#include "quad.h"
#include "frame.h"
#include <algorithm>
#include <numeric>
using namespace std;
using namespace Eigen;
using namespace cv;

namespace dynamic_rendering{
	namespace quad_util{
		bool compareVectorNorm(const Vector3d& v1, const Vector3d& v2){
			return v1.norm() < v2.norm();
		}

		Matrix3d covariance3d(vector<Vector3d>&m, const double ratio){
			Matrix3d cov = Matrix3d::Zero();
			if(m.size() == 0)
				return cov;
			MatrixXd A(3,m.size());
			Vector3d average = std::accumulate(m.begin(), m.end(), Vector3d(0,0,0));
			average = average / (double)m.size();
			//robust covariance
			if(ratio < 0.99){
				for(int i=0; i<m.size(); i++)
					m[i] -= average;
				sort(m.begin(), m.end(), compareVectorNorm);
				const size_t target_size = m.size() * ratio;
				while(m.size() > target_size)
					m.pop_back();
				for(int i=0; i<m.size(); i++)
					m[i] += average;
				average = std::accumulate(m.begin(), m.end(), Vector3d(0,0,0));
				average = average / (double)m.size();
			}

			for(int i=0; i<m.size(); i++)
				A.block<3,1>(0,i) = m[i] - average;
			cov = A * A.transpose();
			for(int i=0; i<3; i++){
				for(int j=0; j<3; j++){
					cov(i,j) = cov(i,j) / ((double)m.size() - 1);
				}
			}
			return cov;
		}

		double getLinesColorVar(const Frame& frame, const vector<Vector2d>& spt, const vector<Vector2d>& ept, const double offset, const double ratio, bool inside){
			if(spt.size() != ept.size()){
				cerr << "getLinesColorVar: spt.size() != ept.size()"<<endl;
				exit(-1);
			}
			const double sampleNum = 100;
			vector<Vector3d> color_array;
			for(int i=0; i<spt.size(); i++){
				Vector2d curline = ept[i] - spt[i];
				Vector2d curnormal(-1*curline[1], curline[0]);
				if(!inside)
					curnormal = -1 * curnormal;
				curnormal.normalize();
				for(double t = 0.0; t<1.0; t+= 1.0/sampleNum){
					Vector2d samplePos =spt[i] + curline * t + curnormal * offset;
					if(!frame.isValidRGB(samplePos))
						continue;
					Vector3d curcolor = frame.getColor(samplePos);
					color_array.push_back(curcolor);
				}
			}
			Matrix3d cov = covariance3d(color_array, ratio);
			return cov(0,0)+cov(1,1)+cov(2,2);
		}

		double getQuadColorVar(const Frame& frame, const Quad& quad, const double offset, const double ratio, bool inside){
			vector<Vector2d>spt, ept;
			for(int i=0; i<quad.cornerpt.size(); i++){
				const int ind1 = i;
				const int ind2 = (i+1)%4;
				spt.push_back(quad.cornerpt[ind1]);
				ept.push_back(quad.cornerpt[ind2]);
			}
			return getLinesColorVar(frame, spt, ept, offset, ratio, inside);
		}

		double getQuadColorDiff(const Quad& q1,const Quad& q2, const Frame& frame1,const Frame& frame2, const double offset, bool inside){
			double result = 0;
			const double sampleNum = 100;

			double color_average1 = 0;
			double color_average2 = 0;
			vector<Vector3d> color_array1;
			vector<Vector3d> color_array2;

//	vector<Vector2d> pos1, pos2;

			for(int i=0; i<q1.cornerpt.size(); i++){
				const int ind1 = i;
				const int ind2 = (i+1) % q1.cornerpt.size();

				const double& startx1 = q1.cornerpt[ind1][0];
				const double& starty1 = q1.cornerpt[ind1][1];
				const double& endx1 = q1.cornerpt[ind2][0];
				const double& endy1 = q1.cornerpt[ind2][1];

				const double& startx2 = q2.cornerpt[ind1][0];
				const double& starty2 = q2.cornerpt[ind1][1];
				const double& endx2 = q2.cornerpt[ind2][0];
				const double& endy2 = q2.cornerpt[ind2][1];

				Vector2d curline1 = q1.cornerpt[ind2] - q1.cornerpt[ind1];
				Vector2d curline2 = q2.cornerpt[ind2] - q2.cornerpt[ind1];
				Vector2d curnormal1(-1*curline1[1], curline1[0]);
				Vector2d curnormal2(-1*curline2[1], curline2[0]);
				if(!inside){
					curnormal1 = -1 * curnormal1;
					curnormal2 = -1 * curnormal2;
				}
				curnormal1.normalize(); curnormal2.normalize();

				int count = 0;
				for(double t = 0; t < 1.0; t += 1.0 / sampleNum){
					double curx1 = startx1 + (endx1 - startx1) * t;
					double cury1 = starty1 + (endy1 - starty1) * t;
					double curx2 = startx2 + (endx2 - startx2) * t;
					double cury2 = starty2 + (endy2 - starty2) * t;
					Vector2d samplePos1 = Vector2d(curx1, cury1) + curnormal1 * offset;
					Vector2d samplePos2 = Vector2d(curx2, cury2) + curnormal2 * offset;
					if(!frame1.isValidRGB(samplePos1))
						continue;
					if(!frame2.isValidRGB(samplePos2))
						continue;

					// pos1.push_back(samplePos1);
					// pos2.push_back(samplePos2);

					Vector3d c1 = frame1.getColor(samplePos1);
					Vector3d c2 = frame2.getColor(samplePos2);

					color_array1.push_back(c1);
					color_array2.push_back(c2);
					color_average1 = color_average1 + c1[0] + c1[1] + c1[2];
					color_average2 = color_average2 + c2[0] + c2[1] + c2[2];

					count++;
				}

			}
			if(color_array1.size() == 0 || color_array2.size() == 0)
				return 0.0;


			//make the color zero mean to compensate exposure difference
			color_average1 /= static_cast<double>(color_array1.size() * 3);
			color_average2 /= static_cast<double>(color_array2.size() * 3);

			Vector3d average1(color_average1, color_average1, color_average1);
			Vector3d average2(color_average2, color_average2, color_average2);

			vector<double> diff_array;
			for(int i=0; i<color_array1.size(); i++){
				Vector3d curdiff = (color_array1[i]-average1) - (color_array2[i]-average2);
				diff_array.push_back(curdiff.norm());
			}

			result = 0;
// 	for(int i=0; i<4; i++){
// 	    double temp = accumulate(diff_array.begin()+i*sampleNum, diff_array.begin()+(i+1)*sampleNum, 0.0) / static_cast<double>(sampleNum);
// //	    result = std::max(temp, result);
// 	    result += temp;
// 	}
// 	result /= 4;
			result = std::accumulate(diff_array.begin(), diff_array.end(), 0.0);
			result /= static_cast<double>(diff_array.size());

//	return inverseHuberLoss(result, 0.0);
			return result * result;
		}

		double getQuadShapeDiff(const Quad& q1, const Quad& q2){
			double result = 0.0;
			for(int i=0; i<4; i++){
				int ind1 = i;
				int ind2 = (i+1)%4;
				Vector2d edge1 = q1.cornerpt[ind2] - q1.cornerpt[ind1];
				Vector2d edge2 = q2.cornerpt[ind2] - q2.cornerpt[ind1];
				result = std::max(result,std::abs(edge1.norm() - edge2.norm()));
			}
			return result;
		}


		void getQuadDepth(const Quad&q,
		                  const Frame& frame,
		                  vector<double> &dv){
			const vector<Vector2d>& corners = q.cornerpt;
			const Depth& depth = frame.getDepth();
			const double epsilon = 0.01;
			dv.resize(4);
			for(int j=0; j<corners.size(); j++){
				const Vector2d depthpix = frame.RGBToDepth(corners[j]);
				if(depth.insideDepth(depthpix))
					dv[j] = depth.getDepthAt(depthpix);
				else{
					//extepolate depth
					Vector2d linedir;
					Vector2d startpt;
					const Vector2d& predepthpix = frame.RGBToDepth(corners[(j+3)%4]);
					const Vector2d& nextdepthpix = frame.RGBToDepth(corners[(j+1)%4]);
					if(depth.insideDepth(predepthpix))
						startpt = predepthpix;
					else
						startpt = nextdepthpix;
					linedir = depthpix - startpt;
					const double sampleNum = 100;
					Vector2d last_inside_pos;
					for(double t=0.0; t<1.0; t += 1.0/sampleNum){
						Vector2d curpos = startpt + linedir * t;
						if(depth.insideDepth(curpos))
							last_inside_pos = curpos;
						else
							break;
					}
					double depth_border = depth.getDepthAt(last_inside_pos);
					double depth_startpt = depth.getDepthAt(startpt);
					double len_inside = (last_inside_pos - startpt).norm();
					if(len_inside < epsilon)
						dv[j] = epsilon;
					else
						dv[j] = depth_startpt + (depth_border-depth_startpt) * (linedir.norm() / len_inside);
				}
			}
		}

		double outsideRatio(const Frame& frame, const Quad& quad){
			double outer_ratio = 0;
			const double sampleNum = 20;
			for(int j=0; j<quad.cornerpt.size(); j++){
				const int ind1 = j;
				const int ind2 = (j+1)%4;
				const Vector2d& startpt = quad.cornerpt[ind1];
				const Vector2d& endpt = quad.cornerpt[ind2];
				Vector2d edge = quad.cornerpt[ind2] - quad.cornerpt[ind1];
				for(double t=0; t<1.0; t+=1.0/sampleNum){
					Vector2d samplePos = startpt + (endpt - startpt) * t;
					if(!frame.isValidRGB(samplePos))
						outer_ratio += 1.0;
				}
			}
			outer_ratio /= sampleNum * 4;
			return outer_ratio;
		}

		void drawQuad(const Quad& quad,
		              Mat& image,
		              bool is_color,
		              int thickness){
			Scalar colors[] = {Scalar(0,255,0), Scalar(0,255,0), Scalar(0,255,0), Scalar(0,255,0)};
			for(int i=0; i<quad.cornerpt.size(); i++){
				const vector<Vector2d>& corners = quad.cornerpt;
				const int ind1 = i; const int ind2 = (i+1)%corners.size();
				if(is_color)
					cv::line(image, cv::Point(corners[ind1][0], corners[ind1][1]), cv::Point(corners[ind2][0], corners[ind2][1]), colors[i], thickness);
				else
					cv::line(image, cv::Point(corners[ind1][0], corners[ind1][1]), cv::Point(corners[ind2][0], corners[ind2][1]), colors[0], thickness);
			}
		}

		void drawSingleLine(const KeyLine& line,
		                    const int lid,
		                    Mat& image,
		                    const Scalar& color,
		                    int thickness){
			if(!image.data){
				cerr << "drawSingleLine():: empty image!"<<endl;
				exit(-1);
			}
			char buffer[100];
			cv::Point pt1(line.startPoint[0], line.startPoint[1]);
			cv::Point pt2(line.endPoint[0], line.endPoint[1]);
			cv::Point midpt((pt1.x+pt2.x)/2, (pt1.y+pt2.y)/2+5);
			cv::line(image, pt1, pt2, color, thickness);
			sprintf(buffer, "%d", lid);
			cv::putText(image, string(buffer), midpt, FONT_HERSHEY_PLAIN, 1, Scalar(255,255,0));
		}

		void drawLineGroup(const vector<vector<KeyLine> >& line_group,
		                   Mat& image,
		                   int thickness){
			if(!image.data){
				cerr << "drawLineGroup(): empty image!"<<endl;
				exit(-1);
			}
			Scalar colors[] = {Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255),
			                   Scalar(255,255,0), Scalar(255,0,255), Scalar(0,255,255)};
			for(int i=0; i<line_group.size(); i++){
				const Scalar& curcolor = colors[i%6];
				for(int j=0; j<line_group[i].size(); j++)
					drawSingleLine(line_group[i][j], j,image, curcolor, thickness);
			}
		}
	}
}
