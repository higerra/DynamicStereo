//
// Created by yanhang on 3/18/16.
//

#ifndef DYNAMICSTEREO_SIMPLEPLY_H
#define DYNAMICSTEREO_SIMPLEPLY_H
#include <vector>
#include <Eigen/Eigen>
#include <glog/logging.h>

namespace ply_util {

    typedef unsigned char uchar;
    struct Vertex{
        Vertex(): p(0,0,0), c(-1,-1,-1), n(0,0,0){}
        Eigen::Vector3d p;
        Eigen::Vector3d c;
        Eigen::Vector3d n;
    };

    struct Triangle{
        Triangle(int a, int b, int c): tri(a,b,c){}
        Eigen::Vector3i tri;
    };

    class SimplePly {
    public:
        bool read(const std::string& filename);
        void write(const std::string& filename, const bool binary = true) const;

        inline const std::vector<Vertex>& getVertex() const{
            return vertex;
        }
        inline const std::vector<Triangle>& getFace() const{
            return face;
        }
        inline std::vector<Vertex>& getVertex(){
            return vertex;
        }
        inline std::vector<Triangle>& getFace(){
            return face;
        }
        inline void addVertex(const Eigen::Vector3d& p_){
            Vertex newv;
            newv.p = p_;
            vertex.push_back(newv);
        }

        inline void addVertex(const Eigen::Vector3d& p_, const Eigen::Vector3d& c_, const Eigen::Vector3d& n_){
            Vertex newv;
            newv.p = p_;
            newv.c = c_;
            newv.n = n_;
            vertex.push_back(newv);
        }

        inline void addFace(const int id1, const int id2, const int id3){
            CHECK_GE(id1, 0);
            CHECK_GE(id2, 0);
            CHECK_GE(id3, 0);
            CHECK_LT(id1, vertex.size());
            CHECK_LT(id2, vertex.size());
            CHECK_LT(id3, vertex.size());
            face.push_back(Triangle(id1, id2, id3));
        }

        inline int getNumVertex() const{
            return (int)vertex.size();
        }

        inline int getNumFace() const{
            return (int)face.size();
        }
    private:
        std::vector<Vertex> vertex;
        std::vector<Triangle> face;
    };
}//namespace ply_util
#endif //DYNAMICSTEREO_SIMPLEPLY_H
