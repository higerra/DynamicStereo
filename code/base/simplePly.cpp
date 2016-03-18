//
// Created by yanhang on 3/18/16.
//

#include "simplePly.h"
#include <iostream>
#include <fstream>

namespace ply_util{
    using namespace std;
    bool SimplePly::read(const std::string &filename) {
        bool hasColor = false;
        bool hasNormal = false;
        bool is_binary = false;
        ifstream headerin(filename.c_str());
        if(!headerin.is_open())
            return false;
        string temp, format;
        int kVertex, kFace;
        headerin >> temp >> temp;
        headerin >> format;
        if(format == "ascii")
            is_binary = false;
        if(format == "binary_little_endian")
            is_binary = true;
        else{
            cerr << "Wrong format in the ply file " << filename << endl;
            return false;
        }
        headerin >> temp >> temp >> kVertex;
        std::getline(headerin, temp);
        headerin >> temp;
        if(temp == "property")
            hasColor = true;
        headerin >> temp >> temp >> temp >> temp >> temp >> temp >> temp >> temp;
        headerin >> temp;
        if(temp == "property")
            hasNormal = true;


    }

    void SimplePly::write(const std::string &filename, const bool binary) const {
        bool hasColor = false;
        bool hasNormal = false;
        for(const auto& v: vertex){
            if(v.c[0] >= 0)
                hasColor = true;
            if(v.n.norm() > 0)
                hasNormal = true;
            if(hasColor && hasNormal)
                break;
        }

        ofstream headerout(filename.c_str());
        CHECK(headerout.is_open()) << "Can not open file for writting";
        headerout << "ply" << endl;
        if(binary)
            headerout << "format binary_little_endian 1.0" << endl;
        else
            headerout << "format ascii 1.0" << endl;
        headerout << "element vertex " << getNumVertex() << endl;
        headerout << "property float x\nproperty float y\nproperty float z" << endl;
        if(hasColor)
            headerout << "property uchar r\nproperty uchar g\nproperty uchar b" << endl;
        if(hasNormal)
            headerout << "property float nx\nproperty float ny\nproperty float nz" << endl;
        headerout << "element face " << getNumFace() << endl;
        headerout << "end_header" << endl;
        headerout.close();

        //rearrange data into an array
        ofstream dataOut;
        if(binary)
            dataOut.open(filename.c_str(), ios::binary | ios::app);
        else
            dataOut.open(filename.c_str(), ios::app);
        for(const auto& v: vertex){
            vector<float>curv{(float)v.p[0], (float)v.p[2], (float)v.p[3]};
            if(binary)
                dataOut.write((char*) curv.data(), sizeof(float)*3);
            else
                dataOut << curv[0] << ' ' << curv[1] << ' ' << curv[2] << endl;
            if(hasColor){
                vector<uchar>curc{(uchar)v.c[0], (uchar)v.c[1], (uchar)v.c[2]};
                if(binary)
                    dataOut.write((char*)curc.data(), sizeof(uchar) * 3);
                else
                    dataOut << curc[0] << ' ' << curc[1] << ' ' << curc[2] << endl;
            }
            if(hasNormal){
                vector<float> curn{(float)v.n[0], (float)v.n[1], (float)v.n[2]};
                if(binary)
                    dataOut.write((char*)curn.data(), sizeof(float) * 3);
                else
                    dataOut << curn[0] << ' ' << curn[1] << ' ' << curn[2] << endl;
            }
        }
        dataOut.close();
    }
}//namespace ply_util