//
// Created by yanhang on 2/29/16.
//

#include <iostream>
#include <theia/theia.h>
#include <stlplus3/file_system.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "base/file_io.h"

using namespace dynamic_stereo;
using namespace std;
DEFINE_bool(pmvsFormat, true, "pmvs format");
int main(int argc, char** argv){
    if(argc < 2){
        cerr << "Usage: DataConvert <path-to-directory>" << endl;
        return 1;
    }

    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    FileIO file_io(argv[1]);

    int index = 0;
    char buffer[1024] = {};
    theia::Reconstruction reconstruction;
    string temp;
    for(auto i=0; i<file_io.getTotalNum(); ++i) {
        theia::FloatImage img;
        img.Read(file_io.getImage(i));
        sprintf(buffer, "image%05d.jpg", i);
        theia::ViewId vid = reconstruction.AddView(string(buffer));
        theia::View *view = reconstruction.MutableView(vid);
        view->SetEstimated(true);
        theia::Matrix3x4d pose;
        sprintf(buffer, "%s/pose/image%05d.jpg.txt", file_io.getDirectory().c_str(), i);
        cout << buffer << endl;
        ifstream fin(buffer);

        CHECK(fin.is_open()) << "Can not open pose file " << buffer;
        if(FLAGS_pmvsFormat)
            fin >> temp;
        for(auto y=0; y<3; ++y){
            for(auto x=0; x<4; ++x)
                fin >> pose(y,x);
        }
        cout << "Camera matrix for view " << i << endl;
        cout << pose << endl;

        theia::Camera* cam = view->MutableCamera();
        cam->InitializeFromProjectionMatrix(img.Cols(), img.Rows(), pose);
    }

    if(!stlplus::folder_exists(file_io.getSfMDirectory()))
        stlplus::folder_create(file_io.getSfMDirectory());

    cout << "View number:" << reconstruction.NumViews() << endl;
    cout << "Track number:" << reconstruction.NumTracks() << endl;
    theia::WriteReconstruction(reconstruction, file_io.getReconstruction());
    sprintf(buffer, "%s/sfm/reconstruction.ply", file_io.getDirectory().c_str());
    theia::WritePlyFile(string(buffer), reconstruction, 1);
    return 0;
}
