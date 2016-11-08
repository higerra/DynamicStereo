#include "framerecorder.h"
using namespace std;

FrameRecorder::FrameRecorder(const std::string& datapath_):
    datapath(datapath_),
    is_interrupted(false),
    max_frame_num(1000),
    frame_counter(0)
{
    t = thread(&FrameRecorder::save, this);
    datapath.append("/record/");
    QDir dir(QString::fromStdString(datapath));
    if(!dir.exists()){
        dir.mkpath(QString::fromStdString(datapath));
    }
}

FrameRecorder::~FrameRecorder(){
    exit();
    if(t.joinable())
        t.join();
}

void FrameRecorder::save(){
    while(true){
        unique_lock<mutex> lock(mt);
        cv.wait(lock, [&]{return !this->saving_queue.empty() || this->is_interrupted.load();});
        if(is_interrupted.load()){
            int total = (int)saving_queue.size() + frame_counter;
            while(!saving_queue.empty()){
                printf("saving image: %d/%d\n", frame_counter, total);
                shared_ptr<QImage> img = saving_queue.front();
                saving_queue.pop_front();
                saveImage(img);
                frame_counter++;
            }
            break;
        }else{
            shared_ptr<QImage> img = saving_queue.front();
            saving_queue.pop_front();
            lock.unlock();
            saveImage(img);
            frame_counter++;
        }
    }
}

void FrameRecorder::submitFrame(const std::shared_ptr<QImage> new_image){
    lock_guard<mutex> guard(mt);
    if(saving_queue.size() < max_frame_num){
        saving_queue.push_back(new_image);
    }
    cv.notify_all();
}

void FrameRecorder::saveImage(shared_ptr<QImage> img){
    char buffer[100];
    sprintf(buffer, "%s/frame%03d.png", datapath.c_str(), frame_counter);
    img->mirrored(false, true).save(QString::fromStdString(string(buffer)));
}
