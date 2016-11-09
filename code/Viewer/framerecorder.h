#ifndef FRAMERECORDER_H
#define FRAMERECORDER_H

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <QImage>
#include <QDir>
#include <string>
#include <atomic>
#include <memory>
#include "libavcodec/avcodec.h"

class FrameRecorder
{
public:
    FrameRecorder(const std::string &datapath_);
    ~FrameRecorder();
    void submitFrame(const std::shared_ptr<QImage> new_image);
    inline void exit(){
        is_interrupted.store(true);
        cv.notify_all();
    }

private:
    void save();
    void saveImage(std::shared_ptr<QImage> img);
    std::string datapath;
    std::deque<std::shared_ptr<QImage> > saving_queue;
    std::condition_variable cv;
    std::mutex mt;
    std::thread t;
    std::atomic<bool> is_interrupted;
    const int max_frame_num;
    int frame_counter;
};

#endif // FRAMERECORDER_H
