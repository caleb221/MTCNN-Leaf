#include <iostream>
#include <time.h>
#include "mtcnn_detect.h"


double what_time_is_it_now
    (
    void
    )
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return now.tv_sec + now.tv_nsec*1e-9;
}

int main
    (
    int ac,
    char *av[]
    )
{
    cv::VideoCapture cap;
    if (ac == 2)
        {
        cap.open(av[1]);
        }
    else
        {
        cap.open(0);
        }

    cv::Mat im;
    cap >> im;
    const int IMW = im.cols;
    const int IMH = im.rows;

    init_mtcnn(IMW, IMH);

    int cnt = 0;
    double timesum = 0.0;
    while (1)
        {
        cap >> im;

        std::vector<obj_info> detectedobj_info;

        double time1 = what_time_is_it_now();

        run_mtcnn(im, detectedobj_info);

        double time2 = what_time_is_it_now();
        if (cnt < 50)
            {
            double duration = time2 - time1;
            timesum += duration;
            ++cnt;
            }
        else
            {
            std::cout << "fps: " << (double)cnt / timesum << std::endl;
            timesum = 0.0;
            cnt = 0;
            }


        unsigned int num_onet_boxes = detectedobj_info.size();
        for (unsigned int i = 0; i < num_onet_boxes; ++i)
            {
            cv::rectangle(im, cv::Point(detectedobj_info[i].bbox.xmin, detectedobj_info[i].bbox.ymin),
                          cv::Point(detectedobj_info[i].bbox.xmax, detectedobj_info[i].bbox.ymax), cv::Scalar(0, 0, 255), 1, 16);
            }

        cv::imshow("demo", im);
        unsigned char key = cv::waitKey(1);
        if (key == 27)
            {
            break;
            }

        }

    return 0;
}

