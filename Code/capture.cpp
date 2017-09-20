#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>
#include <sys/sysinfo.h>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include "opencv2/videoio.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/objdetect/objdetect.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "linefinder.h"

#define NUM_THREADS 5
#define NUM_CPU_CORES 4
#define PI 3.1415926
#define NSEC_PER_SEC (1000000000)
#define NSEC_PER_MSEC (1000000)
#define NSEC_PER_MICROSEC (1000)

#define GREATEST_COMMON_DIVISOR WARNING_SYSTEM_DEADLINE
#define LANE_DETECT_DEADLINE 600
#define OBJECT_DETECT_DEADLINE 150
#define WARNING_SYSTEM_DEADLINE 10
#define VIDEO_CAPTURE_DEADLINE 70
#define SEQUENCER_DEADLINE 20
#define DANGER_DISTANCE 50
#define YES 0
#define NO 1
#define RELATIVE_WIDTH 0.0397 
#define RELATIVE_PIXELS 0.00007
#define AVG_WIDTH_CAR 2
#define AVG_WIDTH_PERSON 0.5
#define AVG_WIDTH_STOP_SIGH 0.75
#define AVG_WIDTH_SIGNS 0.4
#define FRAMES 1
#include <math.h>
#include <iomanip>
using namespace cv;
using namespace std;


void draw_locations(Mat & img, vector< Rect > & locations, const Scalar & color,string text);
void * ObstacleDetect( void* threadp);
void * VideoCaptureThread(void * threadp);
void * Sequencer( void* threadp);
double getTimeMsec(void);

#define VIDEO_FILE_NAME "/home/pi/Downloads/profsams0_best.avi"
#define CASCADE_FILE_NAME "/home/pi/test_Proj/cars3.xml"
#define CASCADE1_FILE_NAME "/home/pi/test_Proj/traffic_light.xml"
#define CASCADE2_FILE_NAME "/home/pi/test_Proj/stop_sign.xml"
#define CASCADE3_FILE_NAME "/home/pi/test_Proj/pedestrian.xml"
#define CASCADE4_FILE_NAME "/home/pi/test_Proj/left-sign.xml"
#define CASCADE5_FILE_NAME "/home/pi/test_Proj/right-sign.xml"

#define LANE_DETECT_OUTPUT_FILE_NAME "profsams0_best_lane_detection_result.avi"
#define OBSTACLE_DETECT_OUTPUT_FILE_NAME "profsams0_best_obstacle_detection_result.avi"
#define CAR_IMAGE "/home/pi/test_Proj/car.png"
#define LEFT_SIGN_IMAGE "/home/pi/test_Proj/left.png"
#define RIGHT_SIGN_IMAGE "/home/pi/test_Proj/right.png"

#define WINDOW_NAME_1 "Obstacle Detection"
#define WINDOW_NAME_2 "WINDOW2"
sem_t semObjDetect,semLaneDetect, semWarningSys, semDanger, semVideoCap;
pthread_mutex_t mutex, laneDetectmutex;
Mat mFrame, mFrameObjDetect, mFrameLaneDetect, mGray, mCanny, imageROI,mGray1, mGray2, mask, mFrame2;
CascadeClassifier cars, traffic_light, stop_sign, pedestrian,sign, sign2;
vector<Rect> cars_found, traffic_light_found, stop_sign_found ,pedestrian_found ,sign_found, sign_found2, cars_tracking;
vector<int> car_timer;
int InDangerDistance = 1;
int DangerDistanceNear =0;
int LaneCrossed= 1;
double start_time, end_time;
VideoCapture capture(VIDEO_FILE_NAME); // open the video file for reading

typedef struct
{
    int threadIdx;
    int MajorPeriods;
} threadParams_t;

double getTimeMsec(void)
{
    struct timespec event_ts = {0, 0};

    clock_gettime(CLOCK_MONOTONIC, &event_ts);
    return ((event_ts.tv_sec)*1000.0) + ((event_ts.tv_nsec)/1000000.0);
}

//delta_t() IS BORROWED FROM SAM SIEWERT'S EXAMPLE CODES AND IS USED TO CALCULATE THE DIFFERNECE BETWEEN THE STOP AND START TIMES.
int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
    int dt_sec=stop->tv_sec - start->tv_sec;
    int dt_nsec=stop->tv_nsec - start->tv_nsec;

    if(dt_sec >= 0)
    {
        if(dt_nsec >= 0)
        {
            delta_t->tv_sec=dt_sec;
            delta_t->tv_nsec=dt_nsec;
        }
        else
        {
            delta_t->tv_sec=dt_sec-1;
            delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
        }
    }
    else
    {
        if(dt_nsec >= 0)
        {
            delta_t->tv_sec=dt_sec;
            delta_t->tv_nsec=dt_nsec;
        }
        else
        {
            delta_t->tv_sec=dt_sec-1;
            delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
        }
    }

    return(1);
}

void * Sequencer( void* threadp)
{
    struct timespec frame_time1;
    double ave_framedt1=0.0, ave_frame_rate1=0.0,fc1=0.0,framedt1=0.0,jitter = 0.0, avg_posJitter = 0.0, avg_negJitter = 0.0;
    double curr_frame_time1 = 0.00, prev_frame_time1=0.00;
    unsigned int frame_count1=0;
    double executionAvgTime = 0;

    struct timespec tim, tim2;
    int laneDetectionCnt=0, obstacleDetectCnt=0, videoCaptureCnt=0;
    while(1)
    {
        double start_time1 =  getTimeMsec();
        //Timing Calculations per frame
        clock_gettime(CLOCK_REALTIME,&frame_time1);
        curr_frame_time1=((double)frame_time1.tv_sec * 1000.0) + ((double)((double)frame_time1.tv_nsec /1000000.0));

        frame_count1++;
        if(frame_count1 > 2)
        {
            fc1=(double)frame_count1;
            ave_framedt1=((fc1-1.0)*ave_framedt1 + framedt1)/fc1;
            ave_frame_rate1=1.0/(ave_framedt1/1000.0);
        }

        // Simulate the C.I. for S1 and S2 and timestamp in log
        //printf("\n**** CI t=%lf\n", (getTimeMsec() - start_time));
        tim.tv_sec = 0;
        tim.tv_nsec = GREATEST_COMMON_DIVISOR * NSEC_PER_MSEC;
        //for complete and more accuracy we can uncomment the while loop
        // do{
        tim2 = {0};
        if(nanosleep(&tim , &tim2) ==EINTR)
        {
            printf("Nano sleep system call failed \n");
        }
        //	  else{
        //		  tim.tv_nsec = tim2.tv_nsec;
        //	  }
        // }while(tim2.tv_nsec != 0 )

        sem_post(&semWarningSys);

        laneDetectionCnt++;
        obstacleDetectCnt++;
        videoCaptureCnt++;
        if(laneDetectionCnt == LANE_DETECT_DEADLINE/GREATEST_COMMON_DIVISOR)
        {
            sem_post(&semLaneDetect);
            laneDetectionCnt=0;
        }
        if(obstacleDetectCnt == OBJECT_DETECT_DEADLINE/GREATEST_COMMON_DIVISOR)
        {
            sem_post(&semObjDetect);
            obstacleDetectCnt=0;
        }
        if(videoCaptureCnt == VIDEO_CAPTURE_DEADLINE/GREATEST_COMMON_DIVISOR)
        {
            sem_post(&semVideoCap);
            videoCaptureCnt=0;
        }

        double end_time1 =  getTimeMsec() - start_time1;
        //printf("Execution Time of Lane Detect thread: %f\n", end_time1);
        if(frame_count1!=0)executionAvgTime =((executionAvgTime * (frame_count1-1)) + end_time1)/(double)frame_count1;

        framedt1=curr_frame_time1 - prev_frame_time1;
        prev_frame_time1=curr_frame_time1;
        if (frame_count1 > 0)
        {
            jitter = SEQUENCER_DEADLINE - framedt1;
            if (jitter < 0)
            {
                //printf("\n Frame Time (ms): %f and Deadline missed for Frame : %d\n", framedt1, frame_count1);
                avg_negJitter  = jitter;//(avg_negJitter + jitter);
				avg_posJitter=0;
            }
            else
            {
                //printf("\n Frame %d finished earlier than Deadline\n", i);
                avg_posJitter = jitter;//(jitter + avg_posJitter);
				avg_negJitter=0;
            }

        }
        if(frame_count1==FRAMES)
        {
            frame_count1=0;

           /* printf("Sequencer Avrg Execution time: %f ms\n", executionAvgTime);
            printf("Sequencer  Avrg Request time: %f ms\n", ave_framedt1);
            printf("Average Calclated for : %d iterations\n",FRAMES);
            printf("Sequencer  frequency %f Hz\n", ave_frame_rate1);
            printf("(Min) Positive Jitter %f:\n" , avg_posJitter/FRAMES);
            printf("(Max) Negative Jitter %f:\n", avg_negJitter/FRAMES);
            printf("Total Avg. Jitter %f: \n", (avg_posJitter+avg_negJitter) /(double) FRAMES);*/
            
	    printf("Sequencer Avrg Execution time: %f ms\n", end_time1);
            printf("Sequencer  frequency %f Hz\n", 1000/framedt1);
            printf("(Min) Positive Jitter %f:\n" , avg_posJitter/FRAMES);
            printf("(Max) Negative Jitter %f:\n", avg_negJitter/FRAMES);
        }
    }
}
//control system thread
void * WarningSystem( void* threadp)
{
    struct timespec frame_time1;
    double ave_framedt1=0.0, ave_frame_rate1=0.0,fc1=0.0,framedt1=0.0,jitter = 0.0, avg_posJitter = 0.0, avg_negJitter = 0.0;
    double curr_frame_time1 = 0.00, prev_frame_time1=0.00;
    unsigned int frame_count1=0;
    double executionAvgTime = 0;
    while(1) {
        sem_wait(&semWarningSys);
        double start_time1 =  getTimeMsec();
        //Timing Calculations per frame
        clock_gettime(CLOCK_REALTIME,&frame_time1);
        curr_frame_time1=((double)frame_time1.tv_sec * 1000.0) + ((double)((double)frame_time1.tv_nsec /1000000.0));

        frame_count1++;
        if(frame_count1 > 2)
        {
            fc1=(double)frame_count1;
            ave_framedt1=((fc1-1.0)*ave_framedt1 + framedt1)/fc1;
            ave_frame_rate1=1.0/(ave_framedt1/1000.0);
        }
        //give delay or take the Mat mframe and display the printf statements on the frames !! so it looks like a realtime service
        if(DangerDistanceNear) {
            printf("Obstacle ahead!! Please take the control or Allow me to take the decision\n");
            printf("Slowing down the car\n");
        }
        pthread_mutex_lock(&mutex);
        if(InDangerDistance==YES) {
            InDangerDistance = NO;
            printf("Obstacle ahead!! stopping the Car\n");
        }
        pthread_mutex_unlock(&mutex);
		
		pthread_mutex_lock(&laneDetectmutex);
		if(LaneCrossed==YES) {
            LaneCrossed = NO;
            printf("Lane crossed!! Please take control\n");
        }
        pthread_mutex_unlock(&laneDetectmutex);

        double end_time1 =  getTimeMsec() - start_time1;
        if(frame_count1!=0)executionAvgTime =((executionAvgTime * (frame_count1-1)) + end_time1)/(double)frame_count1;

        framedt1=curr_frame_time1 - prev_frame_time1;
        prev_frame_time1=curr_frame_time1;
         if (frame_count1 > 0)
        {
            jitter = WARNING_SYSTEM_DEADLINE - framedt1;
            if (jitter < 0)
            {
                //printf("\n Frame Time (ms): %f and Deadline missed for Frame : %d\n", framedt1, frame_count1);
                avg_negJitter  = jitter;//(avg_negJitter + jitter);
				avg_posJitter=0;
            }
            else
            {
                //printf("\n Frame %d finished earlier than Deadline\n", i);
                avg_posJitter = jitter;//(jitter + avg_posJitter);
				avg_negJitter=0;
            }

        }
        if(frame_count1==FRAMES)
        {
            frame_count1=0;

           /* printf("Warning System Avrg Execution time: %f ms\n", executionAvgTime);
            printf("Warning System  Avrg Request time: %f ms\n", ave_framedt1);
            printf("Average Calclated for : %d iterations\n",FRAMES);
            printf("Warning System  frequency  %f Hz\n", ave_frame_rate1);
            printf("(Min) Positive Jitter %f:\n" , avg_posJitter/FRAMES);
            printf("(Max) Negative Jitter %f:\n", avg_negJitter/FRAMES);
            printf("Total Avg. Jitter %f: \n", (avg_posJitter+avg_negJitter) /(double) FRAMES);*/
			
			printf("Warning System Avrg Execution time: %f ms\n", end_time1);
            printf("Warning System  Avrg Request time: %f ms\n", framedt1);
            printf("Warning System  frequency  %f Hz\n", 1000/framedt1);
            printf("(Min) Positive Jitter %f:\n" , avg_posJitter/FRAMES);
            printf("(Max) Negative Jitter %f:\n", avg_negJitter/FRAMES);
        }
    }
}

void *LaneDetect(void *threadp)
{
    // Start and end times ; we can also use the getTimeMsec

    int houghVote = 30;
    bool showSteps = 0;
    string window_name = "Processed Video";

    namedWindow(window_name, CV_WINDOW_KEEPRATIO); //resizable window
    double dWidth = 320;
    double dHeight =  240;

    std::cout << "Frame Size = " << dWidth << "x" << dHeight << std::endl;

    Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	
    VideoWriter output_cap(LANE_DETECT_OUTPUT_FILE_NAME,
               capture.get(CV_CAP_PROP_FOURCC),
               capture.get(CV_CAP_PROP_FPS),
               Size(capture.get(CV_CAP_PROP_FRAME_WIDTH),
               capture.get(CV_CAP_PROP_FRAME_HEIGHT)));
 
    if (!output_cap.isOpened())
    {
        std::cout << "!!! Output video could not be opened" << std::endl;
       // return -1;
    }
	Mat image;
    // Start time
    struct timespec frame_time1;
    double ave_framedt1=0.0, ave_frame_rate1=0.0,fc1=0.0,framedt1=0.0,jitter = 0.0, avg_posJitter = 0.0, avg_negJitter = 0.0;
    double curr_frame_time1 = 0.00, prev_frame_time1=0.00;
    unsigned int frame_count1=0;
    double executionAvgTime = 0;

    while (1)
    {
        sem_wait(&semLaneDetect);
        double start_time1 =  getTimeMsec();
        mFrameLaneDetect = mFrame.clone();

        //Timing Calculations per frame
        clock_gettime(CLOCK_REALTIME,&frame_time1);
        curr_frame_time1=((double)frame_time1.tv_sec * 1000.0) + ((double)((double)frame_time1.tv_nsec /1000000.0));

        frame_count1++;
        if(frame_count1 > 2)
        {
            fc1=(double)frame_count1;
            ave_framedt1=((fc1-1.0)*ave_framedt1 + framedt1)/fc1;
            ave_frame_rate1=1.0/(ave_framedt1/1000.0);
        }

        Mat gray;
        cvtColor(mFrameLaneDetect,gray,CV_RGB2GRAY);
        vector<string> codes;
        Mat corners;
        //findDataMatrix(gray, codes, corners); //doesnot work with Open Cv 3

        //drawDataMatrixCodes(image, codes, corners);//doesnot work with Open Cv 3


        Rect roi(0,mFrameLaneDetect.cols/3,mFrameLaneDetect.cols-1,mFrameLaneDetect.rows - mFrameLaneDetect.cols/3);// set the ROI for the image

        Mat imgROI = mFrameLaneDetect(roi);

        // Canny algorithm
        Mat contours;
        Canny(imgROI,contours,50,250);
        Mat contoursInv;
        threshold(contours,contoursInv,128,255,THRESH_BINARY_INV);



        /*

           Hough tranform for line detection with feedback

           Increase by 25 for the next frame if we found some lines.

           This is so we don't miss other lines that may crop up in the next frame

           but at the same time we don't want to start the feed back loop from scratch.

        */

        std::vector<Vec2f> lines;

        if (houghVote < 1 or lines.size() > 2) { // we lost all lines. reset
            houghVote = 200;
        }

        else {
            houghVote += 25;
        }

        while(lines.size() < 5 && houghVote > 0) {
            HoughLines(contours,lines,1,PI/180, houghVote);
            houghVote -= 5;
        }

        Mat result(imgROI.size(),CV_8U,Scalar(255));
        imgROI.copyTo(result);

        // Draw the limes
        std::vector<Vec2f>::const_iterator it= lines.begin();
        Mat hough(imgROI.size(),CV_8U,Scalar(0));

        while (it!=lines.end()) {
            float rho= (*it)[0];   // first element is distance rho
            float theta= (*it)[1]; // second element is angle theta

            if ( theta > 0.09 && theta < 1.48 || theta < 3.14 && theta > 1.66 ) { // filter to remove vertical and horizontal lines
                // point of intersection of the line with first row
                Point pt1(rho/cos(theta),0);
                // point of intersection of the line with last row
                Point pt2((rho-result.rows*sin(theta))/cos(theta),result.rows);
                // draw a white line
                line( result, pt1, pt2, Scalar(255), 8);
                line( hough, pt1, pt2, Scalar(255), 8);
            }
            ++it;
        }


        // Create LineFinder instance
        LineFinder ld;

        // Set probabilistic Hough parameters
        ld.setLineLengthAndGap(60,10);
        ld.setMinVote(4);
        // Detect lines
        std::vector<Vec4i> li= ld.findLines(contours);
        Mat houghP(imgROI.size(),CV_8U,Scalar(0));
        ld.setShift(0);
        ld.drawDetectedLines(houghP);

        // bitwise AND of the two hough images

        bitwise_and(houghP,hough,houghP);
        Mat houghPinv(imgROI.size(),CV_8U,Scalar(0));
        Mat dst(imgROI.size(),CV_8U,Scalar(0));
        threshold(houghP,houghPinv,150,255,THRESH_BINARY_INV); // threshold and invert to black lines
        if(showSteps) {
            namedWindow("Detected Lines with Bitwise");
            imshow("Detected Lines with Bitwise", houghPinv);
        }

        Canny(houghPinv,contours,100,350);
        li= ld.findLines(contours);
        // Set probabilistic Hough parameters

        ld.setLineLengthAndGap(5,2);
        ld.setMinVote(1);
        ld.setShift(mFrameLaneDetect.cols/3);
        ld.drawDetectedLines(mFrameLaneDetect);

        std::stringstream stream;
        stream << "Lines Segments: " << lines.size();
		if(lines.size()<5)
		{
			pthread_mutex_lock(&laneDetectmutex);
			LaneCrossed==YES;
			pthread_mutex_unlock(&laneDetectmutex);
		}
        putText(mFrameLaneDetect, stream.str(), Point(10,mFrameLaneDetect.rows-10), 2, 0.8, Scalar(0,0,255),0);
        imshow(window_name, mFrameLaneDetect);
        imwrite("processed.bmp", mFrameLaneDetect);

        output_cap.write(mFrameLaneDetect); //writer the frame into the file

        char key = (char) waitKey(10);
        lines.clear();

        double end_time1 =  getTimeMsec() - start_time1;
        //printf("Execution Time of Lane Detect thread: %f\n", end_time1);
        if(frame_count1!=0)executionAvgTime = ((executionAvgTime * (frame_count1-1)) + end_time1)/(double)frame_count1;
        framedt1=curr_frame_time1 - prev_frame_time1;
        prev_frame_time1=curr_frame_time1;
       if (frame_count1 > 0)
        {
            jitter = LANE_DETECT_DEADLINE - framedt1;
            if (jitter < 0)
            {
                //printf("\n Frame Time (ms): %f and Deadline missed for Frame : %d\n", framedt1, frame_count1);
                avg_negJitter  = jitter;//(avg_negJitter + jitter);
				avg_posJitter=0;
            }
            else
            {
                //printf("\n Frame %d finished earlier than Deadline\n", i);
                avg_posJitter = jitter;//(jitter + avg_posJitter);
				avg_negJitter=0;
            }

        }
        if(frame_count1 ==FRAMES) {
            frame_count1=0;
           /* printf("Lane Detection  Avrg Execution time per frame: %f ms\n", executionAvgTime);
            printf("Lane Detection  Avrg Request time: %f ms\n", ave_framedt1);
            printf("Frames Captured : %d\n",FRAMES);
            printf("Lane Detection frequency %f Hz\n", ave_frame_rate1);
            printf("(Min) Positive Jitter %f:\n" , avg_posJitter/FRAMES);
            printf("(Max) Negative Jitter %f:\n", avg_negJitter/FRAMES);
            printf("Total Avg. Jitter %f: \n", (avg_posJitter+avg_negJitter) /(double) FRAMES);*/
			
			printf("Lane Detection   Execution time per frame: %f ms\n", end_time1);
            printf("Lane Detection   Request time: %f ms\n", framedt1);
            printf("Lane Detection frequency %f Hz\n", 1000/framedt1);
            printf("(Min) Positive Jitter %f:\n" , avg_posJitter/FRAMES);
            printf("(Max) Negative Jitter %f:\n", avg_negJitter/FRAMES);
        }
    }
}



void print_scheduler(void)
{
    int schedType;

    schedType = sched_getscheduler(getpid());

    switch(schedType)
    {
    case SCHED_FIFO:
        printf("Pthread Policy is SCHED_FIFO\n");
        break;
    case SCHED_OTHER:
        printf("Pthread Policy is SCHED_OTHER\n");
        exit(-1);
        break;
    case SCHED_RR:
        printf("Pthread Policy is SCHED_RR\n");
        exit(-1);
        break;
    default:
        printf("Pthread Policy is UNKNOWN\n");
        exit(-1);
    }

}
int main()
{


    int i, rc, scope;
    cpu_set_t threadcpu;
    pthread_t threads[NUM_THREADS];
    threadParams_t threadParams[NUM_THREADS];
    pthread_attr_t rt_sched_attr[NUM_THREADS];
    int rt_max_prio, rt_min_prio;
    struct sched_param rt_param[NUM_THREADS];
    struct sched_param main_param;
    pthread_attr_t main_attr;
    pid_t mainpid;
    cpu_set_t allcpuset;

    if (sem_init (&semDanger, 0, 0)) {
        printf ("Failed to initialize semDanger semaphore\n");
        exit (-1);
    }
    if (sem_init (&semObjDetect, 0, 0)) {
        printf ("Failed to initialize  semObjDetect semaphore\n");
        exit (-1);
    }
    if (sem_init (&semLaneDetect, 0, 0)) {
        printf ("Failed to initialize semLaneDetect semaphore\n");
        exit (-1);
    }
    if (sem_init (&semWarningSys, 0, 0)) {
        printf ("Failed to initialize semWarningSys semaphore\n");
        exit (-1);
    }
    if (sem_init (&semVideoCap, 0, 0)) {
        printf ("Failed to initialize semVideoCap semaphore\n");
        exit (-1);
    }

    printf("System has %d processors configured and %d available.\n", get_nprocs_conf(), get_nprocs());

    CPU_ZERO(&allcpuset);

    for(i=0; i < NUM_CPU_CORES; i++)
        CPU_SET(i, &allcpuset);

    printf("Using CPUS=%d from total available.\n", CPU_COUNT(&allcpuset));


    pthread_mutex_init(&mutex, NULL);
	pthread_mutex_init(&laneDetectmutex,NULL);
    mainpid=getpid();

    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    rc=sched_getparam(mainpid, &main_param);
    main_param.sched_priority=rt_max_prio;
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    if(rc < 0) perror("main_param");
    print_scheduler();


    pthread_attr_getscope(&main_attr, &scope);

    if(scope == PTHREAD_SCOPE_SYSTEM)
        printf("PTHREAD SCOPE SYSTEM\n");
    else if (scope == PTHREAD_SCOPE_PROCESS)
        printf("PTHREAD SCOPE PROCESS\n");
    else
        printf("PTHREAD SCOPE UNKNOWN\n");

    printf("rt_max_prio=%d\n", rt_max_prio);
    printf("rt_min_prio=%d\n", rt_min_prio);


    cpu_set_t cpuset;
    int j=0;
    CPU_ZERO(&cpuset);
    CPU_SET(j, &cpuset);

    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (!(CPU_ISSET(j, &cpuset)))
        printf("Could not set the affinity\n");

    int core_count =0;
    for(i=0; i < NUM_THREADS; i++)
    {

        CPU_ZERO(&threadcpu);
#define _MULTI_CORE 1
#if _MULTI_CORE
        core_count++;
        if(core_count >= NUM_CPU_CORES)
        {   core_count = 1;
        }
#endif

        CPU_SET(core_count, &threadcpu);


        rc=pthread_attr_init(&rt_sched_attr[i]);
        rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
        rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);
        rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);

        rt_param[i].sched_priority=rt_max_prio-i;
        pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);

        threadParams[i].threadIdx=i;
    }

    printf("Service threads will run on %d CPU cores\n", CPU_COUNT(&threadcpu));

    // Create Service threads which will block awaiting release for:
    //
    // Create Video Capture thread, which like a cyclic executive, is highest prio
    printf("Start Video Capture\n");

    rc=pthread_create(&threads[1],               // pointer to thread descriptor
                      &rt_sched_attr[1],         // use specific attributes
                      //(void *)0,                 // default attributes
                      VideoCaptureThread,                 // thread function entry point
                      (void *)&(threadParams[1]) // parameters to pass in
                     );
// Create Warning Sytsem thread, which like a cyclic executive, is highest prio

    rc=pthread_create(&threads[4],               // pointer to thread descriptor
                      &rt_sched_attr[4],         // use specific attributes
                      //(void *)0,                 // default attributes
                      WarningSystem,                 // thread function entry point
                      (void *)&(threadParams[4]) // parameters to pass in
                     );

    // Object Detection thread
    rc=pthread_create(&threads[2],               // pointer to thread descriptor
                      &rt_sched_attr[2],         // use specific attributes
                      //(void *)0,                 // default attributes
                      ObstacleDetect,                     // thread function entry point
                      (void *)&(threadParams[2]) // parameters to pass in
                     );
    // Lane Detection system thread
    rc=pthread_create(&threads[3],               // pointer to thread descriptor
                      &rt_sched_attr[3],         // use specific attributes
                      //(void *)0,                 // default attributes
                      LaneDetect  ,                     // thread function entry point
                      (void *)&(threadParams[3]) // parameters to pass in
                     );

// Wait for service threads to calibrate and await relese by LaneDetect
    usleep(300000);

    rc= // Lane Detection system thread
        rc=pthread_create(&threads[0],               // pointer to thread descriptor
                          &rt_sched_attr[0],         // use specific attributes
                          //(void *)0,                 // default attributes
                          Sequencer  ,                     // thread function entry point
                          (void *)&(threadParams[0]) // parameters to pass in
                         );

    for(i=0; i<NUM_THREADS-1; i++)
    {
        pthread_join(threads[i], NULL);
        if(i ==0)
            pthread_join(threads[4], NULL);
    }
    return 0;
}


void * ObstacleDetect( void* threadp)
{
    struct timespec frame_time1;
    double ave_framedt1=0.0, ave_frame_rate1=0.0,fc1=0.0,framedt1=0.0,jitter = 0.0, avg_posJitter = 0.0, avg_negJitter = 0.0;
    double curr_frame_time1 = 0.00, prev_frame_time1=0.00;
    unsigned int frame_count1=0;
    cars.load(CASCADE_FILE_NAME);

    //traffic_light.load(CASCADE1_FILE_NAME);
    //stop_sign.load(CASCADE2_FILE_NAME);
    //pedestrian.load(CASCADE3_FILE_NAME);
    //sign.load(CASCADE4_FILE_NAME);
    //sign2.load(CASCADE5_FILE_NAME);
    VideoWriter obstacle_output(OBSTACLE_DETECT_OUTPUT_FILE_NAME,
               capture.get(CV_CAP_PROP_FOURCC),
               capture.get(CV_CAP_PROP_FPS),
               Size(capture.get(CV_CAP_PROP_FRAME_WIDTH),
               capture.get(CV_CAP_PROP_FRAME_HEIGHT)));
 
    if (!obstacle_output.isOpened())
    {
        std::cout << "!!! Output video could not be opened" << std::endl;
       // return;
    }
	
	
    double executionAvgTime = 0;
    while (1)//cap.read(mFrame))
    {
        sem_wait(&semObjDetect);

        double start_time1 =  getTimeMsec();
        mFrameObjDetect = mFrame.clone();
        //Timing Calculations per frame
        clock_gettime(CLOCK_REALTIME,&frame_time1);
        curr_frame_time1=((double)frame_time1.tv_sec * 1000.0) + ((double)((double)frame_time1.tv_nsec /1000000.0));

        frame_count1++;
        if(frame_count1 > 2)
        {
            fc1=(double)frame_count1;
            ave_framedt1=((fc1-1.0)*ave_framedt1 + framedt1)/fc1;
            ave_frame_rate1=1.0/(ave_framedt1/1000.0);
        }

        resize(mFrameObjDetect, mFrameObjDetect, Size(320, 240), 0, 0, INTER_CUBIC);


        // Apply the classifier to the frame


        imageROI = mFrameObjDetect(Rect(0,mFrameObjDetect.rows/2,mFrameObjDetect.cols,mFrameObjDetect.rows/2));

        cvtColor(imageROI, mGray, COLOR_BGR2GRAY);
        cvtColor(mFrameObjDetect, mGray2, COLOR_BGR2GRAY);


        //cars cascade
        cars.detectMultiScale(mGray, cars_found, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        if(cars_found.empty())
        {
            //printf("no cars detected\n");
        } else {
            draw_locations(mFrameObjDetect, cars_found, Scalar(0, 255, 0),"Car");
        }
        // draw_locations(mFrame2, cars_found, Scalar(0, 255, 0),"Car");

/*        //traffic lights cascade
        traffic_light.detectMultiScale(mGray, traffic_light_found, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        draw_locations(mFrameObjDetect, traffic_light_found, Scalar(0, 255, 255),"traffic light");

        //stop sign cascade
        stop_sign.detectMultiScale(mGray, stop_sign_found, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        draw_locations(mFrameObjDetect, stop_sign_found, Scalar(0, 0, 255),"Stop Sign");


        //pedestrian cascade
        pedestrian.detectMultiScale(mGray, pedestrian_found, 1.1, 1, 0 | CASCADE_SCALE_IMAGE, Size(20,50));
        draw_locations(mFrameObjDetect, pedestrian_found, Scalar(255, 0, 0),"Pedestrian");

        //stop sign cascade
        sign.detectMultiScale(mGray2, sign_found, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        draw_locations(mFrameObjDetect, sign_found, Scalar(0, 143, 255),"Left Arrow");

        sign2.detectMultiScale(mGray2, sign_found2, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        draw_locations(mFrameObjDetect, sign_found2, Scalar(0, 143, 255),"Right Arrow");
  */      

        imshow(WINDOW_NAME_1, mFrameObjDetect);
        waitKey(10);
		obstacle_output.write(mFrameObjDetect);
        double end_time1 =  getTimeMsec() - start_time1;
        //printf("Execution Time of Lane Detect thread: %f\n", end_time1);
        if(frame_count1!=0)executionAvgTime =((executionAvgTime * (frame_count1-1)) + end_time1)/(double)frame_count1;

        framedt1=curr_frame_time1 - prev_frame_time1;
        prev_frame_time1=curr_frame_time1;
       
		if (frame_count1 > 0)
        {
            jitter = OBJECT_DETECT_DEADLINE - framedt1;
            if (jitter < 0)
            {
                //printf("\n Frame Time (ms): %f and Deadline missed for Frame : %d\n", framedt1, frame_count1);
                avg_negJitter  = jitter;//(avg_negJitter + jitter);
				avg_posJitter=0;
            }
            else
            {
                //printf("\n Frame %d finished earlier than Deadline\n", i);
                avg_posJitter = jitter;//(jitter + avg_posJitter);
				avg_negJitter=0;
            }

        }
        if(frame_count1==FRAMES)
        {
            frame_count1=0;

           /* printf("Obstacle Detection Avrg Execution time per frame: %f ms\n", executionAvgTime);
            printf("Obstacle Detection Avrg Request time: %f ms\n", ave_framedt1);
            printf("Frames Captured : %d\n",FRAMES);
            printf("Obstacle Detection frequency %f Hz\n", ave_frame_rate1);
            printf("(Min) Positive Jitter %f:\n" , avg_posJitter/FRAMES);
            printf("(Max) Negative Jitter %f:\n", avg_negJitter/FRAMES);
            printf("Total Avg. Jitter %f: \n", (avg_posJitter+avg_negJitter) /(double) FRAMES);*/
			
			printf("Obstacle Detection  Execution time per frame: %f ms\n", end_time1);
            printf("Obstacle Detection  Request time: %f ms\n", framedt1);
            printf("Obstacle Detection frequency %f Hz\n", 1000/framedt1);
            printf("(Min) Positive Jitter %f:\n" , avg_posJitter/FRAMES);
            printf("(Max) Negative Jitter %f:\n", avg_negJitter/FRAMES);	
			
        }
    }
}
void draw_locations(Mat & img, vector< Rect > &locations, const Scalar & color, string text)
{

    Mat img1, car, carMask ,carMaskInv,car1,roi1, LeftArrow , LeftMask, RightArrow,RightMask;


    img.copyTo(img1);
    string dis;

    if (!locations.empty())
    {

        double distance= 0;

        for( int i = 0 ; i < locations.size() ; ++i) {

            if (text=="Car") {
                car = imread(CAR_IMAGE);
                carMask = car.clone();
                cvtColor(carMask, carMask, CV_BGR2GRAY);
                locations[i].y = locations[i].y + img.rows/2; // shift the bounding box
                distance = (RELATIVE_WIDTH*AVG_WIDTH_CAR)/((locations[i].width)*RELATIVE_PIXELS);// 2 is avg. width of the car
                Size size(locations[i].width/1.5, locations[i].height/3);
                resize(car,car,size, INTER_NEAREST);
                resize(carMask,carMask,size, INTER_NEAREST);
                Mat roi = img.rowRange(locations[i].y-size.height, (locations[i].y+locations[i].height/3)-size.height).colRange(locations[i].x, (locations[i].x  +locations[i].width/1.5));
                bitwise_and(car, roi, car);
                car.setTo(color, carMask);
                add(roi,car,car);
                car.copyTo(img1.rowRange(locations[i].y-size.height, (locations[i].y+locations[i].height/3)-size.height).colRange(locations[i].x, (locations[i].x  +locations[i].width/1.5)));

            } else if((text=="Pedestrian")) {
                distance = (RELATIVE_WIDTH*AVG_WIDTH_PERSON)/((locations[i].width)*RELATIVE_PIXELS);//0.5 is avg. width of a person
            } else if((text=="Stop Sign")) {
                distance = (RELATIVE_WIDTH*AVG_WIDTH_STOP_SIGH)/((locations[i].width)*RELATIVE_PIXELS);//0.75 is avg. width of the stopsign
            } else if((text=="Left Arrow")) {
                LeftArrow = imread(LEFT_SIGN_IMAGE);
                LeftMask = LeftArrow.clone();
                cvtColor(LeftMask, LeftMask, CV_BGR2GRAY);
                //locations[i].y = locations[i].y + img.rows/2; // shift the bounding box
                Size size(locations[i].width/2, locations[i].height/1.5);
                resize(LeftArrow,LeftArrow,size, INTER_NEAREST);
                resize(LeftMask,LeftMask,size, INTER_NEAREST);
                distance = (RELATIVE_WIDTH*AVG_WIDTH_SIGNS)/((locations[i].width)*RELATIVE_PIXELS);//0.35 is avg. width of the   Chevron Arrow sign

                if (locations[i].y-size.height>0) {

                    Mat roi1 = img.rowRange(locations[i].y-size.height,(locations[i].y+locations[i].height/1.5)-size.height).colRange(locations[i].x+5, (locations[i].x+5+locations[i].width/2));
                    bitwise_and(LeftArrow, roi1, LeftArrow);
                    LeftArrow.setTo(color, LeftMask);
                    add(roi1,LeftArrow,LeftArrow);
                    LeftArrow.copyTo(img1.rowRange(locations[i].y-size.height,(locations[i].y+locations[i].height/1.5)-size.height).colRange(locations[i].x+5 ,(locations[i].x +5+locations[i].width/2 )));
                }

            } else if((text=="Right Arrow")) {
                RightArrow = imread(RIGHT_SIGN_IMAGE);
                RightMask = RightArrow.clone();
                cvtColor(RightMask, RightMask, CV_BGR2GRAY);
                //locations[i].y = locations[i].y + img.rows/2; // shift the bounding box
                Size size(locations[i].width/2, locations[i].height/1.5);
                resize(RightArrow,RightArrow,size, INTER_NEAREST);
                resize(RightMask,RightMask,size, INTER_NEAREST);
                distance = (RELATIVE_WIDTH*AVG_WIDTH_SIGNS)/((locations[i].width)*RELATIVE_PIXELS);//0.35 is avg. width of the   Chevron Arrow sign

                if (locations[i].y-size.height>0) {

                    Mat roi1 = img.rowRange(locations[i].y-size.height,(locations[i].y+locations[i].height/1.5)-size.height).colRange(locations[i].x+5, (locations[i].x+5+locations[i].width/2));
                    bitwise_and(RightArrow, roi1, RightArrow);
                    RightArrow.setTo(color, RightMask);
                    add(roi1,RightArrow,RightArrow);
                    RightArrow.copyTo(img1.rowRange(locations[i].y-size.height,(locations[i].y+locations[i].height/1.5)-size.height).colRange(locations[i].x+5 ,(locations[i].x +5+locations[i].width/2 )));
                }

            }
            stringstream stream;
            stream << fixed << setprecision(2) << distance;
            dis = stream.str() + "m";
            rectangle(img,locations[i], color, -1);
        }
        addWeighted(img1, 0.8, img, 0.2, 0, img);

        for( int i = 0 ; i < locations.size() ; ++i) {

            rectangle(img,locations[i],color,1.8);

            putText(img, text, Point(locations[i].x+1,locations[i].y+8), FONT_HERSHEY_DUPLEX, 0.3, color, 1);
            putText(img, dis, Point(locations[i].x,locations[i].y+locations[i].height-5), FONT_HERSHEY_DUPLEX, 0.3, Scalar(255, 255, 255), 1);


            if (text=="Car") {
                locations[i].y = locations[i].y - img.rows/2; // shift the bounding box
                if(distance<DANGER_DISTANCE) {
                    pthread_mutex_lock(&mutex);
                    InDangerDistance = YES;
                    pthread_mutex_unlock(&mutex); //sem_wait(&semDanger);
                    //sem_post(&semDanger);

                }
            }

        }

    }
}


void * VideoCaptureThread(void * threadp)
{

    //VideoCapture capture(VIDEO_FILE_NAME); // open the video file for reading

    if ( !capture.isOpened() )  // if not success, exit program

    {

        cout << "Cannot open the video file" << endl;

    }
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    struct timespec frame_time1;
    double ave_framedt1=0.0, ave_frame_rate1=0.0,fc1=0.0,framedt1=0.0,jitter = 0.0, avg_posJitter = 0.0, avg_negJitter = 0.0;
    double curr_frame_time1 = 0.00, prev_frame_time1=0.00;
    unsigned int frame_count1=0;
    double executionAvgTime = 0;
    while(capture.read(mFrame)) {
        sem_wait(&semVideoCap);
        double start_time1 =  getTimeMsec();
        //Timing Calculations per frame
        clock_gettime(CLOCK_REALTIME,&frame_time1);
        curr_frame_time1=((double)frame_time1.tv_sec * 1000.0) + ((double)((double)frame_time1.tv_nsec /1000000.0));

        frame_count1++;
        if(frame_count1 > 2)
        {
            fc1=(double)frame_count1;
            ave_framedt1=((fc1-1.0)*ave_framedt1 + framedt1)/fc1;
            ave_frame_rate1=1.0/(ave_framedt1/1000.0);
        }

        double end_time1 =  getTimeMsec() - start_time1;
        //printf("Execution Time of Lane Detect thread: %f\n", end_time1);
        if(frame_count1!=0)executionAvgTime =((executionAvgTime * (frame_count1-1)) + end_time1)/(double)frame_count1;

        framedt1=curr_frame_time1 - prev_frame_time1;
        prev_frame_time1=curr_frame_time1;
        
		if (frame_count1 > 0)
        {
            jitter = VIDEO_CAPTURE_DEADLINE - framedt1;
            if (jitter < 0)
            {
                //printf("\n Frame Time (ms): %f and Deadline missed for Frame : %d\n", framedt1, frame_count1);
                avg_negJitter  = jitter;//(avg_negJitter + jitter);
				avg_posJitter=0;
            }
            else
            {
                //printf("\n Frame %d finished earlier than Deadline\n", i);
                avg_posJitter = jitter;//(jitter + avg_posJitter);
				avg_negJitter=0;
            }

        }
        if(frame_count1==FRAMES)
        {
            frame_count1=0;

            /*printf("Video Capture Avrg Execution time per frame : %f ms\n", executionAvgTime);
            printf("Video Capture Avrg Request time: %f ms\n", ave_framedt1);
            printf("Frames Captured : %d\n",FRAMES);
            printf("Video Capture frequency %f Hz\n", ave_frame_rate1);
            printf("(Min) Positive Jitter %f:\n" , avg_posJitter/FRAMES);
            printf("(Max) Negative Jitter %f:\n", avg_negJitter/FRAMES);
            printf("Total Avg. Jitter %f: \n", (avg_posJitter+avg_negJitter) /(double) FRAMES);*/
			
			printf("Video Capture Avrg Execution time per frame : %f ms\n", end_time1);
            printf("Video Capture Avrg Request time: %f ms\n", framedt1);
            printf("Video Capture frequency %f Hz\n", 1000/framedt1);
            printf("(Min) Positive Jitter %f:\n" , avg_posJitter/FRAMES);
            printf("(Max) Negative Jitter %f:\n", avg_negJitter/FRAMES);
        }
    }
}



