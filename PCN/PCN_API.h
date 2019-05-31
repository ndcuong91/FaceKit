#ifndef __PCN_API__
#define __PCN_API__
#include "PCN.h"

struct CPoint{
	int x;
	int y;
};

struct CWindow{
	int x, y, width, angle;
	float score;
	struct CPoint points[14];

	void from_window(Window win)
	{
		x = win.x;
		y = win.y;
		width = win.width;
		angle = win.angle;
		score = win.score;
		for (int f=0; f< win.points14.size(); f++){
			points[f].x = win.points14[f].x;
			points[f].y = win.points14[f].y;
		}
	}
};


extern "C"
{
//C API stuff
#ifndef MODEL_PATH
#define MODEL_PATH "./model"
#endif
	void *init_detector(int min_face_size, float pyramid_scale_factor, float detection_thresh_stage1,
			float detection_thresh_stage2, float detection_thresh_stage3, int tracking_period,
			float tracking_thresh, int do_smooth)
	{
		PCN *detector = new PCN(MODEL_PATH"/PCN.caffemodel",
				MODEL_PATH"/PCN-1.prototxt", MODEL_PATH"/PCN-2.prototxt", MODEL_PATH"/PCN-3.prototxt",
				MODEL_PATH"/PCN-Tracking.caffemodel",
				MODEL_PATH"/PCN-Tracking.prototxt");
		/// detection
		detector->SetMinFaceSize(min_face_size);
		detector->SetImagePyramidScaleFactor(pyramid_scale_factor);
		detector->SetDetectionThresh(
				detection_thresh_stage1,
				detection_thresh_stage2,
				detection_thresh_stage3);
		/// tracking
		detector->SetTrackingPeriod(tracking_period);
		detector->SetTrackingThresh(tracking_thresh);
		detector->SetVideoSmooth((bool)do_smooth);
		//detector->SetIOUThresh(iou_high_thresh,iou_low_thresh);
		return static_cast<void*> (detector);
	}

	CWindow* detect_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, int *lwin)
	{
		PCN* detector = (PCN*) pcn;
		cv::Mat img(rows,cols, CV_8UC3, (void*)raw_img);
		std::vector<Window> faces = detector->DetectTrack(img);

		*lwin = faces.size();
		CWindow* wins = (CWindow*)malloc(sizeof(CWindow)*(*lwin));
		for (int i=0; i < *lwin; i++){
			wins[i].from_window(faces[i]);
		}
		return wins;
	}

	void free_faces(CWindow* wins)
	{
		free(wins);
	}

	void free_detector(void *pcn)
	{
		PCN* detector = (PCN*) pcn;
		delete detector;
	}



}



#endif

