#define SIZE_OF_HAND_DATA 8


typedef struct finger_data {
	Point2d origin_offset;		//base or finger relative to center hand
	double a;					//angle
	vector<double> joints_a;	//angles of joints
	vector<double> joints_d;	//bone length
} FINGER_DATA;

typedef struct hand_data {
	FINGER_DATA fingers[5];		//fingers
	double a;					//angle of whole hand
	Point2d origin;				//center of palm
	Point2d origin_offset;		//offset from center for optimization
	double size;				//relative size of hand = length of a finger
	double palm_size;
} HAND_DATA;

typedef struct data_for_tnc {
	vector<Point2d> targets;	//points to reach
	HAND_DATA hand;
	Mat contour;
	Mat hand_blob;				//8bit (1bit) mask of the hand
	
	bool initialized;
} DATA_FOR_TNC;