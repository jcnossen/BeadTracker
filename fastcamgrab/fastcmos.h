#pragma once

#include "niimaq.h"
#include "utils.h"

class IMAQBuffer {
public:
	IMAQBuffer() { sid=0; data=0; }
	~IMAQBuffer();

	void *data;
	SESSION_ID sid;
};

struct ROI
{
	ROI() { x=y=w=h=0; }
	ROI(int W,int H) { x=y=0; w=W;h=H; }
	int x,y,w,h;
};

class FastCMOS
{
public:

	enum { MAX_ROI=4 };

	FastCMOS(SESSION_ID session);
	~FastCMOS();

	// Set ROI first!
	void setup(int nbuffers);

	void write(const std::string& cmd);
	std::string read();

	float readTemp() { return atof( cmd(":T").c_str() ); } 
	std::string readInfo() { return cmd(":v"); }
	std::string cmd(const std::string& input);
	int icmd(const std::string& input);

	int getMode() { return icmd(":M?"); }
	void setMode(int m) { cmd(SPrintf(":M%x",m)); }

	int getGain() { return icmd(":D?"); }
	void setGain(int gain) { cmd(SPrintf(":D%04x")); }

	int getLostFrames();
	int getFramecount();

	struct FramerateInfo { int maxfps, fps; };
	FramerateInfo getFramerate(); 
	void setFramerate(int fps);
	int getShuttertime();

	bool isFramecounterEnabled() { return icmd(":u?") == 1; }
	void setFramecounter(bool v) { cmd(v ? ":u1" : ":u0" ); }

	ROI setROI(ROI roi); // returns fitted ROI
	ROI setROI(int nroi, ROI* rois);
	ROI getROI();
	ROI getAcqWindow();
	void *getLastFrame();
	void start();
	void stop();

	IMAQBuffer* snap();

protected:
	ROI roi[MAX_ROI];
	int numROI, nbuffers;
	SESSION_ID session;
	std::string readBuf;
	BUFLIST_ID buflist;
	std::vector<void*> buffers;
	int curBuf;

	void checkIMAQ(Int32 errc);
};


