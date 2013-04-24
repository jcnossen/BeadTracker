#include "std_incl.h"
#include "fastcmos.h"
#include "threads.h"

IMAQBuffer::~IMAQBuffer(){
	if (data) imgDisposeBuffer(data);
}

FastCMOS::FastCMOS(SESSION_ID sid)
{
	session = sid;
	numROI = 0;
	buflist = 0;
	curBuf =0 ;
}

FastCMOS::~FastCMOS()
{
	if (buflist)  {
		checkIMAQ(imgDisposeBufList(buflist, 0));
		buflist = 0;
	}
}

void FastCMOS::write(const std::string& s)
{
	uInt32 sz = s.size();
	imgSessionSerialWrite (session, (Int8*) s.c_str(), &sz, 0);
}

std::string FastCMOS::cmd(const std::string& input)
{
	write(input+"\r");
	return read();
}

int FastCMOS::icmd(const std::string& input)
{
	write(input+"\r");
	std::string r = read();
	return atoi(r.c_str());
}

std::string FastCMOS::read()
{
	char buf[256];
	uInt32 bufsize=sizeof(buf);
	Threads::Sleep(100);
	imgSessionSerialReadBytes(session, buf, &bufsize, 500);
	return std::string(buf, buf+bufsize);
}


FastCMOS::FramerateInfo FastCMOS::getFramerate()
{
	std::string r = cmd(":q?");

	int actual, minv, maxv;
	sscanf_s(r.c_str(), "%x %d-%x", &actual, &minv, &maxv);
	FramerateInfo info;
	info.fps = actual;
	info.maxfps = maxv;
	return info;
}

void FastCMOS::setFramerate(int fps)
{
	cmd(SPrintf(":q%06x", fps));
}

int FastCMOS::getShuttertime()
{
	std::string r = cmd(":t?");
	return 0;
}

IMAQBuffer* FastCMOS::snap()
{
	IMAQBuffer* buf = new IMAQBuffer();

	buf->sid = session;
	checkIMAQ(imgSnap(session, &buf->data));
	return buf;
}

ROI FastCMOS::setROI(ROI roi)
{
	ROI rfit;
	checkIMAQ(imgSessionFitROI(session, IMG_ROI_FIT_LARGER, roi.y, roi.x, roi.h, roi.w, (uInt32*) &rfit.y, (uInt32*)&rfit.x, (uInt32*)&rfit.h, (uInt32*)&rfit.w));
	checkIMAQ(imgSessionSetROI(session, rfit.y, rfit.x, rfit.h, rfit.w));
	numROI = 1;
	this->roi[0] = rfit;
	return rfit;
}

ROI FastCMOS::getROI()
{
	ROI r;
	checkIMAQ(imgGetAttribute(session, IMG_ATTR_ROI_LEFT, &r.x));
	checkIMAQ(imgGetAttribute(session, IMG_ATTR_ROI_TOP, &r.y));
	checkIMAQ(imgGetAttribute(session, IMG_ATTR_ROI_WIDTH, &r.w));
	checkIMAQ(imgGetAttribute(session, IMG_ATTR_ROI_HEIGHT, &r.h));

	return r;
}

void FastCMOS::setup(int nbuffers)
{
	this->nbuffers = nbuffers;

	buffers.resize(nbuffers);
	checkIMAQ(imgRingSetup(session, nbuffers, &buffers[0], 0, 0));
}

void FastCMOS::start()
{
	checkIMAQ(imgSessionStartAcquisition(session));
}

void FastCMOS::stop()
{
	checkIMAQ(imgSessionStopAcquisition(session));
}

int FastCMOS::getLostFrames()
{
	uInt32 lost;
	checkIMAQ(imgGetAttribute(session, IMG_ATTR_LOST_FRAMES, &lost));
	return lost;
}


int FastCMOS::getFramecount()
{
	uInt32 v;
	checkIMAQ(imgGetAttribute(session, IMG_ATTR_FRAME_COUNT, &v));
	return v;
}


ROI FastCMOS::getAcqWindow()
{
	ROI roi;

	checkIMAQ(imgGetAttribute(session, IMG_ATTR_ACQWINDOW_LEFT, &roi.x));
	checkIMAQ(imgGetAttribute(session, IMG_ATTR_ACQWINDOW_TOP, &roi.y));
	checkIMAQ(imgGetAttribute(session, IMG_ATTR_ACQWINDOW_WIDTH, &roi.w));
	checkIMAQ(imgGetAttribute(session, IMG_ATTR_ACQWINDOW_HEIGHT, &roi.h));
	
	return roi;
}

void FastCMOS::checkIMAQ(Int32 errc)
{
	if (errc != 0) {
		char errmsg [256];
		imgShowError(errc, errmsg);
		throw std::runtime_error(std::string("IMAQ error: ") + errmsg);
	}
}


void* FastCMOS::getLastFrame() {

	/*
	imgSessionAcquire(session, 0, 0);
	void *r = buffers[curBuf];
	curBuf = (curBuf+1) % buffers.size();*/

	uInt32 bufnum;
	void* bufaddr;

	checkIMAQ(imgSessionExamineBuffer2(session, IMG_CURRENT_BUFFER, &bufnum, &bufaddr));

	dbgprintf("bufnum: %d\n",bufnum);

	return bufaddr;
}
