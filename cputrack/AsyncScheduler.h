/*
Both the CUDA tracker and CPU tracker will copy the image directly to their own memory storage.
The AsyncScheduler is used to deal with the async calls to ScheduleFrame.
*/
#pragma once

#include "QueuedTracker.h"
#include "threads.h"
#include "hash_templates.h"

class AsyncScheduler
{
public:
	AsyncScheduler(QueuedTracker* qtrk) {
		thread = Threads::Create( (Threads::ThreadEntryPoint)ThreadFunc, NULL);
		closeScheduler = false;
		tracker = qtrk;
	}

	~AsyncScheduler()
	{
		closeScheduler = true;
		Threads::WaitAndClose(thread);
	}


	struct Frame {
		Frame() { imgptr = 0; }
		uchar *imgptr;
		int pitch;
		int width;
		int height;
		std::vector<ROIPosition> positions;
		QTRK_PixelDataType pdt;
		LocalizationJob jobInfo;
	};

	void Schedule(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) {
		Frame *cp;
		mutex.lock();
			if (framebuf.empty()) {
				cp = new Frame();
			} else {
				cp = framebuf.back();
				framebuf.pop_back();
			}
			cp->imgptr = imgptr;
			cp->pitch = pitch;
			cp->width = width;
			cp->height = height;
			cp->positions.assign(positions,positions+numROI);
			cp->pdt= pdt;
			cp->jobInfo= *jobInfo;
				 
			queue.push_back(cp);
		mutex.unlock();
		imageMutex.lock();
			images.insert(f.imgptr);
		imageMutex.unlock();
	}

	void WaitForImage(uchar *ptr)
	{
		while (!closeScheduler) {
			imageMutex.lock();
			if (images.find(ptr) == images.end())
				return;
			imageMutex.unlock();
		}
	}

	bool IsFinished(uchar* ptr)
	{
		imageMutex.lock();
		bool r = images.find(ptr) == images.end();
		imageMutex.unlock();
		return r;
	}

protected:
	std::deque<Frame*> queue;
	std::vector<Frame*> framebuf;
	Threads::Handle* thread;
	Threads::Mutex mutex; // locking framebuf and queue
	Threads::Mutex imageMutex; // locking images
	qtrk::hash_set<uchar*> images;// image pointers currently locked for copying
	bool closeScheduler;
	QueuedTracker* tracker;

	void Process()
	{
		auto f = queue.front();
		mutex.lock();
		queue.pop_front();
		mutex.unlock();

		tracker->ScheduleFrame(f->imgptr, f->pitch, f->width, f->height, &f->positions[0], f->positions.size(), f->pdt, &f->jobInfo);

		imageMutex.lock();
		images.erase(f->imgptr);
		imageMutex.unlock();

		mutex.lock();
		framebuf.push_back(f);
		mutex.unlock();
	}

	static Threads::ReturnValue ThreadFunc(AsyncScheduler* this_)
	{
		while(!this_->closeScheduler) {
			this_->Process();
		}
		return 0;
	}

};



