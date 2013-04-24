#include "std_incl.h"
#include "resource.h"
#include "fastcmos.h"
#include "utils.h"
#include <Windows.h>

const char *WindowClass = "wndcls";
HWND window;
INTERFACE_ID ifid;
SESSION_ID session;
FastCMOS* fastCMOS;
HDC drawcontext, hdcMem;
HBITMAP hBitmap;
double last_update=0;
double update_freq=10;

void msgbox(const std::string& msg)
{
	MessageBox(0, msg.c_str(), "Msg:", MB_OK|MB_SYSTEMMODAL);
}

void InitDisplay(int w,int h)
{
	drawcontext = GetDC(window);

	BITMAPINFO bmi = { 0 };
	uchar* data = NULL;
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = w;
	bmi.bmiHeader.biHeight = h;
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 24;
	bmi.bmiHeader.biCompression = BI_RGB;
	hBitmap = CreateDIBSection(drawcontext, &bmi, DIB_RGB_COLORS, (LPVOID*)&data, NULL, 0);

	hdcMem = CreateCompatibleDC(NULL);

	
	// Green test circle
	for (int y=0;y<h;y++) {
		for (int x=0;x<w;x++) {
			int r2 = (x-w/2)*(x-w/2) + (y-h/2)*(y-h/2);
			data[ (y*w+x)*3 + 0 ] = 0;
			data[ (y*w+x)*3 + 1 ] = r2 < 50*50 ? 255 : 0;
			data[ (y*w+x)*3 + 2 ] = 0;
		}
	}
	SelectObject(hdcMem, hBitmap);

	BitBlt(GetDC(window), 0, 0, w,h, hdcMem, 0,0,  SRCCOPY);
}

void CloseDisplay()
{
	if (hBitmap) {
		DeleteDC(hdcMem);
		DeleteObject(hBitmap);
		hBitmap = 0;
	}
}

void CameraClose()
{
	fastCMOS->stop();

	delete fastCMOS;
	fastCMOS = 0;

	imgClose(session, 0);
	imgClose(ifid, 1);

	CloseDisplay();
}


void CameraConfigure()
{
	CloseDisplay();

	int w = 600, h= 600;

	fastCMOS->setFramerate(10);
	int fps=  fastCMOS->getFramerate().fps;

	ROI rfit = fastCMOS->setROI(ROI(w,h));
	ROI roi = fastCMOS->getROI();
	fastCMOS->setup(50);
	fastCMOS->start();
	//IMAQBuffer* buf = fastCMOS->snap();

	int mode = fastCMOS->getMode();
	
	InitDisplay(w,h);

//	IMAQBuffer* buf = fastCMOS->snap();
	

//	void *buf = fastCMOS->getLastFrame();
}


void CameraOpen()
{
	char name[256];
	for (int i=0;;i++) {
		if (0 != imgInterfaceQueryNames(i, name))
			break;

		dbgprintf("IMAQ interface: %s\n", name);
	}

	if (fastCMOS)
		CameraClose();

	imgInterfaceOpen("img0", &ifid);
	imgSessionOpen(ifid, &session);

	fastCMOS = new FastCMOS(session);
	std::string info = fastCMOS->readInfo();

	CameraConfigure();

	int fps = fastCMOS->getFramerate().fps;
	dbgprintf("FPS: %d\n", fps);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_COMMAND:{
		int wmId    = LOWORD(wParam);
		int wmEvent = HIWORD(wParam);
		// Parse the menu selections:
		switch (wmId)
		{
		case IDM_EXIT:
			DestroyWindow(hWnd);
			break;
		case ID_CAMERA_OPEN:
			CameraOpen();
			break;
		case ID_CAMERA_CLOSE:
			CameraClose();
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		break;}
	case WM_PAINT:{
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hWnd, &ps);

		EndPaint(hWnd, &ps);
		break;}
	case WM_CLOSE:
		DestroyWindow(hWnd);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

void Update()
{
	double t = GetPreciseTime();
	if (t-last_update > 1.0f/update_freq) {

		if (fastCMOS) {
			int fc = fastCMOS->getFramecount();

			dbgprintf("Frames: %d\n", fc);
		}
		last_update = t;
	}
}

void Init(HINSTANCE instance)
{
	WNDCLASSEX wcex;
	memset(&wcex, 0, sizeof(WNDCLASSEX));
	wcex.cbSize = sizeof(WNDCLASSEX);
	const char *WindowClass = "test";

	wcex.style			= CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc	= WndProc;
	wcex.hInstance		= instance;
	wcex.lpszMenuName	= MAKEINTRESOURCE(IDR_MENU);
	wcex.lpszClassName	= WindowClass;

	ATOM cls = RegisterClassEx(&wcex);

	window = CreateWindow(WindowClass, "Fast grab test", WS_OVERLAPPEDWINDOW, 
		CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, instance, NULL);
	
	ShowWindow(window, SW_SHOW);
	UpdateWindow(window);
}

int APIENTRY WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPTSTR cmdLine, int nCmdShow)
{
	Init(hInst);

	MSG msg;

	// Main message loop:
	while (1)
	{
		while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT) {
				break;
			}

			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		if(msg.message == WM_QUIT)
			break;

		Update();
	}

	return (int) msg.wParam;
}

