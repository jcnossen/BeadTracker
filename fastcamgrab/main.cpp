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

void msgbox(const std::string& msg)
{
	MessageBox(0, msg.c_str(), "Msg:", MB_OK);
}

void CameraClose()
{
	delete fastCMOS;
	fastCMOS = 0;

	imgClose(session, 0);
	imgClose(ifid, 1);
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
	msgbox(info);

	int fps = fastCMOS->getFramerate();
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
	}

	return (int) msg.wParam;
}

