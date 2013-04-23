#include "std_incl.h"

#include "fastcmos.h"


FastCMOS::FastCMOS(SESSION_ID sid)
{
	session = sid;
}

FastCMOS::~FastCMOS()
{
}

void FastCMOS::write(const std::string& s)
{
	uInt32 sz = s.size();
	imgSessionSerialWrite (session,  (Int8*) s.c_str(), &sz, 0);
}

std::string FastCMOS::cmd(const std::string& input)
{
	write(input);
	return read();
}

int FastCMOS::icmd(const std::string& input)
{
	write(input);
	std::string r = read();
	return atoi(r.c_str());
}

std::string FastCMOS::read()
{
	char buf[256];
	uInt32 bufsize=sizeof(buf);
	imgSessionSerialRead(session, buf, &bufsize, 100);
	return std::string(buf, buf+bufsize);
}



int FastCMOS::getFramerate()
{
	std::string r = cmd(":q?");

	int actual, minv, maxv;
	//sscanf_s(r.c_str(), "%h %d-%h", &actual, &minv, &maxv);
	return actual;
}

int FastCMOS::getShuttertime()
{
	std::string r = cmd(":t?");
	return 0;
}


