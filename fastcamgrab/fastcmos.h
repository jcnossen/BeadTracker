#pragma once

#include "niimaq.h"
#include "utils.h"

class FastCMOS
{
public:

	FastCMOS(SESSION_ID session);
	~FastCMOS();

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

	int getFramerate(); 
	int getShuttertime();


	bool isFramecounterEnabled() { return icmd(":u?") == 1; }
	void setFramecounter(bool v) { cmd(v ? ":u1" : ":u0" ); }

protected:
	SESSION_ID session;
	std::string readBuf;

};


