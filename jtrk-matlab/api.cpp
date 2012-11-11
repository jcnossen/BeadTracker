
#include "utils.h"
#include "QueuedTracker.h"
#include "Matrix.h"
#include <map>

// Pointers vary in size
#ifdef _M_X64
	#define PTR_CLASS_ID mxUINT64_CLASS
#else
	#define PTR_CLASS_ID mxUINT32_CLASS
#endif

#ifdef _CHAR16T
#define CHAR16_T wchar_t
#endif


//#define __STDC_UTF_16__ // somehow matlab always promotes hacks
#include <mex.h>
#include <mat.h>



#include <stdarg.h>

class arg_err : public std::runtime_error {
public:
	arg_err(const std::string& v) : std::runtime_error(v) {}
};

#define ARG_CHECK(c) if (!(c)) {mexErrMsgTxt(#c " failed."); return; }
#define CHECK(v) check(v, #v)

static void check(bool v, const std::string& msg) {
	if(!v) throw arg_err(msg.c_str());
}

#define MEX_CB(_NAME) \
void MexInstance::_NAME(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { \
const char *_FMTSTR = "Exception in " #_NAME ": %s"; try {

#define MEX_END_CB() \
} catch(const std::runtime_error& e) { \
	mexPrintf(_FMTSTR, e.what());	\
	} }



std::string getString(const mxArray *ar)
{
	char* str = mxArrayToString(ar);
	std::string r = str;
	mxFree(str);
	return r;
}

void parseMatrixf(Matrixf* m, const mxArray* v)
{
	int cols = mxGetN(v);
	int rows = mxGetM(v);
	m->init(cols, rows);
	double* src = mxGetPr(v);
	for (int y=0;y<rows;y++)
		for (int x=0;x<cols;x++) 
			m->elem(x,y) = src[x*rows+y];
}

template<typename T>
mxArray* convertMatrix(const Matrix<T>& m) {
	mxArray* var = mxCreateDoubleMatrix(m.h, m.w, mxREAL);
	double* d = mxGetPr(var);
	for (int y=0;y<m.h;y++)
		for (int x=0;x<m.w;x++) // copy and transpose
			d[x*m.h+y] = m.elem(x,y);
	return var;
}

mxArray* createPointerValue(void *p) {
	mxArray*v = mxCreateNumericMatrix(1,1, PTR_CLASS_ID, mxREAL);
	void* mem = mxCalloc(1, sizeof(void*));
	*((void**)mem)=p;
	mxSetData(v, mem);
	return v;
}

void* getPointerValue(const mxArray* v) {
	void* d = mxGetData(v);
	if (!d)
		return 0;
	return *((void**)d);
}


struct MexInstance {
	MexInstance() {
		magic = 0xe839dfe; // magic ID, to recognise if user is passing a valid pointer
		qtrk = 0;
	}
	~MexInstance() {
		if(qtrk) delete qtrk;
	}

	static MexInstance* GetPtr(const mxArray* v) {
		MexInstance* inst = (MexInstance*)getPointerValue(v);
		if (inst->magic != 0xe839dfe)
			throw arg_err("Invalid MIMIL instance passed");
		return inst;
	}

	int magic;
	QueuedTracker* qtrk;

	void create(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
	void free(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
	void addImage(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
	void getResults(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
	void start(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
	void stop(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

	void checkInit() {
		if (!qtrk) 
			throw arg_err("Initialization not done.");
	}

	void parseArgs(int nrhs, const mxArray* prhs[], const char* fmt,...);
};



/*
Parse arguments based on a format string:
i		Integer
f		Scalar float
g		Scalar defined by USE_FLOAT
c		Character array -> std::string
m		2D Matrix -> Matrixf
p		Pointer value
l		Column vector stored in mvecd
?		All following arguments are optional
*/
void MexInstance::parseArgs(int nrhs, const mxArray* prhs[], const char* fmt,...)
{
	va_list l;
	va_start(l, fmt);

	int nexpect = 0;
	for (int i=0;fmt[i] && fmt[i]!='?';i++)
		if (fmt[i]!='S') nexpect++;

	if (nexpect > nrhs-1) {
		throw arg_err(SPrintf("Expecting %d arguments, %d given", nexpect+1, nrhs));
	}

	bool skipInitCheck = false;
	int argp=2;
	for (int a=0;fmt[a];a++) {

		if (fmt[a] == '?') continue;
		if (fmt[a] == 'S') {
			skipInitCheck=true;
			continue;
		}

		if (argp == nrhs)
			break;

		const mxArray* v=prhs[argp++];
		int m=mxGetM(v);
		int n=mxGetN(v);

		switch(fmt[a]) {
		case 'i':{
			check(mxIsNumeric(v) && m*n==1, SPrintf("Expecting scalar value for argument %d", a+1));
			int *val = va_arg(l, int*);
			*val = mxGetScalar(v);
			break;}
/*		case 'l': {
			check(mxIsNumeric(v) && m==1, SPrintf( "Expecting column vector for argument %d", a+1));
			mvecd *val = va_arg(l,mvecd*);
			val->resize(n);
			double* p = mxGetPr(v);
			for (int i=0;i<val->size();i++) (*val)[i] = p[i];
			break;}*/
		case 'f': 
		case 'g':
			{
			check(mxIsNumeric(v) && m*n==1, SPrintf("Expecting scalar value for argument %d", a+1));
			float *val=va_arg(l,float*);
			*val=mxGetScalar(v);
			break;}
		case 'c':{
			std::string *str = va_arg(l, std::string*);
			check(mxIsChar(v), "Expecting character array");
			*str = getString(v);
			break;}
		case 'm':{
			Matrixf* m = va_arg(l, Matrixf*);
			check(mxIsNumeric(v), "Expecting matrix value");
			parseMatrixf(m, v);
			break;}
		case 'p':{
			void** ptrAddr = va_arg(l, void**);
			check(mxIsUint32(v) || mxIsUint64(v), "Expecting pointer value");
			*ptrAddr = getPointerValue(v);
			break; }
		default:{
			assert(0);
			break;}
		}
	}
	va_end(l);

	if (!skipInitCheck) {
		if (!qtrk)
			throw arg_err("Call init() first");
	}

/*	check(mxIsUint32(prhs[1]) || mxIsUint64(prhs[1]), "Expecting pointer value");
	MexInstance* inst = MexInstance::GetPtr(prhs[1]);
	return inst;*/
}


static void set_value(float& dst, mxArray* src) { dst = mxGetScalar(src); }
static void set_value(int& dst, mxArray* src) { dst = (int)mxGetScalar(src); }
static mxArray* make_mxArray(double d) { return mxCreateDoubleScalar(d); }

static void map_QTrkSettings(QTrkSettings *cfg, bool get, std::map<std::string, mxArray*>& values)
{
#define CFGITEM(_VAL) \
	if(get) values[#_VAL] = make_mxArray(cfg->_VAL); \
	else if (values.find(#_VAL) != values.end()) set_value (cfg->_VAL, values[#_VAL]);

	CFGITEM(width);
	CFGITEM(height);
	CFGITEM(numThreads);
	CFGITEM(maxQueueSize);

	CFGITEM(xc1_profileLength);
	CFGITEM(xc1_profileWidth);
	CFGITEM(xc1_iterations);

	CFGITEM(zlut_minradius);
	CFGITEM(zlut_maxradius);
	CFGITEM(zlut_angularsteps);

	CFGITEM(qi_iterations);
	CFGITEM(qi_radialsteps);
	CFGITEM(qi_angularsteps);
	CFGITEM(qi_minradius);
	CFGITEM(qi_maxradius);

#undef CFGITEM
}

MEX_CB(start) {
	parseArgs(nrhs, prhs, "");

	const mxArray* cfg = prhs[2];
	if (nrhs < 3 || !mxIsStruct(cfg)) {
		throw arg_err("Expecting structure with config values for argument 3: [ 'setgeom', instance, cfg ]");
	}

	int nf = mxGetNumberOfFields(cfg);
	std::map<std::string, mxArray*> values;
	for (int x=0;x<nf;x++) {
		mxArray* fieldVal = mxGetFieldByNumber(cfg, 0, x);
		const char* fieldName = mxGetFieldNameByNumber(cfg, x);
		values[fieldName] = fieldVal;
	}
	QTrkSettings settings;
	map_QTrkSettings(&settings, false, values);

	if (nlhs>0) {
//		plhs[0]=convertMatrix(hessian);
	}
}MEX_END_CB()


MEX_CB(free) {
}MEX_END_CB()

MEX_CB(create) {
//	qtrk = CreateQueuedTracker(
}MEX_END_CB()

MEX_CB(addImage) {
}MEX_END_CB()

typedef void (MexInstance::*MatlabCmd)(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

struct mlcmd  {
	MatlabCmd cmdfunc;
	const char *name, *desc;
};

static mlcmd cmds[] = {
	{ &MexInstance::addImage, "addimage", "Schedule image for localization"},
	{ &MexInstance::start, "start", "Start tracker process (in background)"},
	{ &MexInstance::create, "create", "Create tracker instance"},
	{ &MexInstance::free, "free", "Free all tracker memory"},
	{ 0 }
};

void showHelp()
{
	mexPrintf("Syntax: jtrk('COMMAND', command arguments...) \n");
	mexPrintf("Accepted commands:\n");

	for (int i=0;cmds[i].cmdfunc;i++) {
		mexPrintf("'%s': %s\n", cmds[i].name, cmds[i].desc);
	}
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	static std::map<std::string, MatlabCmd> cmdMap;

	if (cmdMap.empty()) {
		for (int i=0;cmds[i].cmdfunc;i++)
			cmdMap[cmds[i].name] = cmds[i].cmdfunc;
	}

	if (nrhs==0)
		showHelp();
	else {
		ARG_CHECK(mxIsChar(prhs[0]));
		
		try {
			std::string cmd = getString(prhs[0]);
			if (cmd == "new") {
				MexInstance* inst = new MexInstance();
				if (nlhs>0) {
					plhs[0]=createPointerValue(inst);
				}
			} else {
				MexInstance* inst = MexInstance::GetPtr(prhs[1]);
				std::map<std::string, MatlabCmd>::iterator cmdIt = cmdMap.find(cmd);
				if (cmdIt != cmdMap.end()) {
					(inst->*(cmdIt->second))(nlhs,plhs,nrhs,prhs);
				} else
					mexErrMsgTxt(SPrintf("No command named %s\n", cmd.c_str()).c_str());
			}
		}
		catch (const arg_err& err) {
			mexErrMsgTxt(err.what());
		}
	}
}

