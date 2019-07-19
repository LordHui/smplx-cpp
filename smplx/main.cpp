#include "smplx.h"


int main()
{
	Smplx smplx("../model/neutral");
	Smplx::Param param;
	smplx.Debug(param,"../test.obj");
	return 0;
}