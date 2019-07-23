#include "smplx.h"


int main()
{
	Smplx smplx("../model/neutral");
	Smplx::Param param(smplx.conf);
	smplx.Debug(param,"../test.obj");
	return 0;
}