#pragma once


//Remember to extract the binary files, you dumbass!
void ProcessMNISTData( float* imageStorage, float* labelStorage, const char* labelFilepath, const char* imageFilepath, const unsigned int num , const unsigned int first = 0);

void ProcessLabel(float* storage, std::ifstream& data, const unsigned int labelSize, const unsigned pos, const unsigned first, const unsigned int start, unsigned end);
void ProcessImage(float* storage, std::ifstream& data, const unsigned int imageSize, const unsigned pos, const unsigned first, const unsigned int start, unsigned end);


void ProcessMNISTDataMT(unsigned threadCount,float* imageStorage, float* labelStorage, const char* labelFilepath, const char* imageFilepath, const unsigned int num, const unsigned int first=0);
void ProcessMNISTDataMTSlaveFunction(float* imageStorage, float* labelStorage, const char* labelFilepath, const char* imageFilepath, const unsigned int num, const unsigned int first, const unsigned int labelSize, const unsigned int imageSize, unsigned thread, unsigned threadCount);