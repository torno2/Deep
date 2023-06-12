#include "pch.h"
#include "DataProcessing.h"

void ProcessMNISTData(float* imageStorage, float* labelStorage, const char* labelFilepath, const char* imageFilepath, const unsigned int num , const unsigned int first)
{

    std::ifstream labelData(labelFilepath, std::ios::binary);
    std::ifstream imageData(imageFilepath, std::ios::binary);



    unsigned int labelCount;
    char labelCountBuffer[sizeof(unsigned int)];

    labelData.seekg(4);
    labelData.read(labelCountBuffer, sizeof(labelCountBuffer));
    char flipedBuffer[4] = { labelCountBuffer[3],labelCountBuffer[2],labelCountBuffer[1],labelCountBuffer[0] };
    memcpy(&labelCount, &flipedBuffer, sizeof(labelCount));


    unsigned int imageCount;
    unsigned int rowCount;
    unsigned int columnCount;
    char intBuffer[sizeof(unsigned int)];

    imageData.seekg(4);
    imageData.read(intBuffer, sizeof(intBuffer));
    char flipedBuffer1[4] = { intBuffer[3],intBuffer[2],intBuffer[1],intBuffer[0] };
    memcpy(&imageCount, &flipedBuffer1, sizeof(imageCount));

    imageData.seekg(8);
    imageData.read(intBuffer, sizeof(intBuffer));
    char flipedBuffer2[4] = { intBuffer[3],intBuffer[2],intBuffer[1],intBuffer[0] };
    memcpy(&rowCount, &flipedBuffer2, sizeof(rowCount));

    imageData.seekg(12);
    imageData.read(intBuffer, sizeof(intBuffer));
    char flipedBuffer3[4] = { intBuffer[3],intBuffer[2],intBuffer[1],intBuffer[0] };
    memcpy(&columnCount, &flipedBuffer3, sizeof(columnCount));


    const unsigned labelSize = 10;
    const unsigned int imageSize = columnCount * rowCount;

    //If this triggers you most likely have missmatching files, or at the very least one was read incorrectly
    assert(imageCount == labelCount);
    //Don't go out of bounds.
    assert(imageCount >= first +num);

    unsigned int it = first;
    while (it < num+ first)
    {
        ProcessLabel(labelStorage, labelData, labelSize, it, first, 0, labelSize);

        ProcessImage(imageStorage, imageData, imageSize, it, first, 0, imageSize);
        it++;
    }


}




void ProcessLabel(float* storage, std::ifstream& data, const unsigned int labelSize, const unsigned int pos, const unsigned first, const unsigned int start, unsigned end)
{
   
    data.seekg(pos + 8);

    unsigned char label;
    char labelBuffer[sizeof(unsigned char)];
    data.read(labelBuffer, sizeof(labelBuffer));
    memcpy(&label, &labelBuffer, sizeof(label));

    unsigned int i = start;
    while (i < end)
    {
        if (i == ((unsigned int)label))
        {
            storage[(pos -first) * labelSize + i ] = 1.0f;
        }
        else {
            storage[(pos-first) * labelSize + i ] = 0.0f;
        }
        i++;
    }
    

}

void ProcessImage(float* storage, std::ifstream& data, const unsigned int imageSize, const unsigned int pos, const unsigned first, const unsigned int start, unsigned end)
{

    data.seekg(16 + pos * imageSize);

    unsigned char pixel;
    char pixelBuffer[sizeof(unsigned char)];

    unsigned int i = start;
    while ( i < end)
    {
        data.read(pixelBuffer, sizeof(pixelBuffer));
        memcpy(&pixel, &pixelBuffer, sizeof(pixel));
        storage[(pos-first) *imageSize+i] = (float)pixel;

        i++;
    }
        
}




void ProcessMNISTDataMT(unsigned threadCount, float* imageStorage, float* labelStorage, const char* labelFilepath, const char* imageFilepath, const unsigned int num, const unsigned int first)
{
    std::ifstream labelData(labelFilepath, std::ios::binary);
    std::ifstream imageData(imageFilepath, std::ios::binary);

    unsigned int labelCount;
    char labelCountBuffer[sizeof(unsigned)];

    labelData.seekg(4);
    labelData.read(labelCountBuffer, sizeof(labelCountBuffer));
    char flipedBuffer[4] = { labelCountBuffer[3],labelCountBuffer[2],labelCountBuffer[1],labelCountBuffer[0] };
    memcpy(&labelCount, &flipedBuffer, sizeof(labelCount));


    unsigned int imageCount;
    unsigned int rowCount;
    unsigned int columnCount;
    char intBuffer[sizeof(unsigned)];

    imageData.seekg(4);
    imageData.read(intBuffer, sizeof(intBuffer));
    char flipedBuffer1[4] = { intBuffer[3],intBuffer[2],intBuffer[1],intBuffer[0] };
    memcpy(&imageCount, &flipedBuffer1, sizeof(imageCount));

    imageData.seekg(8);
    imageData.read(intBuffer, sizeof(intBuffer));
    char flipedBuffer2[4] = { intBuffer[3],intBuffer[2],intBuffer[1],intBuffer[0] };
    memcpy(&rowCount, &flipedBuffer2, sizeof(rowCount));

    imageData.seekg(12);
    imageData.read(intBuffer, sizeof(intBuffer));
    char flipedBuffer3[4] = { intBuffer[3],intBuffer[2],intBuffer[1],intBuffer[0] };
    memcpy(&columnCount, &flipedBuffer3, sizeof(columnCount));

    const unsigned labelSize = 10;
    const unsigned imageSize = columnCount * rowCount;

    //If this triggers you most likely have missmatching files, or at the very least one was read incorrectly
    assert(imageCount == labelCount);
    //Don't go out of bounds.
    assert(imageCount >= first + num);


    std::thread* slaves = new std::thread[threadCount];

    unsigned thread = 0;
    while (thread < threadCount)
    {
        slaves[thread] = std::thread(ProcessMNISTDataMTSlaveFunction, imageStorage, labelStorage, labelFilepath, imageFilepath, num, first, labelSize, imageSize, thread, threadCount);
        thread++;
    }

    thread = 0;
    while (thread < threadCount)
    {
        slaves[thread].join();
        thread++;
    }

}


void ProcessMNISTDataMTSlaveFunction(float* imageStorage, float* labelStorage, const char* labelFilepath, const char* imageFilepath, const unsigned int num, const unsigned int first, const unsigned int labelSize, const unsigned int imageSize, unsigned thread, unsigned threadCount)
{
    std::ifstream labelData(labelFilepath, std::ios::binary);
    std::ifstream imageData(imageFilepath, std::ios::binary);

    const unsigned size = num * labelSize;
    unsigned start = 0;
    unsigned stop = 0;
    if (size < threadCount)
    {
        if (thread < threadCount)
        {
            start = thread;
            stop = thread + 1;
        }
    }
    else
    {
        if (thread < size % threadCount)
        {
            start = ((size / threadCount) + 1) * thread;
            stop = start + ((size / threadCount) + 1);
        }
        else
        {
            start = (size / threadCount) * thread + (size % threadCount);

            stop = start + (size / threadCount);
        }
    }

    const unsigned size2 = num * imageSize;
    unsigned start2 = 0;
    unsigned stop2 = 0;
    if (size2 < threadCount)
    {
        if (thread < threadCount)
        {
            start2 = thread;
            stop2 = thread + 1;
        }
    }
    else
    {
        if (thread < size2 % threadCount)
        {
            start2 = ((size2 / threadCount) + 1) * thread;
            stop2 = start2 + ((size2 / threadCount) + 1);
        }
        else
        {
            start2 = (size2 / threadCount) * thread + (size2 % threadCount);

            stop2 = start2 + (size2 / threadCount);
        }
    }




    labelData.seekg(8 + first + (start / labelSize));
    unsigned char label;
    char labelBuffer[sizeof(unsigned char)];
    if (start % labelSize != 0)
    {
        labelData.read(labelBuffer, sizeof(labelBuffer));
        memcpy(&label, &labelBuffer, sizeof(label));
    }
    unsigned labelIndex = start;
    while (labelIndex < stop)
    {


        if (labelIndex % labelSize == 0)
        {
            labelData.read(labelBuffer, sizeof(labelBuffer));
            memcpy(&label, &labelBuffer, sizeof(label));
        }

        if (labelIndex % labelSize == ((unsigned)label))
        {
            labelStorage[labelIndex] = 1.0f;
        }
        else {
            labelStorage[labelIndex] = 0.0f;
        }
        labelIndex++;
    }


    imageData.seekg(16 + first * imageSize + (start2));
    unsigned char pixel;
    char pixelBuffer[sizeof(unsigned char)];

    unsigned imageIndex = start2;
    while (imageIndex < stop2)
    {
        imageData.read(pixelBuffer, sizeof(pixelBuffer));
        memcpy(&pixel, &pixelBuffer, sizeof(pixel));
        imageStorage[imageIndex] = (float)pixel;

        imageIndex++;
    }
}