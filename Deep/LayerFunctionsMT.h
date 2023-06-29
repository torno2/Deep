#pragma once

#include "NeuralNetworkCustomVariables.h"
#include "NetworkPrototypeMT.h"

namespace TNNT
{

	namespace LayerFunctionsMT
	{

		void FullyConnectedFeedForward(NetworkPrototypeMT* n, unsigned thread);
		void FullyConnectedBackpropegateZ(NetworkPrototypeMT* n, unsigned thread);
		void FullyConnectedBackpropegateBW(NetworkPrototypeMT* n, unsigned thread);
	}



	namespace CostFunctionsMT
	{
		void CrossEntropy(NetworkPrototypeMT* n, unsigned thread);
		void CrossEntropyDerivative(NetworkPrototypeMT* n, unsigned thread);

	}


	namespace TrainingFunctionsMT
	{
		void L2Regularization(NetworkPrototypeMT* n, unsigned thread);

		void GradientDecent(NetworkPrototypeMT* n, unsigned thread);
	}


}