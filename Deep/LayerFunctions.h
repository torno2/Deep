#pragma once
#include "NeuralNetworkCustomVariables.h"
#include "NetworkPrototype.h"

namespace TNNT
{
	namespace LayerFunctions
	{

		void FullyConnectedFeedForward(NetworkPrototype* n);
		void FullyConnectedBackpropegateZ(NetworkPrototype* n);
		void FullyConnectedBackpropegateBW(NetworkPrototype* n);

		void ConvolutionLayerFeedForward(NetworkPrototype* n);
		void ConvolutionLayerBackpropegateZ(NetworkPrototype* n);
		void ConvolutionLayerBackpropegateBW(NetworkPrototype* n);

		void PoolingLayerFeedForward(NetworkPrototype* n);
		void PoolingLayerBackpropegateZ(NetworkPrototype* n);
		void PoolingLayerBackpropegateBW(NetworkPrototype* n);
	}

	namespace CostFunctions
	{

		void CrossEntropy(NetworkPrototype* n);
		void CrossEntropyDerivative(NetworkPrototype* n);

	}

	namespace TrainingFunctions
	{
		void L2Regularization(NetworkPrototype* n );

		void GradientDecent(NetworkPrototype* n );

	}
}