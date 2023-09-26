#pragma once

namespace TMMT
{

	class LayerLayoutPrototype
	{
		unsigned NodesCount;
		unsigned BiasesCount;
		unsigned WeightsCount;


		float LearningRate = 1.0f;
		float RegularizationConstant = 0.0f;



	};




}


