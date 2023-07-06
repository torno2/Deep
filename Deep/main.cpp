#include "pch.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkMT.h"
#include "NeuralNetworkMT.cpp"
#include "DataProcessing.h"
#include "NetworkPrototype.h"
#include "NetworkPrototypeMT.h"
#include "LayerFunctions.h"
#include "LayerFunctionsMT.h"


int main() {


#if 2 < 3
	{


#define testStop 1
#define threadN 10


#if 2<3
		using namespace TNNT;
		Timer t;

		//DataFormating start
		t.Start();

		DataSet data;
		{
			constexpr unsigned labelSize = 10;
			constexpr unsigned inputSize = 28 * 28;

			data.TrainingCount = 50000;
			data.TrainingInputs = new float[data.TrainingCount * inputSize];
			data.TraningTargets = new float[data.TrainingCount * labelSize];

			data.ValidationCount = 10000;
			data.ValidationInputs = new float[data.ValidationCount * inputSize];
			data.ValidationTargets = new float[data.ValidationCount * labelSize];

			data.TestCount = 10000;
			data.TestInputs = new float[data.TestCount * inputSize];
			data.TestTargets = new float[data.TestCount * labelSize];



			//Data Formating start

			ProcessMNISTDataMT(10, data.TrainingInputs, data.TraningTargets, "trainLabel.idx1-ubyte", "trainIm.idx3-ubyte", data.TrainingCount);
			ProcessMNISTDataMT(10, data.ValidationInputs, data.ValidationTargets, "trainLabel.idx1-ubyte", "trainIm.idx3-ubyte", data.ValidationCount, 50000);
			ProcessMNISTDataMT(10, data.TestInputs, data.TestTargets, "testLabel.idx1-ubyte", "testIm.idx3-ubyte", data.TestCount);


		}
		pr("Data formating time:" << t.Stop() << "s");

		//Data Formating end



		//Network setup start

		unsigned int layoutCount = 3;
		unsigned int* layout = new unsigned int[layoutCount];
		{
			layout[0] = 784;
			layout[1] = 10;
			layout[2] = 10;
		}		
		unsigned int* layoutMT = new unsigned int[layoutCount];
		{
			layoutMT[0] = 784;
			layoutMT[1] = 10;
			layoutMT[2] = 10;
		}

		LayerFucntionsLayout funcLayout;
		{
			funcLayout.CostFunction = Math::CrossEntropy;
			funcLayout.CostFunctionDerivative = Math::CrossEntropyCostDerivative;
			funcLayout.NeuronFunction = new LayerFucntionsLayout::NeuronFunctionPointer[layoutCount - 1];
			for (int i = 0; i < layoutCount - 1; i++)
			{
				funcLayout.NeuronFunction[i].Function = Math::Sigmoid;
			}

			funcLayout.NeuronFunctionDerivative = new LayerFucntionsLayout::NeuronFunctionPointer[layoutCount - 2];
			for (int i = 0; i < layoutCount - 2; i++)
			{
				funcLayout.NeuronFunctionDerivative[i].Function = Math::SigmoidDerivative;
			}
		}		
		LayerFucntionsLayout funcLayoutMT;
		{
			funcLayoutMT.CostFunction = Math::CrossEntropy;
			funcLayoutMT.CostFunctionDerivative = Math::CrossEntropyCostDerivative;
			funcLayoutMT.NeuronFunction = new LayerFucntionsLayout::NeuronFunctionPointer[layoutCount - 1];
			for (int i = 0; i < layoutCount - 1; i++)
			{
				funcLayoutMT.NeuronFunction[i].Function = Math::Sigmoid;
			}

			funcLayoutMT.NeuronFunctionDerivative = new LayerFucntionsLayout::NeuronFunctionPointer[layoutCount - 2];
			for (int i = 0; i < layoutCount - 2; i++)
			{
				funcLayoutMT.NeuronFunctionDerivative[i].Function = Math::SigmoidDerivative;
			}
		}


		NeuralNetwork nOld(layout, layoutCount, funcLayout, true);

		NeuralNetworkMT nOldMT(layoutMT, layoutCount, funcLayoutMT, threadN);

		//Network setup stop


		//Prototype Network setup start

		unsigned int playoutCount = 3;
		LayerLayout* pLayout = new LayerLayout[playoutCount];
		{
			pLayout[0].Nodes = 28 * 28;
			pLayout[0].Biases = 0;
			pLayout[0].Weights = 0;

			pLayout[1].Nodes = 30;
			pLayout[1].Biases = pLayout[1].Nodes;
			pLayout[1].Weights = pLayout[1].Nodes * pLayout[0].Nodes;

			pLayout[2].Nodes = 10;
			pLayout[2].Biases = pLayout[2].Nodes;
			pLayout[2].Weights = pLayout[2].Nodes * pLayout[1].Nodes;


		}

		FunctionsLayout pFuncLayout;
		{

			pFuncLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[playoutCount - 1];
			{
				pFuncLayout.NeuronFunctions[0].f = Math::Sigmoid;
				pFuncLayout.NeuronFunctions[1].f = Math::Sigmoid;
				;
			}
			pFuncLayout.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[playoutCount - 1];
			{
				pFuncLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
				pFuncLayout.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;

			}

			pFuncLayout.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[playoutCount - 1];
			{
				pFuncLayout.FeedForwardCallBackFunctions[0].f = LayerFunctions::FullyConnectedFeedForward;
				pFuncLayout.FeedForwardCallBackFunctions[1].f = LayerFunctions::FullyConnectedFeedForward;

			}

			pFuncLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[playoutCount - 2];
			{
				pFuncLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctions::FullyConnectedBackpropegateZ;


			}

			pFuncLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[playoutCount - 1];
			{
				pFuncLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctions::FullyConnectedBackpropegateBW;
				pFuncLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctions::FullyConnectedBackpropegateBW;
			}



			pFuncLayout.CostFunction.f = CostFunctions::CrossEntropy;
			pFuncLayout.CostFunctionDerivative.f = CostFunctions::CrossEntropyDerivative;


			pFuncLayout.RegularizationFunction.f = TrainingFunctions::L2Regularization;
			pFuncLayout.TrainingFunction.f = TrainingFunctions::GradientDecent;
		}

		NetworkPrototype nP(pLayout, pFuncLayout, playoutCount, true);

		//Prototype Network setup stop

		// MT prototype setup start

		FunctionsLayoutMT mtFuncLayout;
		{

			mtFuncLayout.NeuronFunctions = new FunctionsLayoutMT::NeuronFunction[playoutCount - 1];
			{
				mtFuncLayout.NeuronFunctions[0].f = Math::Sigmoid;
				mtFuncLayout.NeuronFunctions[1].f = Math::Sigmoid;
				;
			}
			mtFuncLayout.NeuronFunctionsDerivatives = new FunctionsLayoutMT::NeuronFunction[playoutCount - 1];
			{
				mtFuncLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
				mtFuncLayout.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;

			}

			mtFuncLayout.FeedForwardCallBackFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT [playoutCount - 1] ;
			{
				mtFuncLayout.FeedForwardCallBackFunctions[0].f = LayerFunctionsMT::FullyConnectedFeedForward;
				mtFuncLayout.FeedForwardCallBackFunctions[1].f = LayerFunctionsMT::FullyConnectedFeedForward;

			}

			mtFuncLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayoutMT::NetworkRelayFunctionMT[playoutCount - 2];
			{
				mtFuncLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctionsMT::FullyConnectedBackpropegateZ;


			}

			mtFuncLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayoutMT::NetworkRelayFunctionMT[playoutCount - 1];
			{
				mtFuncLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctionsMT::FullyConnectedBackpropegateBW;
				mtFuncLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctionsMT::FullyConnectedBackpropegateBW;
			}



			mtFuncLayout.CostFunction.f = CostFunctionsMT::CrossEntropy;
			mtFuncLayout.CostFunctionDerivative.f = CostFunctionsMT::CrossEntropyDerivative;


			mtFuncLayout.RegularizationFunction.f = TrainingFunctionsMT::L2Regularization;
			mtFuncLayout.TrainingFunction.f = TrainingFunctionsMT::GradientDecent;
		}

		NetworkPrototypeMT nMT(pLayout,mtFuncLayout,playoutCount, threadN);
		// MT prototype setup stop


		HyperParameters params;
		{
			params.Epochs = 1;

			params.BatchCount = 10;
			params.LearningRate = 0.01f;
			params.RegularizationConstant = 0.01f;
		}
#endif

#if 2>3
		PArr<float>(&nP.m_ABuffer[nP.m_ABufferCount - nP.m_LayerLayout[nP.m_LayerLayoutCount - 1].Nodes], nP.m_LayerLayout[nP.m_LayerLayoutCount - 1].Nodes);

		pr("Separator");
		PArr<float>(&nMT.m_ABuffer[nMT.m_ABufferCount - nMT.m_LayerLayout[nMT.m_LayerLayoutCount - 1].Nodes], nMT.m_LayerLayout[nMT.m_LayerLayoutCount - 1].Nodes);

		pr("Separator");

		nP.SetData(&data);
		nMT.SetData(&data);

		nP.SetInput(data.TestInputs);
		

		for (int i = 0; i < nMT.m_SlaveThreadCount; i++)
		{
			nMT.m_SlaveThreads[i]= std::thread(&NetworkPrototypeMT::SetInput,&nMT, data.TestInputs, i);
			
		}
		for (int i = 0; i < nMT.m_SlaveThreadCount; i++)
		{
			nMT.m_SlaveThreads[i].join();

		}

		nP.FeedForward();

		for (int i = 0; i < nMT.m_SlaveThreadCount; i++)
		{
			nMT.m_SlaveThreads[i] = std::thread(&NetworkPrototypeMT::FeedForward, &nMT, i);
		}
		for (int i = 0; i < nMT.m_SlaveThreadCount; i++)
		{
			nMT.m_SlaveThreads[i].join();

		}


		PArr<float>(&nP.m_ABuffer[nP.m_ABufferCount - nP.m_LayerLayout[nP.m_LayerLayoutCount - 1].Nodes], nP.m_LayerLayout[nP.m_LayerLayoutCount - 1].Nodes);

		pr("Separator");
		PArr<float>(&nMT.m_ABuffer[nMT.m_ABufferCount - nMT.m_LayerLayout[nMT.m_LayerLayoutCount - 1].Nodes], nMT.m_LayerLayout[nMT.m_LayerLayoutCount - 1].Nodes);
#endif


#if 2<3
#if testStop > 2
		pr("Old: ");
		{
			//Training start
			t.Start();


			nOld.Train(data, params);
			pr("Train time: " << t.Stop() << "s");
	
			//Traning End


			//Check Start

			t.Start();
			pr("Cost: " << nOld.CheckCost(data));
			pr("Guessrate: " << nOld.CheckSuccessRate(data));

			pr("Check time: " << t.Stop() << "s");

			//Check Stop
		}
#endif
#if testStop > 2
		pr("OldMT: ");
		{
			//Training start
			t.Start();

			nOldMT.Train(data, params);

			pr("Train time: " << t.Stop() << "s");

			//Traning End


			//Check Start

			t.Start();
			pr("Cost: " << nOldMT.CheckCost(data));
			pr("Guessrate: " << nOldMT.CheckSuccessRate(data));

			pr("Check time: " << t.Stop() << "s");

			//Check Stop
		}
#endif


		pr("New");
		pr("Prototype: ");
		{

			//Training start
			t.Start();
			nP.SetData(&data);
	#if testStop >0
			nP.Train(&data, params);
	
			pr("Train time: " << t.Stop() << "s");
	#endif
			//Traning End

			
			//Check Start
			t.Start();

			pr("Cost: " << nP.CheckCost());
			pr("Guessrate: " << nP.CheckSuccessRate());

			pr("Check time: " << t.Stop() << "s");
			//Check Stop
		}

		pr("MT");
		pr("Prototype: ");
		{

			//Training start
			t.Start();
			nMT.SetData(&data);
	#if testStop >0
			nMT.Train(&data, params);
	
		
			pr("Train time: " << t.Stop() << "s");
	#endif
			//Traning End
			

			t.Start();

			pr("Cost: " << nMT.CheckCost());
			pr("Guessrate: " << nMT.CheckSuccessRate());


			pr("Check time: " << t.Stop() << "s");
			//Check Stop
		}

#endif
		
#if 2>3
		unsigned i = 0;
		while (i < data.TrainingCount)
		{
			if (nP.m_Indices[i] != nMT.m_Indices[i])
			{
				pr(i);
				pr("P: " << nP.m_Indices[i]);
				pr("MT: " << nMT.m_Indices[i]);
				
			}
			i++;

		}
		pr("Hey");
#endif

#if 2>3
		std::ofstream efile;
		efile.open("Experiment.txt");


	
		unsigned int testlayoutCount = 3;
		FunctionsLayout testFuncLayout;
		{

			testFuncLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[testlayoutCount - 1];
			{
				testFuncLayout.NeuronFunctions[0].f = Math::Sigmoid;
				testFuncLayout.NeuronFunctions[1].f = Math::Sigmoid;
				;
			}
			testFuncLayout.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[testlayoutCount - 1];
			{
				testFuncLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
				testFuncLayout.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;

			}

			testFuncLayout.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[testlayoutCount - 1];
			{
				testFuncLayout.FeedForwardCallBackFunctions[0].f = LayerFunctions::FullyConnectedFeedForward;
				testFuncLayout.FeedForwardCallBackFunctions[1].f = LayerFunctions::FullyConnectedFeedForward;

			}

			testFuncLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[testlayoutCount - 2];
			{
				testFuncLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctions::FullyConnectedBackpropegateZ;


			}

			testFuncLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[testlayoutCount - 1];
			{
				testFuncLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctions::FullyConnectedBackpropegateBW;
				testFuncLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctions::FullyConnectedBackpropegateBW;
			}



			testFuncLayout.CostFunction.f = CostFunctions::CrossEntropy;
			testFuncLayout.CostFunctionDerivative.f = CostFunctions::CrossEntropyDerivative;


			testFuncLayout.RegularizationFunction.f = TrainingFunctions::L2Regularization;
			testFuncLayout.TrainingFunction.f = TrainingFunctions::GradientDecent;
		}


		t.Start();
		unsigned experiments = 0;

		unsigned i = 1;
		while (i < 3)
		{
			
			LayerLayout* testLayout = new LayerLayout[testlayoutCount];
			{
				testLayout[0].Nodes = 28 * 28;
				testLayout[0].Biases = 0;
				testLayout[0].Weights = 0;

				testLayout[1].Nodes = 5*i;
				testLayout[1].Biases = testLayout[1].Nodes;
				testLayout[1].Weights = testLayout[1].Nodes * testLayout[0].Nodes;

				testLayout[2].Nodes = 10;
				testLayout[2].Biases = testLayout[2].Nodes;
				testLayout[2].Weights = testLayout[2].Nodes * testLayout[1].Nodes;


			}

			experiments++;
			efile << "Experiment: " << experiments << '\n' << '\n';
			

			
			
			efile << "Nodes in hidden layer: " << testLayout[1].Nodes << '\n' << '\n';




			efile << "Testing network using different learning rates: " << '\n' << '\n';
			unsigned j = 1;
			while (j < 4)
			{
				NetworkPrototype testN(testLayout, testFuncLayout, testlayoutCount);

				HyperParameters testparams;
				{
					testparams.Epochs = 1;

					testparams.BatchCount = 10;
					testparams.LearningRate = 0.002f*j;
					testparams.RegularizationConstant = 0.01f;
				}

				unsigned i = 0;
				
				

				testN.Train(&data, testparams);

				efile << j << " : " << testparams.LearningRate << '\n' << "Results: | Cost: " << testN.CheckCost() << " |  Success Rate: " << testN.CheckSuccessRate() << '\n' << '\n';

				j++;


				efile << "Training duration: " << testN.m_LastTime[0] << '\n' << '\n';
			}
			
			

			delete[] testLayout;
			

			i++;
			
			
		}

		efile.close();

		
		pr("Experiments completed. Duration: " << t.Stop() << "s");

	#endif



		
	}

#endif





	std::cin.get();
}



