#include "PGNet.h"

#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "Device/Wrappers/KernelTimings.h"
#include "NeuralNetworks/tools.h"
#include "NeuralNetworks/Config/EncodingConfig.h"
#include "NeuralNetworks/Distributions/Mixture.h"

namespace vtx::network
{
#define PRINT_TENSOR_SLICE(tensor) std::cout << #tensor << " :\n" << tensor.slice(0, 0, 10) << std::endl;

	PGNet::PGNet(config::NetworkSettings* _settings) :
		device(torch::kCUDA, 0)
	{
		settings = _settings;
		PGNet::init();
		ntscLuminance = torch::tensor({ 0.299f, 0.587f, 0.114f }, torch::TensorOptions().dtype(torch::kFloat).device(device));
	}
	void PGNet::init()
	{
		batches.clear();
		constexpr int auxAdditionInputSize = 3; // wi
		constexpr int mainOutputSize = 9; // outRadiance, inRadiance, throughput
		distributionParametersCount = distribution::Mixture::getDistributionParametersCount(settings->distributionType, true);
		mixtureParametersCount = settings->mixtureSize * distributionParametersCount;
		outputDim = mixtureParametersCount + settings->mixtureSize + 1;
		int auxFeatureSize = settings->totAuxInputSize - auxAdditionInputSize;

		if (settings->useAuxiliaryNetwork)
		{
			outputDim += auxFeatureSize;
		}

		auto encodingSettings = torchTcnn::getEncodingSettings(settings->inputSettings);

		int configN = 3;
		int mainInputSize = 9; // position, normal, wo
		if (settings->learnInputRadiance)
		{
			configN = 2;
			mainInputSize = 6;
			encodingSettings["nested"].erase(configN);
		}
		if(settings->useMaterialId)
		{
			encodingSettings["nested"][configN] = torchTcnn::getEncodingSettings(settings->materialIdEncodingConfig);
			encodingSettings["nested"][configN]["n_dims_to_encode"] = 1;
			mainInputSize++;
			configN++;
		}

		if (settings->useInstanceId)
		{
			encodingSettings["nested"][configN] = torchTcnn::getEncodingSettings(settings->instanceIdEncodingConfig);
			encodingSettings["nested"][configN]["n_dims_to_encode"] = 1;
			mainInputSize++;
			configN++;
		}

		if (settings->useTriangleId)
		{
			encodingSettings["nested"][configN] = torchTcnn::getEncodingSettings(settings->triangleIdEncodingConfig);
			encodingSettings["nested"][configN]["n_dims_to_encode"] = 1;
			mainInputSize++;
			configN++;
		}

		mlpBase = torchTcnn::TcnnModule(mainInputSize, outputDim, encodingSettings, torchTcnn::getNetworkSettings(settings->mainNetSettings), 1337);
		if(settings->emaUpdate)
		{
			mlpInference = torchTcnn::TcnnModule(mainInputSize, outputDim, encodingSettings, torchTcnn::getNetworkSettings(settings->mainNetSettings), 1337);
		}
		const double eps = std::pow(10, -settings->adamEps);
		if (settings->useAuxiliaryNetwork)
		{
			auto netSettings = torchTcnn::getNetworkSettings(settings->auxiliaryNetSettings);
			auto auxInputSettings = torchTcnn::getEncodingSettings(auxFeatureSize, settings->auxiliaryInputSettings);
			auxiliaryMlp = torchTcnn::TcnnModule(settings->totAuxInputSize, mainOutputSize, auxInputSettings, netSettings, 1440);
			optimizer = std::make_shared<torch::optim::Adam>(std::vector<torch::optim::OptimizerParamGroup>{mlpBase->parameters(), auxiliaryMlp->parameters()}, torch::optim::AdamOptions(settings->learningRate).amsgrad(true).eps(eps));
		}
		else
		{
			optimizer = std::make_shared<torch::optim::Adam>(mlpBase->parameters(), torch::optim::AdamOptions(settings->learningRate).amsgrad(true).eps(eps));
		}

		scheduler = std::make_shared<torch::optim::StepLR>(*optimizer, settings->schedulerStep, settings->schedulerGamma);
		trainingStep                   = 0;
	}

	void PGNet::reset()
	{
		graphs.reset();
		init();
	}

	void PGNet::train()
	{
		//VTX_INFO("Training Step {}", trainingStep);
		CLEAR_TENSOR_DEBUGGER();
		ANOMALY_SWITCH;
		if (trainingStep >= settings->maxTrainingSteps)
		{
			settings->doTraining = false;
			return;
		}
		trainingStep++;
		optimizer->zero_grad();

		if(batches.size()==0)
		{
			generateTrainingInputTensors();
		}
		if(batches.size()==0)
		{
			VTX_WARN("No training data available");
			return;
		}
		InputTensors inputTensors = batches.back();
		batches.pop_back();
		if(inputTensors.position.size(0)==0)
		{
			VTX_WARN("No training data available on Batch");
			return;
		}
		const torch::Tensor& position = inputTensors.position;
		const torch::Tensor& wo = inputTensors.wo;
		const torch::Tensor& normal = inputTensors.normal;
		const torch::Tensor& incomingDirection = inputTensors.wi;
		torch::Tensor& bsdfProb = inputTensors.bsdfProb;
		const torch::Tensor& outRadianceTarget = inputTensors.outRadiance;

		torch::Tensor targetRadiance;
		torch::Tensor input = torch::cat({ position, normal }, -1);
		if(settings->learnInputRadiance)
		{
			targetRadiance = inputTensors.inRadiance;
		}
		else
		{
			input = torch::cat({ input, wo }, -1);
			targetRadiance = outRadianceTarget;
		}
		if(settings->useTriangleId)
		{
			input = torch::cat({ input, inputTensors.triangleId }, -1);
		}
		if(settings->useMaterialId)
		{
			input = torch::cat({ input, inputTensors.materialId }, -1);
		}
		if(settings->useInstanceId)
		{
			input = torch::cat({ input, inputTensors.instanceId }, -1);
		}
		const torch::Tensor rawOutput = mlpBase->forward(input);
		TRACE_TENSOR(rawOutput);
		const auto& [mixtureParameters, mixtureWeights, c] = guidingForward(rawOutput);
		const torch::Tensor neuralProb = distribution::Mixture::prob(incomingDirection, mixtureParameters, mixtureWeights, settings->distributionType);

		torch::Tensor pgLoss;
		pgLoss = guidingLoss(neuralProb, bsdfProb, targetRadiance, c, inputTensors.wiProb);
		/*if (!settings->learnInputRadiance) {
			pgLoss = guidingLoss(neuralProb, bsdfProb, targetRadiance, c, inputTensors.wiProb);
		}
		else
		{
			pgLoss = guidingLoss(neuralProb, bsdfProb, targetRadiance, c, inputTensors.wiProb);
			pgLoss = gudingLossIncomingRadiance(neuralProb, bsdfProb, targetRadiance, c, inputTensors.wiProb);
		}*/


		if (settings->useEntropyLoss)
		{
			//const torch::Tensor entropyLoss = settings->entropyWeight * guidingEntropyLoss(neuralProb);
			// mixture weights entropy loss
			const torch::Tensor mixturesEntropy = (-mixtureWeights * torch::log(mixtureWeights + bEps)).mean(1);
			torch::Tensor entropyLoss = (-mixturesEntropy).mean();
			entropyLoss = settings->entropyWeight * entropyLoss;
			pgLoss += entropyLoss;
			if (settings->plotGraphs)
			{
				graphs.addData(LOSS_PLOT, "Entropy Loss", entropyLoss.mean().item<float>());
			}
		}

		if (settings->useAuxiliaryNetwork)
		{
			const torch::Tensor& inRadianceTarget = inputTensors.inRadiance;
			const torch::Tensor& throughputTarget = inputTensors.throughput;
			const torch::Tensor featureVector = rawOutput.narrow(1, mixtureParametersCount + settings->mixtureSize + 1, settings->totAuxInputSize-3);
			const torch::Tensor auxInput = torch::cat({ featureVector, incomingDirection }, -1);
			auto [incomingRadiance, throughput, outgoingRadiance] = splitAuxiliaryOutput(auxInput);
			const torch::Tensor auxLoss = auxiliaryLoss(incomingRadiance, inRadianceTarget, throughput, throughputTarget, outgoingRadiance, outRadianceTarget);

			pgLoss += (1.0f - lossBlendFactor())*auxLoss;

		}

		// L2 Regularization term
		if(settings->l2WeightExp>0)
		{
			torch::Tensor& params = mlpBase->params;
			double beta = std::pow(10.0, -settings->l2WeightExp);
			const torch::Tensor l2_reg = (beta * params.pow(2)).sum();
			pgLoss += l2_reg;
		}

		pgLoss.backward();

		//torch::nn::utils::clip_grad_norm_(mlpBase->parameters(), 1.0);

		if (TensorDebugger::analyzeGradients()) {
			VTX_ERROR("Gradients are NAN or INF\n");
			return; // NAN or INF gradients
		}
		optimizer->step();
		scheduler->step();

		if(settings->emaUpdate)
		{
			emaUpdate();
		}

		if (settings->plotGraphs)
		{
			graphs.addData(SAMPLING_FRACTION_PLOT, "Sampling Fraction Train", c.mean().item<float>());
			graphs.addData(SAMPLING_FRACTION_PLOT, "Loss Blend", lossBlendFactor());
			graphs.addData("BatchSize","Iteration","BatchSize", "BatchSize", inputTensors.position.size(0));
			graphs.addData("Learning Rate", "Iteration", "Learning Rate", "Learing Rate", optimizer->param_groups().at(0).options().get_lr());

			distribution::Mixture::setGraphData(settings->distributionType, mixtureParameters, mixtureWeights, graphs, true);
		}

	}

	void PGNet::emaUpdate()
	{
		if (trainingStep == 0) {
			// Initialize EMA weights with the same data and device as model weights
			mlpInference->params = mlpBase->params;
		}
		else {
			// Update EMA weights
			mlpInference->params = settings->emaDecay * mlpBase->params + (1 - settings->emaDecay) * mlpInference->params;
		}
	}

	void PGNet::inference(const int& depth)
	{
		CLEAR_TENSOR_DEBUGGER();
		torch::NoGradGuard noGrad;
		int inferenceSize;
		const auto& inputTensors = generateInferenceInputTensors(&inferenceSize);
		if (inferenceSize == 0)
		{
			return;
		}
		const torch::Tensor& position = inputTensors.position;
		const torch::Tensor& wo = inputTensors.wo;
		const torch::Tensor& normal = inputTensors.normal;
		torch::Tensor input = torch::cat({ position, normal }, -1);
		if (!settings->learnInputRadiance){
			input = torch::cat({ input, wo }, -1);
		}

		if (settings->useTriangleId)
		{
			input = torch::cat({ input, inputTensors.triangleId }, -1);
		}
		if (settings->useMaterialId)
		{
			input = torch::cat({ input, inputTensors.materialId }, -1);
		}
		if (settings->useInstanceId)
		{
			input = torch::cat({ input, inputTensors.instanceId }, -1);
		}
		torch::Tensor rawOutput;
		if(!settings->emaUpdate)
		{
			rawOutput = mlpBase->forward(input);
		}
		else
		{
			rawOutput = mlpInference->forward(input);
		}
		TRACE_TENSOR(rawOutput);
		auto [mixtureParameters, mixtureWeights, c] = guidingForward(rawOutput);
		if (settings->samplingFractionBlend)
		{

			const float fractionBlend = std::min(1.0f, (float)trainingStep / (settings->fractionBlendTrainPercentage*(float)settings->maxTrainingSteps));
			c = fractionBlend * c;
		}

		if (settings->clampSamplingFraction)
		{
			c = torch::clamp(c, 0.0f, settings->sfClampValue);
		}

		if(settings->constantSamplingFraction)
		{
			c = torch::ones_like(c) * settings->constantSamplingFractionValue;
		}

		TRACE_TENSOR(c);
		TRACE_TENSOR(mixtureParameters);
		//print first 10 batches of mixturePrameters
		TRACE_TENSOR(mixtureWeights);
		if (settings->plotGraphs)
		{
			float samplingFraction = c.mean().item<float>();
			graphs.addData(SAMPLING_FRACTION_PLOT, "Sampling Fraction Inference", samplingFraction, depth);
			distribution::Mixture::setGraphData(settings->distributionType, mixtureParameters, mixtureWeights, graphs, false, depth);
		}

		copyOutputToCudaBuffer(mixtureParameters, c, mixtureWeights, inferenceSize);

	}

	void PGNet::copyOutputToCudaBuffer(const torch::Tensor& mixtureParameters, const torch::Tensor& c, const torch::Tensor& mixtureWeights, int inferenceSize)
	{
		const device::InferenceDataBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.inferenceDataBuffers;
		const auto distributionParametersPtr = buffers.distributionParametersBuffer.castedPointer<float>();
		const auto samplingFractionPtr = buffers.samplingFractionArrayBuffer.castedPointer<float>();
		const auto mixtureWeightPtr = buffers.mixtureWeightsBuffer.castedPointer<float>();

		int inferenceDistributionParameterCount = distribution::Mixture::getDistributionParametersCount(settings->distributionType);
		const torch::Tensor distributionParameterCuda = torch::from_blob(distributionParametersPtr, { inferenceSize, settings->mixtureSize, inferenceDistributionParameterCount }, at::device(device).dtype(torch::kFloat));
		const torch::Tensor samplingFractionCuda = torch::from_blob(samplingFractionPtr, { inferenceSize, 1 }, at::device(device).dtype(torch::kFloat));
		const torch::Tensor mixtureWeightCuda = torch::from_blob(mixtureWeightPtr, { inferenceSize, settings->mixtureSize }, at::device(device).dtype(torch::kFloat));

		distributionParameterCuda.copy_(mixtureParameters);
		samplingFractionCuda.copy_(c);
		mixtureWeightCuda.copy_(mixtureWeights);

		TRACE_TENSOR(distributionParameterCuda);
		TRACE_TENSOR(samplingFractionCuda);
		TRACE_TENSOR(mixtureWeightCuda);
	}

	GraphsData& PGNet::getGraphs()
	{
		return graphs;
	}

	bool PGNet::doNormalizePosition()
	{
		if (
			settings->inputSettings.normalizePosition// ||
			//settings->inputSettings.position.otype == config::EncodingType::Frequency ||
			//settings->inputSettings.position.otype == config::EncodingType::Grid ||
			//settings->inputSettings.position.otype == config::EncodingType::OneBlob ||
			//settings->inputSettings.position.otype == config::EncodingType::SphericalHarmonics ||
			//settings->inputSettings.position.otype == config::EncodingType::TriangleWave
			)
		{
			return true;
		}
		return false;
	}

	InputTensors& shuffle(InputTensors& inputTensors)
	{
		auto indices = torch::randperm(inputTensors.position.size(0), torch::TensorOptions().device(inputTensors.position.device()));
		if (inputTensors.position.defined()) inputTensors.position = inputTensors.position.index_select(0, indices);
		if (inputTensors.wo.defined()) inputTensors.wo = inputTensors.wo.index_select(0, indices);
		if (inputTensors.normal.defined()) inputTensors.normal = inputTensors.normal.index_select(0, indices);
		if (inputTensors.wi.defined()) inputTensors.wi = inputTensors.wi.index_select(0, indices);
		if (inputTensors.bsdfProb.defined()) inputTensors.bsdfProb = inputTensors.bsdfProb.index_select(0, indices);
		if (inputTensors.outRadiance.defined()) inputTensors.outRadiance = inputTensors.outRadiance.index_select(0, indices);
		if (inputTensors.throughput.defined()) inputTensors.throughput = inputTensors.throughput.index_select(0, indices);
		if (inputTensors.inRadiance.defined()) inputTensors.inRadiance = inputTensors.inRadiance.index_select(0, indices);
		if (inputTensors.instanceId.defined()) inputTensors.instanceId = inputTensors.instanceId.index_select(0, indices);
		if (inputTensors.triangleId.defined()) inputTensors.triangleId = inputTensors.triangleId.index_select(0, indices);
		if (inputTensors.materialId.defined()) inputTensors.materialId = inputTensors.materialId.index_select(0, indices);
		if (inputTensors.wiProb.defined()) inputTensors.wiProb = inputTensors.wiProb.index_select(0, indices);
		return inputTensors;
	}

	void PGNet::sliceToBatches(InputTensors& tensors)
	{
		int batchSize = settings->batchSize;
		int numBatches = tensors.position.size(0) / batchSize;
		int remainder = tensors.position.size(0) % batchSize;

		for (int i = 0; i <= numBatches; ++i) {
			int startIdx = i * batchSize;
			int endIdx = startIdx + batchSize;
			if (i == numBatches)
			{
				if (remainder == 0) break;
				if (remainder < batchSize)
				{
					if (settings->trainingBatchGenerationSettings.skipIncomplete || i > 0)
					{
						break;
					}
				}
				endIdx = startIdx + remainder;
			}

			InputTensors batch;
			if (tensors.position.defined()) batch.position = tensors.position.slice(0, startIdx, endIdx);
			if (tensors.wo.defined()) batch.wo = tensors.wo.slice(0, startIdx, endIdx);
			if (tensors.normal.defined()) batch.normal = tensors.normal.slice(0, startIdx, endIdx);
			if (tensors.wi.defined()) batch.wi = tensors.wi.slice(0, startIdx, endIdx);
			if (tensors.bsdfProb.defined()) batch.bsdfProb = tensors.bsdfProb.slice(0, startIdx, endIdx);
			if (tensors.outRadiance.defined()) batch.outRadiance = tensors.outRadiance.slice(0, startIdx, endIdx);
			if (tensors.throughput.defined()) batch.throughput = tensors.throughput.slice(0, startIdx, endIdx);
			if (tensors.inRadiance.defined()) batch.inRadiance = tensors.inRadiance.slice(0, startIdx, endIdx);
			if (tensors.instanceId.defined()) batch.instanceId = tensors.instanceId.slice(0, startIdx, endIdx);
			if (tensors.triangleId.defined()) batch.triangleId = tensors.triangleId.slice(0, startIdx, endIdx);
			if (tensors.materialId.defined()) batch.materialId = tensors.materialId.slice(0, startIdx, endIdx);
			if (tensors.wiProb.defined()) batch.wiProb = tensors.wiProb.slice(0, startIdx, endIdx);
			batches.push_back(batch);
		}
	}

	void debugInputTensors(InputTensors& tensors, std::string msg)
	{
		VTX_INFO("{}", msg);
		PRINT_TENSOR_SLICE(tensors.position);
		PRINT_TENSOR_SLICE(tensors.wo);
		PRINT_TENSOR_SLICE(tensors.normal);
		PRINT_TENSOR_SLICE(tensors.outRadiance);
	}

	void PGNet::generateTrainingInputTensors()
	{
		prepareDataset();
		const device::TrainingDataBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.trainingDataBuffers;

		InputDataPointers inputPointers;
		inputPointers.wi = buffers.wiBuffer.castedPointer<float>();
		inputPointers.bsdfProb = buffers.bsdfProbBuffer.castedPointer<float>();
		inputPointers.position = buffers.positionBuffer.castedPointer<float>();

		if (settings->useTriangleId)
		{
			inputPointers.triangleId = buffers.triangleIdBuffer.castedPointer<float>();
		}
		if (settings->useMaterialId)
		{
			inputPointers.materialId = buffers.matIdBuffer.castedPointer<float>();
		}
		if (settings->useInstanceId)
		{
			inputPointers.instanceId = buffers.instanceIdBuffer.castedPointer<float>();
		}
		inputPointers.wo = buffers.woBuffer.castedPointer<float>();
		inputPointers.normal = buffers.normalBuffer.castedPointer<float>();
		inputPointers.outRadiance = buffers.LoBuffer.castedPointer<float>();
		inputPointers.inRadiance = buffers.LiBuffer.castedPointer<float>();
		if(settings->useAuxiliaryNetwork)
		{
			inputPointers.throughput = buffers.bsdfBuffer.castedPointer<float>();
		}
		inputPointers.wiProb = buffers.wiProbBuffer.castedPointer<float>();



		CUDABuffer trainingDataSizeBuffer = buffers.sizeBuffer;
		int trainingDataSize = 0;
		trainingDataSizeBuffer.download(&trainingDataSize);
		if (trainingDataSize == 0)
		{
			VTX_WARN("No training data available");
			return;
		}

		InputTensors output = generateInputTensors(trainingDataSize, inputPointers);
		TRACE_TENSOR(output.position);
		TRACE_TENSOR(output.wo);
		TRACE_TENSOR(output.normal);
		TRACE_TENSOR(output.wi);
		TRACE_TENSOR(output.bsdfProb);
		TRACE_TENSOR(output.outRadiance);


		shuffle(output);
		//auto bsdfwiProb = torch::stack({ output.bsdfProb, output.wiProb }, 1);
		//PRINT_TENSOR_SLICE(bsdfwiProb);
		sliceToBatches(output);
	}

	InputTensors PGNet::generateInferenceInputTensors(int* inferenceSize)
	{
		const device::InferenceDataBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.inferenceDataBuffers;
		CUDABuffer inferenceSizeBuffers = buffers.sizeBuffer;
		inferenceSizeBuffers.download(inferenceSize);
		if (*inferenceSize == 0)
		{
			return {};
		}

		InputDataPointers inputPointers;
		inputPointers.position = buffers.positionBuffer.castedPointer<float>();
		inputPointers.wo = buffers.woBuffer.castedPointer<float>();
		inputPointers.normal = buffers.normalBuffer.castedPointer<float>();
		if (settings->useTriangleId)
		{
			inputPointers.triangleId = buffers.triangleIdBuffer.castedPointer<float>();
		}
		if (settings->useMaterialId)
		{
			inputPointers.materialId = buffers.matIdBuffer.castedPointer<float>();
		}
		if (settings->useInstanceId)
		{
			inputPointers.instanceId = buffers.instanceIdBuffer.castedPointer<float>();
		}
		InputTensors output = generateInputTensors(*inferenceSize, inputPointers);
		TRACE_TENSOR(output.position);
		TRACE_TENSOR(output.wo);
		TRACE_TENSOR(output.normal);

		return output;
	}

	torch::Tensor PGNet::pointerToTensor(float* pointer, int batchSize, int dim, float minClamp = 0.0f, float maxClamp = 0.0f)
	{
		if (pointer == nullptr) return {};
		torch::Tensor tensor = torch::from_blob(pointer, { batchSize, dim }, torch::TensorOptions().device(device).dtype(torch::kFloat))toPrecision;
		if (minClamp != 0.0f || maxClamp != 0.0f)
		{
			tensor = torch::clamp(tensor, minClamp, maxClamp);
		}

		return tensor;
	}

	torch::Tensor PGNet::pointerToTensor(int* pointer, int batchSize, int dim)
	{
		if (pointer == nullptr) return {};
		torch::Tensor tensor = torch::from_blob(pointer, { batchSize, dim }, torch::TensorOptions().device(device).dtype(torch::kInt))toPrecision;

		return tensor;
	}

	InputTensors PGNet::generateInputTensors(const int batchDim, const InputDataPointers& inputPointers)
	{
		InputTensors tensors;

		tensors.position = pointerToTensor(inputPointers.position, batchDim, 3);
		if (doNormalizePosition())
		{

			auto          aabb = onDeviceData->launchParamsData.getHostImage().aabb;
			// inflate aabb
			constexpr float percentage = 0.005f;
			const float dx = aabb.maxX - aabb.minX;
			const float dy = aabb.maxY - aabb.minY;
			const float dz = aabb.maxZ - aabb.minZ;
			aabb.minX -= dx * percentage;
			aabb.minY -= dy * percentage;
			aabb.minZ -= dz * percentage;
			aabb.maxX += dx * percentage;
			aabb.maxY += dy * percentage;
			aabb.maxZ += dz * percentage;
			const torch::Tensor minExtents = torch::tensor({ aabb.minX, aabb.minY, aabb.minZ }).to(device).to(torch::kFloat);
			const torch::Tensor maxExtents = torch::tensor({ aabb.maxX, aabb.maxY, aabb.maxZ }).to(device).to(torch::kFloat);
			const torch::Tensor deltaExtents = maxExtents - minExtents;
			if (!deltaExtents.eq(0).any().item<bool>())
			{
				//PRINT_TENSOR_ALWAYS("minExtents",minExtents);
				//PRINT_TENSOR_ALWAYS("maxExtents", deltaExtents);
				//PRINT_TENSOR_SLICE(tensors.position);
				tensors.position = (tensors.position - minExtents) / deltaExtents;
				//PRINT_TENSOR_SLICE(tensors.position);
			}
		}
		tensors.wo = pointerToTensor(inputPointers.wo, batchDim, 3);
		tensors.normal = pointerToTensor(inputPointers.normal, batchDim, 3);
		if (settings->inputSettings.wo.otype == config::EncodingType::SphericalHarmonics)
		{
			tensors.wo = (tensors.wo + 1.0f) / 2.0f; // map to [0,1]
		}
		if (settings->inputSettings.normal.otype == config::EncodingType::SphericalHarmonics)
		{
			tensors.normal = (tensors.normal + 1.0f) / 2.0f; // map to [0,1]
		}
		tensors.wi = pointerToTensor(inputPointers.wi, batchDim, 3);
		tensors.bsdfProb = pointerToTensor(inputPointers.bsdfProb, batchDim, 1, -65504, 65504);
		tensors.outRadiance = pointerToTensor(inputPointers.outRadiance, batchDim, 3);
		tensors.inRadiance = pointerToTensor(inputPointers.inRadiance, batchDim, 3);
		tensors.throughput = pointerToTensor(inputPointers.throughput, batchDim, 3);
		tensors.triangleId = pointerToTensor(inputPointers.triangleId, batchDim, 1);
		tensors.instanceId = pointerToTensor(inputPointers.instanceId, batchDim, 1);
		tensors.materialId = pointerToTensor(inputPointers.materialId, batchDim, 1);
		tensors.wiProb = pointerToTensor(inputPointers.wiProb, batchDim, 1);
		return tensors;
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PGNet::splitGuidingOutput(const torch::Tensor& rawOutput)
	{
		const torch::Tensor rawSamplingFraction = rawOutput.narrow(1, 0, 1);
		torch::Tensor rawMixtureParameters = rawOutput.narrow(1, 1, mixtureParametersCount);
		rawMixtureParameters = rawMixtureParameters.view({ rawOutput.size(0), settings->mixtureSize, distributionParametersCount });
		const torch::Tensor rawMixtureWeights = rawOutput.narrow(1, 1 + mixtureParametersCount, settings->mixtureSize);

		TRACE_TENSOR(rawMixtureParameters);
		TRACE_TENSOR(rawMixtureWeights);
		TRACE_TENSOR(rawSamplingFraction);

		return { rawMixtureParameters, rawMixtureWeights, rawSamplingFraction };
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PGNet::finalizeGuidingOutput(torch::Tensor& rawMixtureParameters, const torch::Tensor& rawMixtureWeights, const torch::Tensor& rawSamplingFraction)
	{
		torch::Tensor mixtureParameters = distribution::Mixture::finalizeParams(rawMixtureParameters, settings->distributionType);
		torch::Tensor mixtureWeights = softmax(rawMixtureWeights, 1);
		torch::Tensor cTensor = sigmoid(rawSamplingFraction/2.0f);
		TRACE_TENSOR(mixtureParameters);
		TRACE_TENSOR(mixtureWeights);
		TRACE_TENSOR(cTensor);
		return { mixtureParameters, mixtureWeights, cTensor };
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PGNet::guidingForward(const torch::Tensor& rawOutput)
	{
		const torch::Tensor rawOutputFullPrecision = rawOutput.to(torch::kFloat32);
		auto [rawMixtureParameters, rawMixtureWeights, rawSamplingFraction] = splitGuidingOutput(rawOutputFullPrecision);
		auto [mixtureParameters, mixtureWeights, cTensor] = finalizeGuidingOutput(rawMixtureParameters, rawMixtureWeights, rawSamplingFraction);
		return { mixtureParameters, mixtureWeights, cTensor };
	}

	torch::Tensor crossEntropy(const torch::Tensor& target, const torch::Tensor& prediction)
	{
		return -target * torch::log(prediction + bEps);
	}
	
	torch::Tensor sigmoidClamp(const torch::Tensor& input, const float clampValue, const float steepness)
	{
		torch::Tensor output = clampValue * torch::tanh(steepness * input);
		return output;
	}

	void printMinMaxLines(const torch::Tensor& target, const std::vector<torch::Tensor>& tensors) {
		// Find the indices of min and max values in the target tensor
		auto minIndex = target.argmin().item<int64_t>();
		auto maxIndex = target.argmax().item<int64_t>();

		// Iterate over the tensors and print the lines corresponding to min and max indices
		for (const auto& tensor : tensors) {
			// Ensure the tensor has the same batch size as the target tensor
			if (tensor.size(0) != target.size(0)) {
				std::cerr << "Tensor size mismatch in the batch dimension." << std::endl;
				continue;
			}

			std::cout << "Tensor line at min index (" << minIndex << "): \n" << tensor[minIndex] << std::endl;
			std::cout << "Tensor line at max index (" << maxIndex << "): \n" << tensor[maxIndex] << std::endl;
		}
	}
	torch::Tensor PGNet::guidingLoss(const torch::Tensor& neuralProb, torch::Tensor& bsdfProb, const torch::Tensor& outRadiance, const torch::Tensor& c, const torch::Tensor& sampleProb)
	{

		torch::Tensor targetLuminance = settings->targetScale * (outRadiance * ntscLuminance).sum(-1, true);
		// clamp to a max of 10k
		if (settings->clampBsdfProb)
		{
			bsdfProb = torch::min(bsdfProb, targetLuminance);
		}
		const torch::Tensor blendedQ = c * neuralProb + (1 - c) * bsdfProb;
		torch::Tensor lossQ = divergenceLoss(targetLuminance, neuralProb, sampleProb);
		torch::Tensor lossBlendedQ = divergenceLoss(targetLuminance, blendedQ, sampleProb);

		if(settings->scaleLossBlendedQ)
		{
			lossBlendedQ  = lossBlendedQ / (bsdfProb + 0.01);
		}

		if(settings->lossClamp!=0.0f)
		{
			lossBlendedQ = sigmoidClamp(lossBlendedQ, settings->lossClamp, 0.01f);
			lossQ = sigmoidClamp(lossQ, settings->lossClamp, 0.01f);
		}

		//float clampValue = 1.0f;
		//lossBlendedQ = torch::clamp(lossBlendedQ, -clampValue, clampValue);
		//lossQ = torch::clamp(lossQ, -clampValue, clampValue);

		torch::Tensor loss = lossBlendFactor() * lossQ + (1.0f - lossBlendFactor()) * lossBlendedQ;
		torch::Tensor additionalLoss = torch::zeros_like(loss);
		if(settings->distributionType == config::D_NASG_AXIS_ANGLE && false)
		{
			additionalLoss = (1.0f - lossBlendFactor()) * 0.5f * pow((0.9f - c), 2.0f);
			loss += additionalLoss;
		}

		//float lossMin = loss.min().item<float>();
		//float lossMax = loss.max().item<float>();
		// Loss min max
		if(false)
		{
			const auto minIndex = loss.argmin().item<int64_t>();
			const auto maxIndex = loss.argmax().item<int64_t>();

			VTX_INFO("LOSS MIN {} with NeuralProb {}, BSDFProb {}, TargetLuminance {}, outRadiance [{},{},{}], LossQ {}, LossBlendedQ {}, Loss {}, C {}, SampleProb {}",
				loss[minIndex].item<float>(),
				neuralProb[minIndex].item<float>(),
				bsdfProb[minIndex].item<float>(),
				targetLuminance[minIndex].item<float>(),
				outRadiance[minIndex][0].item<float>(),
				outRadiance[minIndex][1].item<float>(),
				outRadiance[minIndex][2].item<float>(),
				lossQ[minIndex].item<float>(),
				lossBlendedQ[minIndex].item<float>(),
				loss[minIndex].item<float>(),
				c[minIndex].item<float>(),
				sampleProb[minIndex].item<float>()
			);

			VTX_INFO("LOSS MAX {} with NeuralProb {}, BSDFProb {}, TargetLuminance {}, outRadiance [{},{},{}], LossQ {}, LossBlendedQ {}, Loss {}, C {}, SampleProb {}",
					loss[maxIndex].item<float>(),
					neuralProb[maxIndex].item<float>(),
					bsdfProb[maxIndex].item<float>(),
					targetLuminance[maxIndex].item<float>(),
					outRadiance[maxIndex][0].item<float>(),
					outRadiance[maxIndex][1].item<float>(),
					outRadiance[maxIndex][2].item<float>(),
					lossQ[maxIndex].item<float>(),
					lossBlendedQ[maxIndex].item<float>(),
					loss[maxIndex].item<float>(),
					c[maxIndex].item<float>(),
					sampleProb[maxIndex].item<float>()
			);
		}

		if (settings->scaleBySampleProb)
		{
			loss /= (sampleProb + 0.01);
		}
		//torch::Tensor debugPrint = torch::stack({ targetLuminance, neuralProb, lossQ}, -1);
		//PRINT_TENSOR_SLICE(debugPrint);
		//PRINT_TENSOR_MIN_MAX(loss);
		//PRINT_TENSOR_MIN_MAX(neuralProb);
		//PRINT_TENSOR_MIN_MAX(bsdfProb);
		//PRINT_TENSOR_MIN_MAX(targetLuminance);
		//PRINT_TENSOR_MIN_MAX(lossQ);
		//PRINT_TENSOR_MIN_MAX(lossBlendedQ);

		switch (settings->lossReduction)
		{
		case config::SUM:
			loss = loss.sum();
			break;
		case config::MEAN:
			loss = loss.mean();
			break;
		case config::ABS:
			loss = torch::abs(loss);
			break;
		default:
			VTX_ERROR("Loss Reduction Not implemented");
		}



		TRACE_TENSOR(loss);

		if (settings->plotGraphs)
		{
			graphs.addData(LOSS_PLOT, "Path Guiding Loss", loss.unsqueeze(-1).item<float>());
			graphs.addData(LOSS_PLOT, "Loss Q", lossQ.mean().unsqueeze(-1).item<float>());
			graphs.addData(LOSS_PLOT, "Loss Blended Q", lossBlendedQ.mean().unsqueeze(-1).item<float>());
			graphs.addData(LOSS_PLOT, "Loss Additional", additionalLoss.mean().unsqueeze(-1).item<float>());

			//graphs.addData(LOSS_PLOT, "Loss Min", lossMin);
			//graphs.addData(LOSS_PLOT, "Loss Max", lossMax);

			graphs.addData(PROB_PLOT, "Neural Prob", neuralProb.mean().unsqueeze(-1).item<float>());
			graphs.addData(PROB_PLOT, "Target Prob", targetLuminance.mean().unsqueeze(-1).item<float>());

			graphs.addData(SAMPLING_PROB_PLOT, "BSDF Prob", bsdfProb.mean().unsqueeze(-1).item<float>());
			graphs.addData(SAMPLING_PROB_PLOT, "Blended Prob", blendedQ.mean().unsqueeze(-1).item<float>());

			if (settings->scaleBySampleProb)
			{
				graphs.addData("Overall Prob", "Iteration", "Prob", "Overall Sample Prob", sampleProb.mean().unsqueeze(-1).item<float>());
			}

		}


		TRACE_TENSOR(blendedQ);
		TRACE_TENSOR(lossQ);
		TRACE_TENSOR(lossBlendedQ);
		TRACE_TENSOR(loss);
		return loss;
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PGNet::splitAuxiliaryOutput(const torch::Tensor& rawAuxiliaryOutput)
	{
		torch::Tensor incomingRadiance = rawAuxiliaryOutput.narrow(1, 0, 3);
		torch::Tensor throughput = rawAuxiliaryOutput.narrow(1, 3, 3);
		torch::Tensor outgoingRadiance = rawAuxiliaryOutput.narrow(1, 6, 3);
		TRACE_TENSOR(incomingRadiance);
		TRACE_TENSOR(throughput);
		TRACE_TENSOR(outgoingRadiance);
		return { incomingRadiance, throughput, outgoingRadiance };
	}

	torch::Tensor PGNet::relativeL2LossRadiance(const torch::Tensor& radiance, const torch::Tensor& radianceTarget)
	{
		const torch::Tensor scaledTarget = (radianceTarget * settings->radianceTargetScaleFactor).clamp(0.0f, 1e+6f); // GfxExp says it is required to avoid dominance of the epsilon
		const torch::Tensor difference = radiance - scaledTarget;
		const torch::Tensor luminance = (radiance * ntscLuminance).sum(-1, true).detach();
		const torch::Tensor predictionSqPlusEpsilon = torch::pow(luminance, 2) + 0.01f;
		const torch::Tensor relativeL2 = torch::pow(difference, 2) / predictionSqPlusEpsilon;
		const torch::Tensor relativeL2Mean = relativeL2;// .mean();

		TRACE_TENSOR(relativeL2);

		return relativeL2Mean;
	}

	torch::Tensor PGNet::relativeL2LossThroughput(const torch::Tensor& throughput, const torch::Tensor& throughputTarget)
	{
		const torch::Tensor scaledTarget = (throughputTarget * settings->throughputTargetScaleFactor).clamp(0.0f, 1e+6f); // GfxExp says it is required to avoid dominance of the epsilon
		const torch::Tensor difference = throughput - scaledTarget;

		const torch::Tensor denominator = torch::pow(throughput, 2).detach() + 0.01f;
		const torch::Tensor relativeL2 = torch::pow(difference, 2) / denominator;
		torch::Tensor relativeL2Mean = relativeL2.mean();

		//PRINT_TENSOR_SLICE(throughputTarget);
		//PRINT_TENSOR_SLICE(scaledTarget);
		//PRINT_TENSOR_SLICE(throughput);
		//PRINT_TENSOR_SLICE(difference);
		//PRINT_TENSOR_SLICE(denominator);
		//PRINT_TENSOR_SLICE(relativeL2);
		//PRINT_TENSOR_ALWAYS("relativeL2Mean", relativeL2Mean);

		TRACE_TENSOR(denominator);
		TRACE_TENSOR(relativeL2);

		return relativeL2Mean;
	}

	torch::Tensor PGNet::auxiliaryLoss(
		const torch::Tensor& inRadiance, const torch::Tensor& inRadianceTarget,
		const torch::Tensor& throughput, const torch::Tensor& throughputTarget,
		const torch::Tensor& outRadiance, const torch::Tensor& outRadianceTarget)

	{
		const float totWeights = settings->inRadianceLossFactor + settings->outRadianceLossFactor + settings->throughputLossFactor;
		const float inRadScale = settings->inRadianceLossFactor; // / totWeights;
		const float outRadScale = settings->outRadianceLossFactor;// / totWeights;
		const float throughputScale = settings->throughputLossFactor;// / totWeights;

		const torch::Tensor lossInRadiance = inRadScale * relativeL2LossRadiance(inRadiance, inRadianceTarget);
		const torch::Tensor lossOutRadiance = outRadScale * relativeL2LossRadiance(outRadiance, outRadianceTarget);
		const torch::Tensor lossThroughput = throughputScale * relativeL2LossThroughput(throughput, throughputTarget);

		torch::Tensor loss =( settings->auxiliaryWeight * (lossInRadiance +  lossOutRadiance +  lossThroughput)).mean();

		if (settings->plotGraphs)
		{
			graphs.addData("Auxiliary Loss","Iteration","Loss", "Input Radiance loss", lossInRadiance.mean().item<float>());
			graphs.addData("Auxiliary Loss", "Iteration", "Loss", "Output Radiance loss", lossOutRadiance.mean().item<float>());
			graphs.addData("Auxiliary Loss", "Iteration", "Loss", "Throughput loss", lossThroughput.mean().item<float>());
			graphs.addData("Auxiliary Loss", "Iteration", "Loss", "Total loss", loss.item<float>());
		}
		TRACE_TENSOR(loss);

		return loss;
	}

	torch::Tensor PGNet::guidingEntropyLoss(const torch::Tensor& neuralProb)
	{
		const torch::Tensor entropy = -torch::sum(neuralProb * torch::log(neuralProb + bEps), 1);
		const float targetEntropy = -logf(settings->targetEntropy);
		torch::Tensor entropyLoss = torch::pow(entropy - targetEntropy, 2).mean();
		TRACE_TENSOR(entropy);
		TRACE_TENSOR(entropyLoss);
		return entropyLoss;
	}

	torch::Tensor PGNet::divergenceLoss(const torch::Tensor& targetProb, const torch::Tensor& neuralProb, const torch::Tensor& wiProb)
	{

		torch::Tensor loss;
		switch (settings->lossType)
		{
		case config::L_KL_DIV:
			loss = kl_div(torch::log(neuralProb + bEps), targetProb, at::Reduction::None);
			break;
		case config::L_KL_DIV_MC_ESTIMATION:
			{
				const torch::Tensor weights = 1.0f/ (wiProb + bEps);
				loss = -targetProb * log(neuralProb + bEps);
				loss = weights * loss;
			}
			break;
		case config::L_KL_DIV_MC_ESTIMATION_NORMALIZED:
			{
				const torch::Tensor logRatio = torch::log(targetProb+bEps) -torch::log(neuralProb + bEps);
				const torch::Tensor weights = targetProb / (wiProb + bEps);
				loss = weights * logRatio;
			}
			break;
		case config::L_MSE:
			{
				loss = torch::pow(targetProb - neuralProb, 2.0f);
			}
			break;
		case config::L_PEARSON_DIV_MC_ESTIMATION:
			{
				const torch::Tensor sampleWeight = 1.0f / (wiProb + bEps);
				loss = torch::pow(targetProb, 2.0f) / (neuralProb + bEps);
				loss = loss * sampleWeight;
			}
			break;
		default:
			VTX_ERROR("Loss Type Not implemented");
			loss = torch::zeros({ 1 });
		}
		TRACE_TENSOR(loss);

		return loss;
	}

	float PGNet::lossBlendFactor()
	{
		if (settings->constantBlendFactor)
		{
			return settings->blendFactor;
		}
		return std::min(1.0f, pow(0.3333f, 5.0f * tau()));
	}

	float PGNet::tau()
	{
		float tau = std::min(1.0f, (float)trainingStep / ((float)settings->maxTrainingSteps));
		return tau;
	}

	void InputTensors::printInfo()
	{
		if (position.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(position);
		}
		else{
			VTX_INFO("Tensor position is not defined");
		}
		if (wo.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(wo);
		}
		else{
			VTX_INFO("Tensor wo is not defined");
		}
		if (normal.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(normal);
		}
		else{
			VTX_INFO("Tensor normal is not defined");
		}
		if (wi.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(wi);
		}
		else{
			VTX_INFO("Tensor wi is not defined");
		}
		if (bsdfProb.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(bsdfProb);
		}
		else{
			VTX_INFO("Tensor bsdfProb is not defined");
		}
		if (outRadiance.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(outRadiance);
		}
		else{
			VTX_INFO("Tensor outRadiance is not defined");
		}
		if (throughput.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(throughput);
		}
		else{
			VTX_INFO("Tensor throughput is not defined");
		}
		if (inRadiance.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(inRadiance);
		}
		else{
			VTX_INFO("Tensor inRadiance is not defined");
		}
		if (instanceId.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(instanceId);
		}
		else{
			VTX_INFO("Tensor instanceId is not defined");
		}
		if (triangleId.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(triangleId);
		}
		else{
			VTX_INFO("Tensor triangleId is not defined");
		}
		if (materialId.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(materialId);
		}
		else{
			VTX_INFO("Tensor materialId is not defined");
		}
		if (wiProb.defined()){
			PRINT_TENSOR_SIZE_ALWAYS(wiProb);
		}
		else{
			VTX_INFO("Tensor wiProb is not defined");
		}
	}
}
