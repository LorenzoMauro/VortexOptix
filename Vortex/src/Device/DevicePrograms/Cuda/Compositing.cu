
#include "../CudaKernels.h"
#include "device_launch_parameters.h"
#include "Device/CUDAChecks.h"
#include "Device/DevicePrograms/ToneMapper.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Device/Wrappers/KernelLaunch.h"
#include "NeuralNetworks/Interface/NetworkInterface.h"

namespace vtx
{
	__forceinline__ __device__ math::vec3f fireflyRemoval(const math::vec3f* inputBuffer, const int x, const int y, const int width, const int height, const int kernelSize, const NoiseType noiseType, const float threshold)
	{
		// kernelSize is assumed to be an odd number
		const int halfKernel = kernelSize / 2;

		math::vec3f centerValue;
		if (noiseType == LUMINANCE)
		{
			centerValue = inputBuffer[y * width + x];
		}

		math::vec3f sumValue = math::vec3f(0.0f);
		int   count = 0;

		for (int dx = -halfKernel; dx <= halfKernel; ++dx)
		{
			for (int dy = -halfKernel; dy <= halfKernel; ++dy)
			{
				const int nx = x + dx;
				const int ny = y + dy;

				// exclude the center pixel
				if ((dx != 0 || dy != 0) && nx >= 0 && nx < width && ny >= 0 && ny < height)
				{
					sumValue += inputBuffer[ny * width + nx];
					count++;
				}
			}
		}

		// Calculate average outRadiance in the kernel (excluding center)
		float avgLuminance = utl::luminance(sumValue / count);

		// If the center pixel's outRadiance is higher than average plus threshold
		// replace it with average, otherwise keep the original
		if (utl::luminance(centerValue) > avgLuminance*threshold)
		{
			return sumValue / count;
		}

		return inputBuffer[y * width + x];
	}


	__global__ void fireFlyPass(const LaunchParams* launchParams, const int kernelSize, const NoiseType noiseType, const float threshold)
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;

		const FrameBufferData* frameBuffer = &launchParams->frameBuffer;
		const math::vec2ui& frameSize = frameBuffer->frameSize;
		if (x >= frameSize.x || y >= frameSize.y) return;

		const uint32_t fbIndex = x + y * frameSize.x;

		/*const math::vec3f& directLightBuffer = frameBuffer->directLight[fbIndex];
		const math::vec3f& transmissionIndirect = frameBuffer->transmissionIndirect[fbIndex];

		const math::vec3f diffuse = fireflyRemoval(frameBuffer->diffuseIndirect, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);
		const math::vec3f glossy = fireflyRemoval(frameBuffer->glossyIndirect, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);
		const math::vec3f transmission = fireflyRemoval(frameBuffer->transmissionIndirect, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);*/

		const math::vec3f filteredRadiance = fireflyRemoval(frameBuffer->hdriRadiance, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);

		frameBuffer->fireflyPass[fbIndex] = filteredRadiance;
	}

	__forceinline__ __device__ void prepareOutput(math::vec3f* inputBuffer, const LaunchParams* params, const int pixelId, const bool normalizeBySamples, const bool dotoneMap)
	{
		math::vec3f output3f = inputBuffer[pixelId];
		if(normalizeBySamples)
		{
			output3f /= params->frameBuffer.samples[pixelId];
		}
		if(dotoneMap)
		{
			output3f = toneMap(params->settings.renderer.toneMapperSettings, output3f);
		}
		reinterpret_cast<math::vec4f*>(params->frameBuffer.outputBuffer)[pixelId] = math::vec4f(output3f, 1.0f);
	}

	__forceinline__ __device__ void toneMapRadiance(const int id, const LaunchParams* params)
	{
		params->frameBuffer.hdriRadiance[id] = params->frameBuffer.radianceAccumulator[id] / params->frameBuffer.samples[id];
		params->frameBuffer.normalNormalized[id] = params->frameBuffer.normalAccumulator[id] / params->frameBuffer.samples[id];
		params->frameBuffer.albedoNormalized[id] = params->frameBuffer.albedoAccumulator[id] / params->frameBuffer.samples[id];
		params->frameBuffer.tmRadiance[id]= toneMap(params->settings.renderer.toneMapperSettings, params->frameBuffer.hdriRadiance[id]);

	}

	__global__ void outputSelector(const LaunchParams* launchParams, math::vec3f* beauty) {

		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;
		const FrameBufferData* frameBuffer = &launchParams->frameBuffer;
		const math::vec2ui& frameSize = frameBuffer->frameSize;
		if (x >= frameSize.x || y >= frameSize.y) return;

		const uint32_t fbIndex = x + y * frameSize.x;
		const OnDeviceSettings settings = launchParams->settings;
		const auto                    outputBuffer = reinterpret_cast<math::vec4f*>(frameBuffer->outputBuffer);

		math::vec3f* input = nullptr;
		bool normalizeBySamples = true;
		bool dotoneMap = true;

		/*if(launchParams->settings.neural.active)
		{
			launchParams->networkInterface->paths->accumulatePath(fbIndex);
		}*/

		switch (settings.renderer.displayBuffer)
		{

		case(FB_BEAUTY):
		{
			if (beauty != nullptr)
			{
				input = beauty;
				dotoneMap = true;
			}
			else
			{
				input = frameBuffer->tmRadiance;
				dotoneMap = false;
			}
			normalizeBySamples = false;
		}
		break;
		case(FB_NOISY):
		{
			input = frameBuffer->tmRadiance;
			dotoneMap = false;
			normalizeBySamples = false;
		}
		break;

		case(FB_DIFFUSE):
		{
			input = frameBuffer->albedoNormalized;
			dotoneMap = false;
			normalizeBySamples = false;
		}
		break;
		case(FB_ORIENTATION):
		{
			input = frameBuffer->orientation;
			dotoneMap = false;
		}
		break;
		case(FB_TRUE_NORMAL):
		{
			input = frameBuffer->trueNormal;
			dotoneMap = false;
		}
		break;
		case(FB_SHADING_NORMAL):
		{
			input = frameBuffer->normalNormalized;
			dotoneMap = false;
			normalizeBySamples = false;
		}
		break;
		case(FB_TANGENT):
		{
			input = frameBuffer->tangent;
			dotoneMap = false;
		}
		break;
		case(FB_UV):
		{
			input = frameBuffer->uv;
			dotoneMap = false;
		}
		break;
		case(FB_NOISE):
		{
			math::vec3f value = floatToScientificRGB(frameBuffer->noiseBuffer[fbIndex].noiseAbsolute);
			outputBuffer[fbIndex] = math::vec4f(value, 1.0f);
		}
		break;
		case(FB_GBUFFER):
		{
			vtxID instanceID = frameBuffer->gBuffer[fbIndex];
			instanceID = ((instanceID << 5) | (instanceID >> 27)) ^ ((instanceID << 13) | (instanceID >> 19));;

			math::vec3f color;
			color.x = (instanceID & 0x000000FF) / 255.0f;
			color.y = ((instanceID & 0x0000FF00) >> 8) / 255.0f;
			color.z = ((instanceID & 0x00FF0000) >> 16) / 255.0f;
			outputBuffer[fbIndex] = math::vec4f(color, 1.0f);
		}
		case(FB_SAMPLES):
		{
			float sampleMetric = (float)(frameBuffer->samples[fbIndex] - launchParams->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples) / (float)(launchParams->settings.renderer.iteration - launchParams->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples);
			sampleMetric *= 0.01f;
			sampleMetric = toneMap(launchParams->settings.renderer.toneMapperSettings, math::vec3f(sampleMetric)).x;

			math::vec3f value = floatToScientificRGB(sampleMetric);
			outputBuffer[fbIndex] = math::vec4f(value, 1.0f);
		}
		break;
		case(FB_DEBUG_1):
		{
			input = frameBuffer->debugColor1;
			normalizeBySamples = true;
			//outputBuffer[fbIndex] = math::vec4f(frameBuffer->debugColor1[fbIndex], 1.0f);
			dotoneMap = false;
		}
		break;
		case(FB_NETWORK_INFERENCE_STATE_POSITION):
		case(FB_NETWORK_INFERENCE_STATE_NORMAL):
		case(FB_NETWORK_INFERENCE_OUTGOING_DIRECTION):
		case(FB_NETWORK_INFERENCE_MEAN):
		case(FB_NETWORK_INFERENCE_SAMPLE):
		case(FB_NETWORK_INFERENCE_SAMPLE_DEBUG):
		case(FB_NETWORK_INFERENCE_CONCENTRATION):
		case(FB_NETWORK_INFERENCE_ANISOTROPY):
		case(FB_NETWORK_INFERENCE_PDF):
		case(FB_NETWORK_INFERENCE_IS_FRONT_FACE):
		case(FB_NETWORK_INFERENCE_SAMPLING_FRACTION):
		{
			if (launchParams->settings.neural.active)
			{
				outputBuffer[fbIndex] = math::vec4f{ launchParams->networkInterface->debugBuffers->inferenceDebugBuffer[fbIndex], 1.0f };
			}
		}
		break;
		case(FB_NETWORK_REPLAY_BUFFER_REWARD):
		{
			//if (launchParams->settings.neural.active)
			//{
			//	/*outputBuffer[fbIndex] = math::vec4f{ launchParams->networkInterface->debugBuffer2[fbIndex] / (launchParams->networkInterface->debugBuffer3[fbIndex].x), 1.0f };*/
			//	const float value = launchParams->networkInterface->debugBuffer1[fbIndex].x / (float)launchParams->networkInterface->debugBuffer1[fbIndex].z;
			//	outputBuffer[fbIndex] = math::vec4f(math::vec3f(value), 1.0f);// math::vec4f(floatToScientificRGB(value), 1.0f);
			//}
		}
		break;
		case(FB_NETWORK_REPLAY_BUFFER_SAMPLES):
		{
			//if (launchParams->settings.neural.active)
			//{
			//	math::vec3f* debugBuffer = launchParams->networkInterface->debugBuffer3;

			//	/*if (launchParams->networkInterface->paths->paths[fbIndex].isDepthAtTerminalZero)
			//	{
			//		debugBuffer[fbIndex].y += 1.0f;
			//	}
			//	float value = debugBuffer[fbIndex].y / ((float)launchParams->settings.renderer.iteration+1.0f);

			//	math::vec3f color = floatToScientificRGB(value);
			//	outputBuffer[fbIndex] = math::vec4f(color, 1.0f);*/

			//	const int& batchSize = launchParams->settings.neural.batchSize;
			//	int totPixel = frameSize.x * frameSize.y;
			//	int iteration = launchParams->frameBuffer.samples[fbIndex] + 1;
			//	//const int& numberOfTrainingStep = launchParams->settings.neural.maxTrainingStepPerFrame;
			//	int totSamples = batchSize * iteration; // *numberOfTrainingStep;
			//	float maxSamplesPerPixel = (float)totSamples / (float)(totPixel);

			//	float& totSampleAtPixel = debugBuffer[fbIndex].x;
			//	float value = fmaxf(0.0f, fminf(totSampleAtPixel / maxSamplesPerPixel, 1.0f));
			//	if (value < 0.0f || value > 1.0f || isnan(value))
			//	{
			//		printf(
			//			"ToT sample at pixel %d is %f\n"
			//			"Max sample per pixel is %f\n",
			//			fbIndex,
			//			totSampleAtPixel,
			//			maxSamplesPerPixel
			//		);
			//	}

			//	math::vec3f color = floatToScientificRGB(value);
			//	outputBuffer[fbIndex] = math::vec4f(color, 1.0f);
			//}
			

		}
		break;
		case(FB_NETWORK_DEBUG_PATHS):
		{
			if (launchParams->settings.neural.active)
			{

				math::vec4f filmBuffer = launchParams->networkInterface->debugBuffers->filmBuffer[fbIndex];
				math::vec3f reconstructedRadiance = { filmBuffer.x, filmBuffer.y, filmBuffer.z };
				reconstructedRadiance /= filmBuffer.w;
				reconstructedRadiance = toneMap(launchParams->settings.renderer.toneMapperSettings, reconstructedRadiance);
				outputBuffer[fbIndex] = math::vec4f(reconstructedRadiance, 1.0f);

				//const math::vec3f& integratedRadiance = frameBuffer->tmRadiance[fbIndex];
				//const math::vec3f difference = integratedRadiance - reconstructedRadiance;
				//const float value = math::length(difference) * 0.5f;
				//const math::vec3f fts = floatToScientificRGB(value);
				//outputBuffer[fbIndex] = math::vec4f(fts, 1.0f);
			}
		}
		break;
		}
		

		if(input!=nullptr)
		{
			prepareOutput(input, launchParams, fbIndex, normalizeBySamples, dotoneMap);
		}
	}

	void removeFireflies(const LaunchParams* launchParams, const int kernelSize, const float threshold, const int width, const int height)
	{
		dim3 threadsPerBlock(16, 16);  // a common choice for 2D data
		// Calculate the number of blocks needed in each dimension
		dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		fireFlyPass << <numBlocks, threadsPerBlock >> > (launchParams, kernelSize, LUMINANCE, threshold);
		CUDA_SYNC_CHECK();
	}

	void switchOutput(const LaunchParams* launchParams, const int width, const int height, math::vec3f* beauty)
	{
		dim3 threadsPerBlock(16, 16);  // a common choice for 2D data
		// Calculate the number of blocks needed in each dimension
		dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		outputSelector <<<numBlocks, threadsPerBlock>>>(launchParams, beauty);
		//CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_SYNC_CHECK();
	}

	void toneMapRadianceKernel(const LaunchParams* launchParams, const int width, const int height, const char* name)
	{
		gpuParallelFor(name,
			width * height,
			[=] __device__(const int id)
		{
			toneMapRadiance(id, launchParams);
		});
	}

	void toneMapBuffer(CUDABuffer& src, CUDABuffer& dst, const int width, const int height, ToneMapperSettings settings)
	{
		const auto* const srcDevice = src.castedPointer<math::vec3f>();
		auto*                    dstDevice = dst.castedPointer<math::vec3f>();

		gpuParallelFor("ToneMap",
			width * height,
			[=] __device__(const int id)
		{
			dstDevice[id] = toneMap(settings, srcDevice[id]);
		});
	}
}
