#pragma once
#include "ErrorTypes.h"
#include "Device/DevicePrograms/nvccUtils.h"
#include "Core/Math.h"
#include "Core/VortexID.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Device/DevicePrograms/Utils.h"
#include "Device/UploadCode/CUDABuffer.h"
#include "Device/Wrappers/KernelLaunch.h"

#define CUDA_INTERFACE
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "NeuralNetworks/Distributions/Mixture.h"
#include "NeuralNetworks/Interface/NetworkInterface.h"

namespace vtx::cuda{

#define BLACK math::vec3f(0.0f, 0.0f, 0.0f)
#define BLUE math::vec3f(0.0f, 0.0f, 1.0f)
#define PURPLE math::vec3f(0.5f, 0.0f, 0.5f)
#define YELLOW math::vec3f(1.0f, 1.0f, 0.0f)
#define WHITE math::vec3f(1.0f, 1.0f, 1.0f)

	__forceinline__ __device__ math::vec3f logViridisMagmaColorMap(float value)
	{
		// Define the breakpoints for the value ranges
		const float breakpoint1 = 0.5f;
		const float breakpoint2 = 1.5f;
		const float breakpoint3 = 10.0f;
		const float breakpoint4 = 100.0f;

		math::vec3f color;
		if (value <= breakpoint1)
		{
			// Interpolate between BLACK and BLUE
			float lerpValue = value / breakpoint1;
			color = BLACK * (1.0f - lerpValue) + BLUE * lerpValue;
		}
		else if (value <= breakpoint2)
		{
			// Interpolate between BLUE and PURPLE
			float lerpValue = (value - breakpoint1) / (breakpoint2 - breakpoint1);
			color = BLUE * (1.0f - lerpValue) + PURPLE * lerpValue;
		}
		else if (value <= breakpoint3)
		{
			// Interpolate between PURPLE and YELLOW
			float lerpValue = (value - breakpoint2) / (breakpoint3 - breakpoint2);
			color = PURPLE * (1.0f - lerpValue) + YELLOW * lerpValue;
		}
		else if (value <= breakpoint4)
		{
			// Interpolate between YELLOW and WHITE
			float lerpValue = (value - breakpoint3) / (breakpoint4 - breakpoint3);
			color = YELLOW * (1.0f - lerpValue) + WHITE * lerpValue;
		}
		else
		{
			// Value is above the highest breakpoint, so we set it to WHITE
			color = WHITE;
		}

		return color;
	}

	__forceinline__ __device__ void printDistribution(
		const int id,
		const int width,
		const int height,
		const LaunchParams * const params,
		const math::vec3f& mean,
		const math::vec3f& normal,
		const math::vec3f& sample,
		math::vec3f* buffer
	)
	{
		const float x = (int)(id % width) / (float)width; //range in [0, 1]
		const float y = (int)(id / width) / (float)height; //range in [0, 1]
		const float u = x*2.0f-1.0f; //range in [-1, 1]
		const float v = y*2.0f-1.0f; //range in [-1, 1]
		const float theta = M_PI * u;  // Horizontal angle, azimuthal
		const float phi = acosf(v);  // Vertical angle, polar
		
		const math::vec3f dir = math::vec3f(
			sinf(phi) * cosf(theta),  // x
			sinf(phi) * sinf(theta),  // y
			cosf(phi)                 // z
		);
		if(length(dir-mean) < 0.04f)
		{
			buffer[id] = math::vec3f(1.0f, 0.0f, 1.0f);
			return;
		}
		if (length(dir-normal) < 0.04f)
		{
			buffer[id] = math::vec3f(0.0f, 1.0f, 0.5f);
			return;
		}
		if(length(dir-sample) < 0.04f)
		{
			buffer[id] = math::vec3f(1.0f, 0.0f, 0.0f);
			return;
		}

		const float* mixtureWeights = params->networkInterface->debugInfo->mixtureWeights;
		const float* mixtureParams = params->networkInterface->debugInfo->mixtureParameters;
		if (mixtureWeights == nullptr || mixtureParams == nullptr)
		{
			buffer[id] = math::vec3f(0.0f);
			return;
		}
		const int mixtureSize = params->settings.neural.mixtureSize;
		const auto type = params->settings.neural.distributionType;
		const float pdf = distribution::Mixture::evaluate(mixtureParams, mixtureWeights, mixtureSize, type, dir);
		buffer[id] = logViridisMagmaColorMap(pdf);
	}

	void printDistribution(
		CUDABuffer& buffer,
		const int width,
		const int height,
		const math::vec3f& mean,
		const math::vec3f& normal,
		const math::vec3f& sample
	)
	{
		if (buffer.bytesSize() != width * height * sizeof(math::vec3f))
			buffer.resize(width * height * sizeof(math::vec3f));

		LaunchParams* params = onDeviceData->launchParamsData.getDeviceImage();
		gpuParallelFor(eventNames[K_DISTRIBUTION_PRINT],
		width * height,
		[width, height, params, mean, normal, sample, bufferPtr = buffer.castedPointer<math::vec3f>()] __device__(const int id)
		{
			printDistribution(id, width, height, params, mean, normal, sample, bufferPtr);
		});
	}
	static const int kernelSize = 1;

	__forceinline__ __device__ int getPixelFromDirection(const math::vec3f& direction, const int width, const int height)
	{

		// 1. Convert from Cartesian to Spherical Coordinates
		const float phi = acosf(direction.z);
		const float theta = atan2f(direction.y, direction.x);

		// 2. Convert Spherical Coordinates to Normalized Screen Coordinates (u, v)
		const float v = cosf(phi);
		const float u = theta / M_PI;

		// 3. Convert Normalized Screen Coordinates to Pixel Coordinates (x, y)
		const float y = (v + 1.0f) * 0.5f * height;
		const float x = (u + 1.0f) * 0.5f * width;

		// 4. Calculate the ID
		const int bufferid = static_cast<int>(y) * width + static_cast<int>(x);

		return bufferid;
	}
	
	__forceinline__ __device__ void accumulateSample(
		const int id,
		const int pixId,
		const LaunchParams* params,
		const int width,
		const int height,
		math::vec3f* buffer)
	{
		// Calculate offsets
		const int offsetX = (id % kernelSize) - (kernelSize / 2);
		const int offsetY = (id / kernelSize) - (kernelSize / 2);
		const int actualPixel = pixId + offsetX + offsetY * width;

		if (params->networkInterface->maxPathLength[actualPixel] < 0)
		{
			return;
		}
		auto& p = params->networkInterface->get(actualPixel, 0);
		NetworkInterface::finalizePathStatic(params->settings.neural, params->networkInterface->maxPathLength[actualPixel], &p);

		buffer[getPixelFromDirection(p.bsdfSample.wi, width, height)] = (math::isZero(p.bsdfSample.Lo)) ? BLUE : YELLOW;
		//if(p.lightSample.valid)
		//{
		//	buffer[getPixelFromDirection(p.lightSample.wi, width, height)] += p.lightSample.Lo;
		//}
	}

	static int prevPixelId = -1;

	void accumulateAtDebugBounce(
		CUDABuffer& buffer,
		const int width,
		const int height,
		const int pixel)
	{
		if (buffer.bytesSize() != width * height * sizeof(math::vec3f) || prevPixelId != pixel)
		{
			buffer.free();
			buffer.resize(width * height * sizeof(math::vec3f));
			prevPixelId = pixel;
		}

		LaunchParams* params = onDeviceData->launchParamsData.getDeviceImage();
		gpuParallelFor(eventNames[K_DISTRIBUTION_PRINT],
			kernelSize*kernelSize,
			[pixel, width, height, params, bufferPtr = buffer.castedPointer<math::vec3f>()] __device__(const int id)
		{
			accumulateSample(id, pixel, params, width, height, bufferPtr);
		});
	}

	void copyRGBtoRGBA(const CUDABuffer& src, CUDABuffer&  dst, const int& width, const int& height)
	{
		auto dstSize = width * height * 4 * sizeof(float);
		if(dst.bytesSize() != dstSize)
			dst.resize(dstSize);

		const int   size   = width * height;
		auto* const srcPtr = src.castedPointer<math::vec3f>();
		auto* const dstPtr = dst.castedPointer<math::vec4f>();
		gpuParallelFor(eventNames[K_MAPE],
			size,
			[srcPtr, dstPtr] __device__(const int id)
		{
			dstPtr[id] = math::vec4f(srcPtr[id], 1.0f);
		});
	}


	void copyRtoRGBA(const CUDABuffer& src, CUDABuffer&  dst, const int& width, const int& height)
	{
		if (const auto dstSize = width * height * 4 * sizeof(float); dst.bytesSize() != dstSize)
			dst.resize(dstSize);

		const int   size = width * height;
		auto* const srcPtr = src.castedPointer<float>();
		auto* const dstPtr = dst.castedPointer<math::vec4f>();
		gpuParallelFor(eventNames[K_MAPE],
			size,
			[srcPtr, dstPtr] __device__(const int id)
		{
			dstPtr[id] = math::vec4f(srcPtr[id]);
		});
	}



	__forceinline__ __device__ void computeMapeDevice(const math::vec3f* target, const math::vec3f* input, math::vec2f* mape, const int& id)
	{
		constexpr float e = 0.01f;
		const float rMape = fabsf(input[id].x - target[id].x) / (target[id].x + e);
		const float gMape = fabsf(input[id].y - target[id].y) / (target[id].y + e);
		const float bMape = fabsf(input[id].z - target[id].z) / (target[id].z + e);
		float newValue = (rMape + gMape + bMape) / 3.0f;
		float& mapeValue = (*mape).x;
		float& count = (*mape).y;
		float addedMapeValue = cuAtomicAdd(&mapeValue, newValue);
		float addedCountValue = cuAtomicAdd(&count, 1);
	}

	__forceinline__ __device__ void computeMseDevice(const math::vec3f* target, const math::vec3f* input, math::vec2f* mse, const int& id)
	{
		const math::vec3f delta = ((input[id] - target[id]));
		const float rMse = delta.x * delta.x;
		const float gMse = delta.y * delta.y;
		const float bMse = delta.z * delta.z;
		float newValue = (rMse + gMse + bMse) / 3.0f;
		float& mseValue = (*mse).x;
		float& count = (*mse).y;
		float addedMseValue = cuAtomicAdd(&mseValue, newValue);
		float addedCountValue = cuAtomicAdd(&count, 1);

	}

	float computeMape(math::vec3f* groundTruth, math::vec3f* input, const int& width, const int& height)
	{
		CUDABuffer mapeBuffer;
		mapeBuffer.resize(sizeof(math::vec2f));
		mapeBuffer.upload(math::vec2f{0.0f, 0.0f});
		auto* mape = mapeBuffer.castedPointer<math::vec2f>();

		const int size = width * height;
		gpuParallelFor(eventNames[K_MAPE],
			size,
			[groundTruth, input, mape] __device__(const int id)
		{
			computeMapeDevice(groundTruth, input, mape, id);
		});

		math::vec2f hostMape;
		mapeBuffer.download(&hostMape);
		const float mapeValue = hostMape.x / hostMape.y;
		mapeBuffer.free();

		return mapeValue;
	}


	float computeMse(math::vec3f* groundTruth, math::vec3f* input, const int& width, const int& height)
	{
		CUDABuffer mseBuffer;
		mseBuffer.resize(sizeof(math::vec2f));
		mseBuffer.upload(math::vec2f{0.0f, 0.0f});
		auto* mse = mseBuffer.castedPointer<math::vec2f>();

		const int size = width * height;
		gpuParallelFor(eventNames[K_MSE],
			size,
			[groundTruth, input, mse] __device__(const int id)
		{
			computeMseDevice(groundTruth, input, mse, id);
		});

		math::vec2f hostMse;
		mseBuffer.download(&hostMse);
		const float mseValue = hostMse.x / hostMse.y;
		mseBuffer.free();

		return mseValue;
	}

	__forceinline__ __device__ float sum(const math::vec3f& vec)
	{
		return vec.x + vec.y + vec.z;
	}

	__forceinline__ __device__ void increaseSumAndCount(math::vec2f& sumCount, float newVal)
	{
		float                addedMseValue = cuAtomicAdd(&sumCount.x, newVal);
		float                addedMseCountValue = cuAtomicAdd(&sumCount.y, 1);
	}

	Errors computeErrors(const CUDABuffer& reference, const CUDABuffer& input, CUDABuffer& errorMaps, const int& width, const int& height)
	{
		const size_t errorMapSize = width * height * ((int)ErrorType::ERROR_TYPE_COUNT) * sizeof(float);
		if (errorMaps.bytesSize() != errorMapSize)
		{
			errorMaps.resize(errorMapSize);
		}

		CUDABuffer errorsSumBuffer;
		std::vector<math::vec2f> errorsSum((int)ErrorType::ERROR_TYPE_COUNT, math::vec2f{0.0f, 0.0f});
		errorsSumBuffer.upload(errorsSum);

		auto* const referencePtr = reference.castedPointer<math::vec3f>();
		auto* const inputPtr = input.castedPointer<math::vec3f>();
		auto* const errorMapsPtr = errorMaps.castedPointer<float>();
		auto* const errorsSumPtr = errorsSumBuffer.castedPointer<math::vec2f>();
		const int size = width * height;


		gpuParallelFor(eventNames[K_MSE],
			size,
			[referencePtr, inputPtr, errorMapsPtr, errorsSumPtr, size] __device__(const int id)
		{

			const math::vec3f delta = (inputPtr[id] - referencePtr[id]);

			const float mse = sum(delta * delta) / 3.0f;
			const float mape = sum(abs(delta) / (referencePtr[id] + 0.01f)) / 3.0f;

			errorMapsPtr[size*(int)ErrorType::MAPE + id] = mape;
			errorMapsPtr[size * (int)ErrorType::MSE + id] = mse * 10.0f;

			increaseSumAndCount(errorsSumPtr[(int)ErrorType::MSE], mse);
			increaseSumAndCount(errorsSumPtr[(int)ErrorType::MAPE], mape);
		});


		errorsSumBuffer.download(errorsSum.data());

		Errors errors;
		errors.mse = errorsSum[(int)ErrorType::MSE].x / errorsSum[(int)ErrorType::MSE].y;
		errors.mape = errorsSum[(int)ErrorType::MAPE].x / errorsSum[(int)ErrorType::MAPE].y;
		errors.dMapeMap = errorMapsPtr + size * (int)ErrorType::MAPE;
		errors.dMseMap = errorMapsPtr + size * (int)ErrorType::MSE;

		return errors;
	}

	__forceinline__ __device__ float edgeTransform(const float x, const float curvature, const float scale) {
		if (x > 0.0f && x < 1.0f) {
			//float value = fminf(1.0f, scale * expf((-powf((x - 1.0f), 2.0f)) / curvature));
			float tX = x - 1.0f;
			float value = scale * expf(-(tX * tX) / curvature);
			return value;
		}
		else {
			return 0.0f;
		}
	}

	__forceinline__ __device__ void overlayEdgeKernel(
		const int id,
		const int width,
		const float* edgeData,
		math::vec4f* image,
		const float curvature,
		const float scale
	) {
		const int x = id % width;
		const int y = id / width;

		// Define the orange color
		const math::vec3f orange = math::vec3f(1.0f, 0.5f, 0.0f);
		float edgeValue = edgeData[y * width + x];
		
		edgeValue = edgeTransform(edgeValue, curvature, scale);

		image[y * width + x] = math::vec4f(utl::lerp<math::vec3f>(math::vec3f(image[y * width + x]), orange, edgeValue), 1.0f);
		//image[y * width + x] = math::vec4f(edgeValue, 1.0f);
	}

	void overlayEdgesOnImage(
		float*       d_edgeData,
		math::vec4f* d_image,
		int          width,
		const int    height,
		const float  curvature,
		const float  scale
	)
	{
		const int dataSize = width * height;
		gpuParallelFor("OverlayEdges",
			dataSize,
			[d_edgeData, d_image, width, curvature, scale] __device__(const int id)
		{
			overlayEdgeKernel(id, width, d_edgeData, d_image, curvature, scale);
		});
	}

	__forceinline__ __device__ bool belongs(
		const float x,
		const float* values,
		const int numValues
	)
	{
		for (int i = 0; i < numValues; i++)
		{
			if (gdt::isEqual(x, values[i]))
			{
				return true;
			}
		}
		return false;
	}

	__forceinline__ __device__ void selectionEdgeDevice(
		const int id,
		const int width,
		const int height,
		const float* data,
		float* output,
		const float* values,
		const int nValues,
		const int thickness
	)
	{
		const int x = id % width;
		const int y = id / width;

		int matchCount = 0;      // Count of neighboring pixels matching the target value.
		int nonMatchCount = 0;   // Count of neighboring pixels not matching the target value.
		int totalNeighbors = 0;  // Total neighboring pixels within the thickness.

		if(belongs(data[id], values, nValues))
		{
			output[id] = 0.0f;
		}
		for (int dx = -thickness; dx <= thickness; dx++)
		{
			for (int dy = -thickness; dy <= thickness; dy++)
			{
				// Skip the center pixel.
				//if (dx == 0 && dy == 0) continue;

				const int nx = x + dx;
				const int ny = y + dy;
				if (nx >= 0 && nx < width && ny >= 0 && ny < height)
				{
					const float neighborValue = data[ny * width + nx];

					if (belongs(neighborValue, values, nValues))
					{
						matchCount++;
					}
					else
					{
						nonMatchCount++;
					}

					totalNeighbors++;
				}
			}
		}

		// Compute the edge value based on the ratio of match and non-match counts.
		if (matchCount == nonMatchCount)
		{
			output[id] = 1.0f;
		}
		else
		{
			float ratio = static_cast<float>(abs(matchCount - nonMatchCount)) / static_cast<float>(totalNeighbors);
			output[id] = 1.0f - ratio;
		}
	}

	void overlaySelectionEdge(
		float* gBuffer,
		math::vec4f* outputImage,
		int    width,
		int    height,
		std::vector<float> selectedIds,
		const float curvature,
		const float scale,
		CUDABuffer* edgeMapBuffer = nullptr,
		CUDABuffer* valuesBuffer = nullptr
	)
	{
		const int  dataSize = width * height;

		bool deleteEdgeMapBuffer = false;
		bool deleteValuesBuffer = false;

		// Create Buffers if not provided
		{
			if (edgeMapBuffer == nullptr)
			{
				edgeMapBuffer = new CUDABuffer();
				deleteEdgeMapBuffer = true;
			}

			if (valuesBuffer == nullptr)
			{
				valuesBuffer = new CUDABuffer();
				deleteValuesBuffer = true;
			}
		}

		auto* edgeMap = edgeMapBuffer->alloc<float>(dataSize);
		float* values = valuesBuffer->upload(selectedIds);
		int nValues = selectedIds.size();

		gpuParallelFor("Overlay Selection Edge",
			dataSize,
			[edgeMap, width, height, gBuffer, values, nValues, outputImage, curvature, scale] __device__(const int id)
			{
				selectionEdgeDevice(id, width, height, gBuffer, edgeMap, values, nValues, 1);
				overlayEdgeKernel(id, width, edgeMap, outputImage, curvature, scale);
			}
		);

		// Free Buffers if created
		{
			if (deleteEdgeMapBuffer)
			{
				edgeMapBuffer->free();
				delete edgeMapBuffer;
			}
			if (deleteValuesBuffer)
			{
				valuesBuffer->free();
				delete valuesBuffer;
			}
		}
	}
}
