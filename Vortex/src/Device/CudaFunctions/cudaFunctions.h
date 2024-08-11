#pragma once
#include "ErrorTypes.h"
#include "Core/Math.h"

namespace vtx::cuda
{
	float computeMape(math::vec3f* groundTruth, math::vec3f* input, const int& width, const int& height);

	float computeMse(math::vec3f* groundTruth, math::vec3f* input, const int& width, const int& height);

	Errors computeErrors(const CUDABuffer& reference, const CUDABuffer& input, CUDABuffer& errorMaps, const int& width, const int& height);

	void copyRGBtoRGBA(const CUDABuffer& src, CUDABuffer&  dst, const int& width, const int& height);

	void copyRtoRGBA(const CUDABuffer& src, CUDABuffer&  dst, const int& width, const int& height);


	void overlayEdgesOnImage(
		float* d_edgeData,
		math::vec4f* d_image,
		int width,
		int height,
		const float curvature,
		const float scale
	);

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
	);

	float printDistribution(
		CUDABuffer& buffer,
		const int width,
		const int height,
		const math::vec3f& normal,
		const math::vec3f& sample
	);

	void accumulateAtDebugBounce(
		CUDABuffer& buffer,
		const int width,
		const int height,
		const int pixel);
}
