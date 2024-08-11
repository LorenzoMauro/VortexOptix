#pragma once
#ifndef NASG_H
#define NASG_H

#ifndef CUDA_INTERFACE
#include <torch/torch.h>
#include "NeuralNetworks/tools.h"
#include "TransformUtils.h"
#else
#include "cuda_runtime.h"
#include "Core/Math.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#include "TransformUtils.h"
#include "Device/DevicePrograms/Utils.h"
#endif
#include "NeuralNetworks/Config/DistributionConfig.h"

namespace vtx
{
	namespace network
	{
		struct GraphsData;
	}
}

namespace vtx::distribution
{
	class Nasg
	{
	public:
		__forceinline__ __device__ __host__ static int getParametersCount(const network::config::DistributionType dt, const bool forNetworkOutput = false)
		{
			int paramCount;
			if (forNetworkOutput)
			{
				if (dt == network::config::D_NASG_TRIG || dt == network::config::D_NASG_TRIG_NORMALIZED)
				{
					paramCount = 7;
				}
				else if (dt == network::config::D_NASG_ANGLE)
				{
					paramCount = 5;
				}
				else if (dt == network::config::D_NASG_AXIS_ANGLE)
				{
					paramCount = 6;
				}
			}
			else
			{
				paramCount = 8;
			}
			return paramCount;
		}
		#ifdef CUDA_INTERFACE

		__forceinline__ __device__ static void splitParameters(
			const math::vec3f*& xAxis,
			const math::vec3f*& zAxis,
			const float*& lambda,
			const float*& a,
			const float* distributionParams)
		{
			xAxis = reinterpret_cast<const math::vec3f*>(distributionParams);
			zAxis = reinterpret_cast<const math::vec3f*>(distributionParams + 3);
			lambda = distributionParams + 6;
			a = distributionParams + 7;
		}

		__forceinline__ __device__ static float normalizationFactor(
			const float& lambda,
			const float& a
		)
		{
			const float factor = 2.0f * M_PI * (1.0f - expf(-2.0f * lambda)) / (lambda * sqrtf(1.0f + a) + sEps);
			return factor;
		}

		__forceinline__ __device__ static float prob(
			const math::vec3f& v,
			const math::vec3f& xAxis,
			const math::vec3f& zAxis,
			const float& lambda,
			const float& a
		)
		{
			if(!(math::length(xAxis) > 0.0f && math::length(zAxis) > 0.0f && lambda > 0.0f && a > 0.0f))
			{
				return -1.0f;
			}
			
			float lambdaClamp = fmaxf(lambda, sEps);
			float aClamp = fmaxf(a, sEps);

			const float vDotZ = dot(v, zAxis);

			// From the paper, still, it's not quite implemented in Torch
			if (vDotZ <= -1.0f)
			{
				return 0.0f;
			}

			const float normalization = normalizationFactor(lambdaClamp, aClamp);
			if (vDotZ >= 1.0f)
			{
				return 1.0f / normalization;
			}

			const float vDotX = dot(v, xAxis);

			const float scaleValue = (vDotZ + 1.0f) / 2.0f;
			const float denominator = fmaxf(sEps, 1.0f - (vDotZ * vDotZ));

			const float powerValue = aClamp * vDotX * vDotX / denominator;

			const float g = expf(2.0f * lambdaClamp * (powf(scaleValue, 1 + powerValue) - 1.0f)) * powf(scaleValue, powerValue);

			const float result = g / normalization;

			if(isnan(result) || isinf(result))
			{
				return -1.0f;
			}
			return result;

		}

		__forceinline__ __device__ static float prob(const float* distributionParams, const math::vec3f& sample)
		{
			const math::vec3f* xAxis = nullptr;
			const math::vec3f* zAxis = nullptr;
			const float* lambda = nullptr;
			const float* a = nullptr;
			splitParameters(xAxis, zAxis, lambda, a, distributionParams);

			const float result = prob(
				sample,
				*xAxis,
				*zAxis,
				*lambda,
				*a
			);

			return result;
		}

		__forceinline__ __device__ static math::vec3f sample(
			const math::vec3f& xAxis,
			const math::vec3f& zAxis,
			const float& lambda,
			const float& a,
			unsigned& seed)
		{
			// Sample epsilons
			const float uniform0 = rng(seed);
			const float uniform1 = rng(seed);
			const float uniform2 = rng(seed);

			// Map epsilons
			// s is in the range [exp(-2*lambda), 1] we don't want it to be 0 so we add a small epsilon
			const float sLowerBound = expf(-2.0f * lambda);
			const float sUpperBound = 1.0f;
			const float sDelta = sUpperBound - sLowerBound;
			float s = sLowerBound + uniform0 * sDelta;
			s = fmaxf(s, sEps);

			// rho is in the range [-pi/2, pi/2]s
			const float rhoLowerBound = -(float)M_PI_2;
			const float rhoUpperBound = (float)M_PI_2;
			const float rhoDelta = rhoUpperBound - rhoLowerBound;
			const float rho = rhoLowerBound + uniform1 * rhoDelta;

			// Compute theta and phi based on sampling
			float phi = atanf(sqrtf(1.0f  + a) * tanf(rho));
			// East or West Hemisphere?
			phi = uniform2 > 0.5f ? phi : phi + (float)M_PI;

			// for the computation of theta I have introduced the fmaxf to avoid NaNs and also compute cosRho and multiply it by itself again, for stability issues
			float cosRho = cosf(rho);
			const float base = fmaxf(logf(s) / (2.0f * lambda) + 1.0f, sEps);
			const float exponent = (1.0f + a * (1.0f - cosRho * cosRho)) / (1.0f + a);
			const float theta = acosf(2.0f * powf(base, exponent) - 1.0f);

			// Theta is the polar angle, phi is the azimuthal angle
			const float sinTheta = sinf(theta);
			auto        vNonTransformed        = math::vec3f(
				sinTheta * cosf(phi),
				sinTheta * sinf(phi),
				cosf(theta)
			);

			const math::vec3f yAxis = cross(zAxis, xAxis);

			math::vec3f v = vNonTransformed.x * xAxis + vNonTransformed.y * yAxis + vNonTransformed.z * zAxis;
			
			return v;
		}

		__forceinline__ __device__ static math::vec3f sample(const float* distributionParams, unsigned& seed)
		{

			const math::vec3f* xAxis = nullptr;
			const math::vec3f* zAxis = nullptr;
			const float* lambda = nullptr;
			const float* a = nullptr;
			splitParameters(xAxis, zAxis, lambda, a, distributionParams);

			const math::vec3f result = sample(
				*xAxis,
				*zAxis,
				*lambda,
				*a,
				seed
			);

			return result;
		}


		#else

		static torch::Tensor normalizationFactor(
			const torch::Tensor& lambda,
			const torch::Tensor& a);


		static torch::Tensor finalizeRawParams(const torch::Tensor& rawParams, const network::config::DistributionType& type);

		static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Nasg::splitParams(const torch::Tensor& params);

		static torch::Tensor prob(
			const torch::Tensor& sample,
			const torch::Tensor& zAxis,
			const torch::Tensor& xAxis,
			const torch::Tensor& lambda,
			const torch::Tensor& a
		);

		static torch::Tensor sample(
			const torch::Tensor& transform,
			const torch::Tensor& lambda,
			const torch::Tensor& a
		);

		static torch::Tensor prob(
			const torch::Tensor& sample,
			const torch::Tensor& params
		);

		static torch::Tensor sample(
			const torch::Tensor& params
		);

		static void setGraphData(
			const torch::Tensor& params,
			const torch::Tensor& mixtureWeights,
			network::GraphsData& graphData,
			const bool           isTraining,
			const int            depth = 0
		);
		#endif
	};
}


#endif
