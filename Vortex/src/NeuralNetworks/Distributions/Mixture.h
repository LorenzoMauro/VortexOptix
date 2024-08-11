#pragma once
#ifndef MIXTURE_H
#define MIXTURE_H

#ifndef CUDA_INTERFACE
#include <torch/torch.h>
#else
#include "Core/Math.h"
#include "Device/DevicePrograms/Utils.h"
#endif
#include "cuda_runtime.h"
#include "Nasg.h"
#include "SphericalGaussian.h"
#include "NeuralNetworks/Config/DistributionConfig.h"

namespace vtx::distribution
{
	class Mixture
	{
	public:
		#ifndef CUDA_INTERFACE
		static torch::Tensor finalizeParams(torch::Tensor& mixtureParameters, const network::config::DistributionType& type);

		static torch::Tensor prob(const torch::Tensor& x, const torch::Tensor& mixtureParams, const torch::Tensor& mixtureWeights, network::config::DistributionType type);

		static std::tuple<torch::Tensor, torch::Tensor> sample(const torch::Tensor& mixtureParams, const torch::Tensor& mixtureWeights, network::config::DistributionType type);

		static void setGraphData(
			network::config::DistributionType type,
			const torch::Tensor&              params,
			const torch::Tensor&              mixtureWeights,
			network::GraphsData&              graphData,
			const bool                        isTraining,
			const int                         depth = 0);

		#else
		__forceinline__ __device__ static float evaluate(const float* mixtureParameters, const float* weights, const int mixtureSize, network::config::DistributionType type, const math::vec3f& sample)
		{
			float prob = 0.0f;
			int parameterCount = getDistributionParametersCount(type);
			if (type == network::config::D_SPHERICAL_GAUSSIAN)
			{
				for (int i = 0; i < mixtureSize; ++i)
				{
					const float* params = mixtureParameters + i * parameterCount;
					prob += weights[i] * SphericalGaussian::prob(params, sample);
				}
			}
			else if (type == network::config::D_NASG_TRIG || type == network::config::D_NASG_ANGLE || type == network::config::D_NASG_AXIS_ANGLE || type == network::config::D_NASG_TRIG_NORMALIZED)
			{
				for (int i = 0; i < mixtureSize; ++i)
				{
					const float* params = mixtureParameters + i * parameterCount;
					prob += weights[i] * Nasg::prob(params, sample);
				}
			}

			return prob;
		}

		__forceinline__ __device__ static math::vec3f getAvarageAxis(const float* mixtureParameters, const float* weights, const int mixtureSize, network::config::DistributionType type)
		{
			math::vec3f meanAxis = math::vec3f(0.0f);
			int parameterCount = getDistributionParametersCount(type);
			if (type == network::config::D_SPHERICAL_GAUSSIAN)
			{
				for (int i = 0; i < mixtureSize; ++i)
				{
					const float* params = mixtureParameters + i * parameterCount;
					const auto& mean = reinterpret_cast<const math::vec3f&>(params[0]);
					meanAxis += weights[i] * mean;
				}
			}
			else if (type == network::config::D_NASG_TRIG || type == network::config::D_NASG_ANGLE || type == network::config::D_NASG_AXIS_ANGLE || type == network::config::D_NASG_TRIG_NORMALIZED)
			{
				for (int i = 0; i < mixtureSize; ++i)
				{
					const float* params = mixtureParameters + i * parameterCount;
					const math::vec3f& zAxis = (reinterpret_cast<const math::vec3f&>(params[3]));
					meanAxis += weights[i] * zAxis;
				}
			}
			return meanAxis;
		}

		__forceinline__ __device__ static float getAvarageConcentration(const float* mixtureParameters, const float* weights, const int mixtureSize, network::config::DistributionType type)
		{
			float meanConcentration = 0.0f;
			int parameterCount = getDistributionParametersCount(type);
			if (type == network::config::D_SPHERICAL_GAUSSIAN)
			{
				for (int i = 0; i < mixtureSize; ++i)
				{
					const float* params = mixtureParameters + i * parameterCount;
					const float& k = params[3];
					meanConcentration += weights[i] * k;
				}
			}
			else if (type == network::config::D_NASG_TRIG || type == network::config::D_NASG_ANGLE || type == network::config::D_NASG_AXIS_ANGLE || type == network::config::D_NASG_TRIG_NORMALIZED)
			{
				for (int i = 0; i < mixtureSize; ++i)
				{
					const float* params = mixtureParameters + i * parameterCount;
					const float lambda = params[6];
					meanConcentration += weights[i] * lambda;
				}
			}
			return meanConcentration;
		}

		__forceinline__ __device__ static float getAverageAnisotropy(const float* mixtureParameters, const float* weights, const int mixtureSize, network::config::DistributionType type)
		{
			float     meanA          = 0.0f;
			const int parameterCount = getDistributionParametersCount(type);
			if (type == network::config::D_SPHERICAL_GAUSSIAN)
			{
				return 0.0f;
			}
			else if (type == network::config::D_NASG_TRIG || type == network::config::D_NASG_ANGLE || type == network::config::D_NASG_AXIS_ANGLE || type == network::config::D_NASG_TRIG_NORMALIZED)
			{
				for (int i = 0; i < mixtureSize; ++i)
				{
					const float* params = mixtureParameters + i * parameterCount;
					const float a = params[7];
					meanA += weights[i] * a;
				}
			}
			return meanA;
		}


		__forceinline__ __device__ static math::vec3f sample(const float* mixtureParameters, const float* weights, const int mixtureSize, const network::config::DistributionType& type, unsigned& seed)
		{
			math::vec3f sample = math::vec3f(0.0f);
			int parameterCount = getDistributionParametersCount(type);
			const int sampledDistributionIdx = utl::selectFromWeights(weights, mixtureSize, seed);
			const float* params = mixtureParameters + sampledDistributionIdx * parameterCount;
			if (type == network::config::D_SPHERICAL_GAUSSIAN)
			{
				sample = SphericalGaussian::sample(params, seed);
			}
			else if(type == network::config::D_NASG_TRIG || type == network::config::D_NASG_ANGLE || type == network::config::D_NASG_AXIS_ANGLE || type == network::config::D_NASG_TRIG_NORMALIZED)
			{
				sample = Nasg::sample(params, seed);
			}

			return sample;
		}

		#endif
		__forceinline__ __device__ __host__ static int getDistributionParametersCount(const network::config::DistributionType& type, const bool forNetwork = false)
		{
			if (type == network::config::D_SPHERICAL_GAUSSIAN)
			{
				return SphericalGaussian::getParametersCount(forNetwork);
			}
			if (type == network::config::D_NASG_TRIG || type == network::config::D_NASG_ANGLE || type == network::config::D_NASG_AXIS_ANGLE || type == network::config::D_NASG_TRIG_NORMALIZED)
			{
				return Nasg::getParametersCount(type, forNetwork);
			}
			return 0;
		}
	};
}


#endif
