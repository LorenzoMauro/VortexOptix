#pragma once
#ifndef SPHERICAL_GAUSSIAN_H
#define SPHERICAL_GAUSSIAN_H

#ifndef CUDA_INTERFACE
#include <torch/torch.h>
#else
#include "Core/Math.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#endif
#include "cuda_runtime.h"

namespace vtx
{
	namespace network
	{
		struct GraphsData;
	}
}

namespace vtx::distribution
{
    class SphericalGaussian {
    public:
#ifndef CUDA_INTERFACE
        static torch::Tensor sample(const torch::Tensor& loc, const torch::Tensor& scale);

        static torch::Tensor prob(const torch::Tensor& x, const torch::Tensor& loc, const torch::Tensor& scale);

        static torch::Tensor logLikelihood(const torch::Tensor& x, const torch::Tensor& loc, const torch::Tensor& scale);

        static std::tuple<torch::Tensor, torch::Tensor> splitParams(const torch::Tensor& params);

    	static torch::Tensor prob(const torch::Tensor& x, const torch::Tensor& params);

        static torch::Tensor logLikelihood(const torch::Tensor& x, const torch::Tensor& params);

        static torch::Tensor sample(const torch::Tensor& params);

        static torch::Tensor finalizeRawParams(const torch::Tensor& params);

        static void setGraphData(const torch::Tensor& params, const torch::Tensor& mixtureWeights, network::GraphsData& graphData, const bool isTraining, const
								 int depth);
#else
    	__forceinline__ __device__ static float prob(const math::vec3f& mean, const float& k, const math::vec3f& action)
        {
            const float pdf = k / (2.0f * M_PI * (1 - expf(-2.0f * k))) * expf(k * (dot(mean, action) - 1.0f));

            if (isnan(pdf) || isinf(pdf))
            {
                return -1.0f;
            }
            return pdf;
        }

        __forceinline__ __device__ static math::vec3f sample(const math::vec3f& mean, const float& k, unsigned& seed)
        {
            const float uniform = rng(seed);
            const float w = 1.0f + logf(uniform + (1.0f - uniform) * expf(-2.0f * k) + sEps) / (k + sEps);

            const float angleUniform = rng(seed) * 2.0f * M_PI;
            const math::vec2f v = math::vec2f(cosf(angleUniform), sinf(angleUniform));

            float      w_ = sqrtf(math::max(0.0f, 1.0f - w * w));
            const auto x = math::vec3f(w, w_ * v.x, w_ * v.y);

            const auto  e1 = math::vec3f(1.0f, 0.0f, 0.0f);
            math::vec3f u = e1 - mean;
            u = math::normalize(u);
            math::vec3f sample = x - 2.0f * math::dot(x, u) * u;

            return sample;
        }

        __forceinline__ __device__ static void splitPrams(const math::vec3f*& mean, const float*& k, const float* params)
    	{
    		mean = reinterpret_cast<const math::vec3f*>(params);
			k =params+3;
		}
        __forceinline__ __device__ static math::vec3f sample(const float* parameters, unsigned& seed)
        {
            const math::vec3f* mean = nullptr;
    		const float* k = nullptr;
            splitPrams(mean, k, parameters);
            const math::vec3f  sample = SphericalGaussian::sample(*mean, *k, seed);
            return sample;
        }

        __forceinline__ __device__ static float prob(const float* parameters, const math::vec3f& action)
        {
            const math::vec3f* mean = nullptr;
            const float* k = nullptr;
            splitPrams(mean, k, parameters);
			const float pdf = SphericalGaussian::prob(*mean, *k, action);
			return pdf;
		}

#endif
        __forceinline__ __device__ __host__ static int getParametersCount(bool forNetwork)
        {
            if (forNetwork)
            {
	            return 3;
			}
            return 4;
        }
    };
}

#endif