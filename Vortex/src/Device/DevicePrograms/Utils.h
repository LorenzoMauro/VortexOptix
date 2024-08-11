#pragma once
#ifndef UTILS_H
#define UTILS_H
#include "Core/Math.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#undef min
#undef max

namespace vtx::utl
{

	__forceinline__ __device__ bool isNan(const math::vec3f& vec)
	{
		return (isnan(vec.x) || isnan(vec.y) || isnan(vec.z));
	}

	__forceinline__ __device__ bool isInf(const math::vec3f& vec)
	{
		return (isinf(vec.x) || isinf(vec.y) || isinf(vec.z));
	}

	template<typename T>
	__device__ T lerp(const T& a, const T& b, const float& t)
	{
		return a * (1.0f - t) + b * t;
	}

	__forceinline__ __device__ float clampf(const float& value, const float& min, const float& max)
	{
		return fminf(fmaxf(value, min), max);
	}

	__forceinline__ __device__ float smoothstep(const float edge0, const float edge1, float x) {
		x = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
		return x * x * (3 - 2 * x);
	}

	__forceinline__ __device__ void accumulate(const math::vec3f& input, math::vec3f& buffer, const float kFactor, const float jFactor)
	{
		if(!isNan(input))
		{
			buffer = buffer * jFactor + input * kFactor;
		}
	}

	__forceinline__ __device__ void lerpAccumulate(const math::vec3f& input, math::vec3f& buffer, const float samples)
	{
		if(!isNan(input))
		{
			buffer = utl::lerp(buffer, input, samples);
		}
	}

	__forceinline__ __device__ void replaceNanCheck(const math::vec3f& input, math::vec3f& buffer)
	{
		if(!isNan(input))
		{
			buffer = input;
		}
	}



	__forceinline__ __host__ __device__ float luminance(const math::vec3f& rgb)
	{
		const math::vec3f ntscLuminance{ 0.30f, 0.59f, 0.11f };
		return dot(rgb, ntscLuminance);
	}

	__forceinline__ __host__ __device__ float intensity(const math::vec3f& rgb)
	{
		return (rgb.x + rgb.y + rgb.z) * 0.3333333333f;
	}

	template<typename T>
	__device__ T pow(const T& a, const T& b);


	template<>
	__forceinline__ __device__ float pow(const float& a, const float& b) {
		return powf(a, b);
	}


	template<>
	__forceinline__ __device__ math::vec3f pow(const math::vec3f& a, const math::vec3f& b)
	{
		return { pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z) };
	}


	template<typename T>
	__device__ T max(const T& a, const T& b);

	template<>
	__forceinline__ __device__ float max(const float& a, const float& b) {
		return a>b? a: b;
	}

	template<>
	__forceinline__ __device__ math::vec3f max(const math::vec3f& a, const math::vec3f& b)
	{
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}

	//-------------------------------------------------------------------------------------------------
		// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
		//-------------------------------------------------------------------------------------------------
	__forceinline__ __device__ float intAsFloat(const uint32_t v)
	{
		union
		{
			uint32_t bit;
			float    value;
		} temp;

		temp.bit = v;
		return temp.value;
	}

	__forceinline__ __device__ uint32_t floatAsInt(const float v)
	{
		union
		{
			uint32_t bit;
			float    value;
		} temp;

		temp.value = v;
		return temp.bit;
	}

	__forceinline__ __device__ void offsetRay(math::vec3f& hitPosition, const math::vec3f& normal)
	{
		constexpr float origin = 1.0f / 32.0f;
		constexpr float floatScale = 1.0f / 65536.0f;
		constexpr float intScale = 256.0f;

		const math::vec3i ofI(
			static_cast<int>(intScale * normal.x),
			static_cast<int>(intScale * normal.y),
			static_cast<int>(intScale * normal.z));

		math::vec3f pI(
			intAsFloat(floatAsInt(hitPosition.x) + ((hitPosition.x < 0.0f) ? -ofI.x : ofI.x)),
			intAsFloat(floatAsInt(hitPosition.y) + ((hitPosition.y < 0.0f) ? -ofI.y : ofI.y)),
			intAsFloat(floatAsInt(hitPosition.z) + ((hitPosition.z < 0.0f) ? -ofI.z : ofI.z)));

		hitPosition.x = abs(hitPosition.x) < origin ? hitPosition.x + floatScale * normal.x : pI.x;
		hitPosition.y = abs(hitPosition.y) < origin ? hitPosition.y + floatScale * normal.y : pI.y;
		hitPosition.z = abs(hitPosition.z) < origin ? hitPosition.z + floatScale * normal.z : pI.z;
	}


	__forceinline__ __device__ uint32_t makeColor(const math::vec3f& radiance)
	{
		const auto     r = static_cast<uint8_t>(radiance.x * 255.0f);
		const auto     g = static_cast<uint8_t>(radiance.y * 255.0f);
		const auto     b = static_cast<uint8_t>(radiance.z * 255.0f);
		const auto     a = static_cast<uint8_t>(1.0f * 255.0f);
		const uint32_t returnColor = (a << 24) | (b << 16) | (g << 8) | r;
		return returnColor;
	}

	template<typename T>
	__forceinline__ __host__ __device__ bool isNull(const T& v)
	{
		return v==0;
	}

	template<typename T>
	__forceinline__ __host__ __device__ bool isNull(const gdt::vec_t<T,3>& v)
	{
		return (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f);
	}

	__forceinline__ __host__ __device__ float balanceHeuristic(const float a, const float b)
	{
		//return __fdiv_rn(a,__fadd_rn(a,b));
		return a / (a + b);
	}

	__forceinline__ __host__ __device__ float powerHeuristic(const float a, const float b)
	{
		//return __fdiv_rn(a,__fadd_rn(a,b));
		return a*a / (a*a + b*b);
	}

	__forceinline__ __host__ __device__ float heuristic(const float a, const float b)
	{
		//return __fdiv_rn(a,__fadd_rn(a,b));
		return balanceHeuristic(a, b);
	}

	// Binary-search and return the highest cell index with CDF value <= sample.
	// Arguments are the CDF values array pointer, the index of the last element and the random sample in the range [0.0f, 1.0f).
	__forceinline__ __device__ unsigned int binarySearchCdf(const float* cdf, const unsigned int last, const float sample)
	{
		unsigned int lowerBound = 0;
		unsigned int upperBound = last; // Index on the last entry containing 1.0f. Can never be reached with the sample in the range [0.0f, 1.0f).

		while (lowerBound + 1 != upperBound) // When a pair of limits have been found, the lower index indicates the cell to use.
		{
			const unsigned int index = (lowerBound + upperBound) >> 1;

			if (sample < cdf[index]) // If the CDF value is greater than the sample, use that as new higher limit.
			{
				upperBound = index;
			}
			else // If the sample is greater than or equal to the CDF value, use that as new lower limit.
			{
				lowerBound = index;
			}
		}

		return lowerBound;
	}

	__forceinline__ __device__ unsigned int selectFromWeights(const float* weights, const unsigned int numberOfWeights, unsigned& seed)
	{
		const float sample = rng(seed);

		float runningSum = 0.0f;

		for (unsigned int i = 0; i < numberOfWeights; ++i)
		{
			runningSum += weights[i];
			if (sample <= runningSum)
				return i;
		}

		return numberOfWeights - 1; // This should ideally not happen if weights are normalized and the sample is in [0.0f, 1.0f), but acts as a safeguard.
	}
}

#endif
