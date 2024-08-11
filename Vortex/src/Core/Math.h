#pragma once
#ifndef H_MATH
#define H_MATH
#include "Math/vec.h"
#include "Math/AffineSpace.h"
#include <tuple>
#include <cuda_runtime.h>
#include "Constants.h"

namespace vtx::math
{


	template<typename T>
	__both__ T max(T a , T b)
	{
		return a > b ? a : b;
	}

	template<typename T>
	__both__ T min(T a, T b)
	{
		return a < b ? a : b;
	}

	using OneTy = gdt::OneTy;
	static OneTy Identity = gdt::one;

	template<typename T>
	T toRadians(T degree) {
		return degree * T(PI_180);
	}

	template<typename T>
	T toDegrees(T  radian) {
		return radian * _180_PI;
	}

	///////////////////////////////////////////////////////////////////////////
	///// Vector Definitions //////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	
	template<typename T, int N>
	using vec_t = gdt::vec_t<T, N>;

	template<typename T>
	using vec3a_t = gdt::vec3a_t<T>;
		
#define DEFINE_VEC_TYPES(T,t)		\
		using vec2##t = vec_t<T,2>;			\
		using vec3##t = vec_t<T,3>;			\
		using vec4##t = vec_t<T,4>;			\
		using vec3##t##a = vec3a_t<T>;		\
		
	DEFINE_VEC_TYPES(int8_t, c);
	DEFINE_VEC_TYPES(int16_t, s);
	DEFINE_VEC_TYPES(int32_t, i);
	DEFINE_VEC_TYPES(int64_t, l);
	DEFINE_VEC_TYPES(uint8_t, uc);
	DEFINE_VEC_TYPES(uint16_t, us);
	DEFINE_VEC_TYPES(uint32_t, ui);
	DEFINE_VEC_TYPES(uint64_t, ul);
	DEFINE_VEC_TYPES(float, f);
	DEFINE_VEC_TYPES(double, d);

	__inline__ __both__ vec3f normalize(const vec3f& v) {
		return gdt::normalize(v);
	}

	__inline__ __both__ vec2f normalize(const vec2f& v) {
		return gdt::normalize(v);
	}

	__inline__ __both__ float dot(const vec3f& a, const vec3f& b) {
		return gdt::dot(a,b);
	}

	__inline__ __both__ float dot(const vec2f& a, const vec2f& b) {
		return gdt::dot(a,b);
	}

	__inline__ __both__ float length(const vec3f& a) {
		return gdt::length(a);
	}

	__inline__ __both__ float length(const vec2f& a) {
		return gdt::length(a);
	}

	__inline__ __both__ float saturate(const float f)
	{
		return min(1.f, max(0.f, f));
	}

	__inline__ __both__ bool isNan(const vec3f& v)
	{
		return isnan(v.x) || isnan(v.y) || isnan(v.z);
	}

	__inline__ __both__ bool isZero(const vec3f& v)
	{
		return v == math::vec3f(0.0f);
	}

	__inline__ __both__ bool isInf(const vec3f& v)
	{
		return isinf(v.x) || isinf(v.y) || isinf(v.z);
	}

	static inline __both__ vec3f saturate(const vec3f& v) {
		return vec3f(saturate(v.x), saturate(v.y), saturate(v.z));
	}

	inline __both__ vec3f randomColor(const int i)
	{
		return gdt::randomColor(i);
	}

	inline __both__ vec3f exp(const vec3f& v)
	{
		math::vec3f result;
		result.x = expf(v.x);
		result.y = expf(v.y);
		result.z = expf(v.z);
		return result;
	}

	inline __both__ vec3f pow(const vec3f& v, const float p)
	{
		math::vec3f result;
		result.x = powf(v.x, p);
		result.y = powf(v.y, p);
		result.z = powf(v.z, p);
		return result;
	}

	inline __both__ vec2f exp(const vec2f& v)
	{
		math::vec2f result;
		result.x = expf(v.x);
		result.y = expf(v.y);
		return result;
	}

	inline __both__ vec2f pow(const vec2f& v, const float p)
	{
		math::vec2f result;
		result.x = powf(v.x, p);
		result.y = powf(v.y, p);
		return result;
	}

	inline __both__ vec2f abs(const vec2f& v)
	{
		math::vec2f result;
		result.x = fabs(v.x);
		result.y = fabs(v.y);
		return result;
	}
		
	///////////////////////////////////////////////////////////////////////////
		///// Linear Space Definitions ////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////

	template<typename T>
	using LinearSpace2 = gdt::LinearSpace2<T>;

	template<typename T>
	using LinearSpace3 = gdt::LinearSpace3<T>;

	/*! Shortcuts for common linear spaces. */
	using LinearSpace2f = LinearSpace2<vec2f>;
	using LinearSpace3f = LinearSpace3<vec3f>;
	using LinearSpace3fa = LinearSpace3<vec3fa>;

	using linear2f = LinearSpace2f;
	using linear3f = LinearSpace3f;


	///////////////////////////////////////////////////////////////////////////
	///// Quaternion Definitions //////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////

	template<typename T>
	using QuaternionT = gdt::QuaternionT<T>;

	/*! default template instantiations */
	typedef QuaternionT<float>  Quaternion3f;
	typedef QuaternionT<double> Quaternion3d;

	///////////////////////////////////////////////////////////////////////////
	///// Affine Space Definition	///////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////

	/// NB : AffineSpace to float[12] assignment performed added in gdt/AffineSpace.h


	struct rowLinearSpace
	{
		float4 l[3];
	};

#define Scalar_t typename T::vector_t::scalar_t

	template<typename T>
	struct AffineSpaceT : public gdt::AffineSpaceT<T> {

		using gdt::AffineSpaceT<T>::AffineSpaceT; // Inherit all constructors

		inline __host__ __device__ void toFloat(float* out) const
		{
			out[0] = (float)l.vx.x; out[1] = (float)l.vy.x; out[2] = (float)l.vz.x; out[3] = (float)p.x;
			out[4] = (float)l.vx.y; out[5] = (float)l.vy.y; out[6] = (float)l.vz.y; out[7] = (float)p.y;
			out[8] = (float)l.vx.z; out[9] = (float)l.vy.z; out[10] = (float)l.vz.z; out[11] = (float)p.z;
		}

		inline __host__ __device__ void toFloat4(float4* out) const {
			out[0].x = (float)l.vx.x; out[0].y = (float)l.vx.y; out[0].z = (float)l.vx.z; out[0].w = p.x;
			out[1].x = (float)l.vy.x; out[1].y = (float)l.vy.y; out[1].z = (float)l.vy.z; out[1].w = p.y;
			out[2].x = (float)l.vz.x; out[2].y = (float)l.vz.y; out[2].z = (float)l.vz.z; out[2].w = p.z;
		}

		//Constructor from Scalar_t[12] array in row major order
		__both__ AffineSpaceT(const Scalar_t* rowMajor)
		{
			l.vx.x = rowMajor[0]; l.vy.x = rowMajor[1]; l.vz.x = rowMajor[2]; p.x = rowMajor[3];
			l.vx.y = rowMajor[4]; l.vy.y = rowMajor[5]; l.vz.y = rowMajor[6]; p.y = rowMajor[7];
			l.vx.z = rowMajor[8]; l.vy.z = rowMajor[9]; l.vz.z = rowMajor[10]; p.z = rowMajor[11];
		}
		//Constructor from float4 array in row major order
		__both__  AffineSpaceT(const float4* rowMajor)
		{
			l.vx.x = rowMajor[0].x;
			l.vx.y = rowMajor[1].x;
			l.vx.z = rowMajor[2].x;

			l.vy.x = rowMajor[0].y;
			l.vy.y = rowMajor[1].y;
			l.vy.z = rowMajor[2].y;

			l.vz.x = rowMajor[0].z;
			l.vz.y = rowMajor[1].z;
			l.vz.z = rowMajor[2].z;

			p.x = rowMajor[0].w;
			p.y = rowMajor[1].w;
			p.z = rowMajor[2].w;
		}
	};

	template<typename T>
	AffineSpaceT<T> AffineFromEuler(const vec_t<Scalar_t, 3>& euler) {
		// Extract the individual Euler angles for clarity
		Scalar_t yaw = euler.z;
		Scalar_t pitch = euler.y;
		Scalar_t roll = euler.x;

		// Precompute cosines and sines
		Scalar_t cy = cos(yaw);
		Scalar_t sy = sin(yaw);
		Scalar_t cp = cos(pitch);
		Scalar_t sp = sin(pitch);
		Scalar_t cr = cos(roll);
		Scalar_t sr = sin(roll);

		// Construct rotation matrix using the ZYX convention 
		T L;
		L.vx.x = cy * cp;
		L.vx.y = cy * sp * sr - sy * cr;
		L.vx.z = cy * sp * cr + sy * sr;

		L.vy.x = sy * cp;
		L.vy.y = sy * sp * sr + cy * cr;
		L.vy.z = sy * sp * cr - cy * sr;

		L.vz.x = -sp;
		L.vz.y = cp * sr;
		L.vz.z = cp * cr;

		return AffineSpaceT<T>(L);
	}

	template<typename T>
	void VectorFromAffine(AffineSpaceT<T> affine, vec_t<Scalar_t, 3>& translation, vec_t<Scalar_t, 3>& scale, vec_t<Scalar_t, 3>& euler) {
		translation = affine.p;

		scale.x = length(affine.l.vx);
		scale.y = length(affine.l.vy);
		scale.z = length(affine.l.vz);

		vec_t<Scalar_t, 3> reverseScale = 1.0f / scale;
		vec_t<Scalar_t, 3> rVx = affine.l.vx * reverseScale.x;
		vec_t<Scalar_t, 3> rVy = affine.l.vy * reverseScale.y;
		vec_t<Scalar_t, 3> rVz = affine.l.vz * reverseScale.z;

		Scalar_t m00 = rVx.x;
		Scalar_t m10 = rVx.y;
		Scalar_t m20 = rVx.z;

		Scalar_t m01 = rVy.x;
		Scalar_t m11 = rVy.y;
		Scalar_t m21 = rVy.z;

		Scalar_t m02 = rVz.x;
		Scalar_t m12 = rVz.y;
		Scalar_t m22 = rVz.z;

		constexpr float epsilon = 1e-6f; // Or choose another suitable small value

		// New Method
		{
			// Yaw
			if (fabsf(m20) < 1 - epsilon) {
				euler.z = atan2(-m10, m00);
				euler.x = atan2(-m21, m22);
			}
			else {
				// Gimbal lock case
				euler.z = atan2(m01, m11);
				euler.x = 0;
			}
			// Pitch
			euler.y = asin(m20);
		}
	}

#undef ScalarT

	using AffineSpace2f = AffineSpaceT<LinearSpace2f>;
	using AffineSpace3f = AffineSpaceT<LinearSpace3f>;
	using AffineSpace3fa = AffineSpaceT<LinearSpace3fa>;
	using OrthonormalSpace3f = AffineSpaceT<Quaternion3f>;

	using affine2f = AffineSpace2f;
	using affine3f = AffineSpace3f;

	__device__ __host__ static inline vec3f transformVector3F(const affine3f& affine, const vec3f& vec) {
		return gdt::xfmVector<LinearSpace3f>(affine, vec);
	}

	__device__ __host__ static inline vec3f transformPoint3F(const affine3f& affine, const vec3f& vec) {
		return gdt::xfmPoint<LinearSpace3f>(affine, vec);
	}

	__device__ __host__ static inline vec3f transformNormal3F(const affine3f& affine, const vec3f& vec) {
		return gdt::xfmNormal<LinearSpace3f>(affine, vec);
	}

	/////////////////////////////////////////////////////////////
		/////////////////// Some Utils Definition ///////////////////
		/////////////////////////////////////////////////////////////

	static vec3f xAxis = vec3f{ 1.0f, 0.0f, 0.0f };
	static vec3f yAxis = vec3f{ 0.0f, 1.0f, 0.0f };
	static vec3f zAxis = vec3f{ 0.0f, 0.0f, 1.0f };
	static vec3f origin = vec3f{ 0.0f, 0.0f, 1.0f };

}

#endif // !__GDT_MATH_H__