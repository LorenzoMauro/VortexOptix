#pragma once
#include <map>

namespace vtx
{
	struct AdaptiveSamplingSettings
	{
		bool  active;
		int   noiseKernelSize;
		int   minAdaptiveSamples;
		int   minPixelSamples;
		int   maxPixelSamples;
		float albedoNormalNoiseInfluence;
		float noiseCutOff;
		bool  isUpdated = true;
	};

	struct FireflySettings
	{
		bool  active;
		int   kernelSize;
		float threshold;
		int   start;
		bool  isUpdated = true;
	};

	struct DenoiserSettings
	{
		bool  active;
		int   denoiserStart;
		float denoiserBlend;
		bool  isUpdated = true;
	};

	struct WavefrontSettings
	{
		bool  active;
		bool  fitWavefront;
		bool  optixShade;
		bool  parallelShade;
		float longPathPercentage;
		bool  useLongPathKernel;
		bool  isUpdated = true;
	};

	struct ToneMapperSettings
	{
		math::vec3f whitePoint     = {1.0f, 1.0f, 1.0f};
		math::vec3f invWhitePoint  = {1.0f, 1.0f, 1.0f};
		math::vec3f colorBalance   = {1.0f, 1.0f, 1.0f};
		float       burnHighlights = 0.0f;
		float       crushBlacks    = 1.0f;
		float       saturation     = 1.0f;
		float       gamma          = 2.2f;
		float       invGamma       = 1.0f / 2.2f;
		bool        isUpdated      = true;

		__both__ __forceinline__ void print()
		{
			printf(
				   "whitePoint: %f %f %f\n"
				   "invWhitePoint: %f %f %f\n"
				   "colorBalance: %f %f %f\n"
				   "burnHighlights: %f\n"
				   "crushBlacks: %f\n"
				   "saturation: %f\n"
				   "gamma: %f\n"
				   "invGamma: %f\n"
				   ,
				   whitePoint.x, whitePoint.y, whitePoint.z,
				   invWhitePoint.x, invWhitePoint.y, invWhitePoint.z,
				   colorBalance.x, colorBalance.y, colorBalance.z,
				   burnHighlights,
				   crushBlacks,
				   saturation,
				   gamma,
				   invGamma
				  );
		}
	};

	enum DisplayBuffer
	{
		FB_BEAUTY,
		FB_NOISY,

		FB_DIFFUSE,
		FB_ORIENTATION,
		FB_TRUE_NORMAL,
		FB_SHADING_NORMAL,
		FB_TANGENT,
		FB_UV,
		FB_NOISE,
		FB_GBUFFER,
		FB_SAMPLES,
		FB_DEBUG_1,

		FB_NETWORK_INFERENCE_STATE_POSITION,
		FB_NETWORK_INFERENCE_STATE_NORMAL,
		FB_NETWORK_INFERENCE_OUTGOING_DIRECTION,
		FB_NETWORK_INFERENCE_CONCENTRATION,
		FB_NETWORK_INFERENCE_ANISOTROPY,
		FB_NETWORK_INFERENCE_MEAN,
		FB_NETWORK_INFERENCE_SAMPLE,
		FB_NETWORK_INFERENCE_SAMPLE_DEBUG,
		FB_NETWORK_INFERENCE_PDF,
		FB_NETWORK_INFERENCE_IS_FRONT_FACE,
		FB_NETWORK_INFERENCE_SAMPLING_FRACTION,

		FB_NETWORK_REPLAY_BUFFER_REWARD,
		FB_NETWORK_REPLAY_BUFFER_SAMPLES,
		FB_NETWORK_DEBUG_PATHS,

		FB_COUNT,
	};

	inline static const char* displayBufferNames[] = {
		"Beauty",
		"Noisy",
		"Diffuse",
		"Orientation",
		"True Normal",
		"Shading Normal",
		"Tangent",
		"Uv",
		"Noise",
		"GBuffer",
		"Samples",
		"Debug1",
		"Network Inference State Position",
		"Network Inference State Normal",
		"Network Inference Outgoing Direction",
		"Network Inference Concentration",
		"Network Inference Anisotropy",
		"Network Inference Mean",
		"Network Inference Sample",
		"Network Inference Sample Debug",
		"Network Inference Pdf",
		"Network Inference Is Front Face",
		"Network Inference Sampling Fraction",
		"Network Replay Buffer Reward",
		"Network Replay Buffer Samples",
		"Network Debug Paths",
		"Count"
	};

	inline static std::map<std::string, DisplayBuffer> displayBufferNameToEnum =
	{
		{"Beauty", FB_BEAUTY},
		{"Noisy", FB_NOISY},
		{"Diffuse", FB_DIFFUSE},
		{"Orientation", FB_ORIENTATION},
		{"True Normal", FB_TRUE_NORMAL},
		{"Shading Normal", FB_SHADING_NORMAL},
		{"Tangent", FB_TANGENT},
		{"Uv", FB_UV},
		{"Noise", FB_NOISE},
		{"GBuffer", FB_GBUFFER},
		{"Samples", FB_SAMPLES},
		{"Debug1", FB_DEBUG_1},
		{"Network Inference State Position", FB_NETWORK_INFERENCE_STATE_POSITION},
		{"Network Inference State Normal", FB_NETWORK_INFERENCE_STATE_NORMAL},
		{"Network Inference Outgoing Direction", FB_NETWORK_INFERENCE_OUTGOING_DIRECTION},
		{"Network Inference Concentration", FB_NETWORK_INFERENCE_CONCENTRATION},
		{"Network Inference Anisotropy", FB_NETWORK_INFERENCE_ANISOTROPY},
		{"Network Inference Mean", FB_NETWORK_INFERENCE_MEAN},
		{"Network Inference Sample", FB_NETWORK_INFERENCE_SAMPLE},
		{"Network Inference Sample Debug", FB_NETWORK_INFERENCE_SAMPLE_DEBUG},
		{"Network Inference Pdf", FB_NETWORK_INFERENCE_PDF},
		{"Network Inference Is Front Face", FB_NETWORK_INFERENCE_IS_FRONT_FACE},
		{"Network Inference Sampling Fraction", FB_NETWORK_INFERENCE_SAMPLING_FRACTION},
		{"Network Replay Buffer Reward", FB_NETWORK_REPLAY_BUFFER_REWARD},
		{"Network Replay Buffer Samples", FB_NETWORK_REPLAY_BUFFER_SAMPLES},
		{"Network Debug Paths", FB_NETWORK_DEBUG_PATHS},
		{"Count", FB_COUNT}
	};

	enum SamplingTechnique
	{
		S_BSDF,
		S_DIRECT_LIGHT,
		S_MIS,

		S_COUNT
	};

	inline static const char* samplingTechniqueNames[] = {
		"Bsdf Sampling",
		"Light Sampling",
		"Multiple Importance Sampling",
	};

	inline static std::map<std::string, SamplingTechnique> samplingTechniqueNameToEnum =
	{
		{"Bsdf Sampling", S_BSDF},
		{"Light Sampling", S_DIRECT_LIGHT},
		{"Multiple Importance Sampling", S_MIS}
	};


	enum Quadrant
	{
		Q_TOP_LEFT,
		Q_TOP_RIGHT,
		Q_BOTTOM_LEFT,
		Q_BOTTOM_RIGHT
	};

	struct QuadrantTechniqueSplit
	{
		SamplingTechnique st              = S_MIS;
		bool              neuralActivated = false;
	};

	struct QuadrantsSettings
	{
		bool                   isActivated = false;
		QuadrantTechniqueSplit topLeft;
		QuadrantTechniqueSplit topRight;
		QuadrantTechniqueSplit bottomLeft;
		QuadrantTechniqueSplit bottomRight;
		bool                   isUpdated = false;
	};

	struct RendererSettings
	{
		int                      iteration;
		int                      maxBounces;
		int                      maxSamples;
		bool                     accumulate;
		SamplingTechnique        samplingTechnique;
		DisplayBuffer            displayBuffer;
		float                    minClamp;
		float                    maxClamp;
		bool                     useRussianRoulette;
		bool                     runOnSeparateThread;
		bool                     isUpdated = true;
		AdaptiveSamplingSettings adaptiveSamplingSettings;
		FireflySettings          fireflySettings;
		DenoiserSettings         denoiserSettings;
		ToneMapperSettings       toneMapperSettings;
		QuadrantsSettings        quadrantsSettings;
		int                      debugPixel      = -1;
		int                      debugDepth      = -1;
		float                    tangentRotation = 0.0f;
		bool                     viewBackground  = true;

		void resetUpdate()
		{
			isUpdated                          = false;
			adaptiveSamplingSettings.isUpdated = false;
			fireflySettings.isUpdated          = false;
			denoiserSettings.isUpdated         = false;
			toneMapperSettings.isUpdated       = false;
			quadrantsSettings.isUpdated        = false;
		}

		bool isAnyUpdated()
		{
			return isUpdated || adaptiveSamplingSettings.isUpdated || fireflySettings.isUpdated || denoiserSettings.isUpdated || toneMapperSettings.isUpdated || quadrantsSettings.isUpdated;
		}
	};
}
