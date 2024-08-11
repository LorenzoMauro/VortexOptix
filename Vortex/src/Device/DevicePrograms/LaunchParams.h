#ifndef LAUNCH_PARAMS_H
#define LAUNCH_PARAMS_H
#pragma once

#include "Core/Math.h"
#include "Core/VortexID.h"
#include "cuda.h"
#include <mi/neuraylib/target_code_types.h>
#include "NoiseData.h"
#include "Device/Structs/GeometryData.h"
#include "Device/Wrappers/WorkQueue.h"
#include "NeuralNetworks/Config/NetworkSettings.h"
#include "Scene/DataStructs/VertexAttribute.h"
#include "Scene/Nodes/LightTypes.h"
#include "Scene/Nodes/RendererSettings.h"

namespace vtx {
	struct DebugData;
	struct NetworkInterface;
	struct AccumulationWorkItem;
	struct EscapedWorkItem;
	struct ShadowWorkItem;
	struct RayWorkItem;
	struct TraceWorkItem;
	struct RayData;

    struct SbtProgramIdx
    {
        int raygen          = -1;
        int exception       = -1;
        int miss            = -1;
        int hit             = -1;
        int pinhole         = -1;
        int meshLightSample = -1;
		int envLightSample = -1;
	};

    // Camera
    struct CameraData {
        math::vec3f  position;
        math::vec3f  vertical;
        math::vec3f  horizontal;
        math::vec3f  direction;
    };



    struct DeviceShaderConfiguration
    {
        bool isEmissive = false;
        bool isThinWalled = true;
        bool hasOpacity = false;
        bool directCallable;

        int idxCallEvaluateMaterialStandard = -1;
        int idxCallEvaluateMaterialWavefront = -1;
        int idxCallEvaluateMaterialWavefrontCuda = -1;

        int idxCallInit = -1; // The material global init function.

        int idxCallThinWalled = -1;

        int idxCallSurfaceScatteringSample = -1;
        int idxCallSurfaceScatteringEval = -1;
        int idxCallSurfaceScatteringAuxiliary = -1;

        int idxCallBackfaceScatteringSample = -1;
        int idxCallBackfaceScatteringEval = -1;
        int idxCallBackfaceScatteringAuxiliary = -1;

        int idxCallSurfaceEmissionEval = -1;
        int idxCallSurfaceEmissionIntensity = -1;
        int idxCallSurfaceEmissionIntensityMode = -1;

        int idxCallBackfaceEmissionEval = -1;
        int idxCallBackfaceEmissionIntensity = -1;
        int idxCallBackfaceEmissionIntensityMode = -1;

        int idxCallIor = -1;

        int idxCallVolumeAbsorptionCoefficient = -1;
        int idxCallVolumeScatteringCoefficient = -1;
        int idxCallVolumeDirectionalBias = -1;

        int idxCallGeometryCutoutOpacity = -1;

        int idxCallHairSample = -1;
        int idxCallHairEval = -1;

        // The constant expression values:
        //bool thin_walled; // Stored inside flags.
        math::vec3f surfaceIntensity;
        int         surfaceIntensityMode;
        math::vec3f backfaceIntensity;
        int         backfaceIntensityMode;
        math::vec3f ior;
        math::vec3f absorptionCoefficient;
        math::vec3f scatteringCoefficient;
        float       directionalBias;
        float       cutoutOpacity;
	};

    struct TextureData
    {
        cudaTextureObject_t     texObj;
        cudaTextureObject_t     texObjUnfiltered;
        math::vec4ui            dimension;
        math::vec3f             invSize;

    };

    struct BsdfSamplingPartData
    {
        math::vec2ui    angularResolution;
        math::vec2f     invAngularResolution;
        int             numChannels;
        float*          sampleData;
        float*          albedoData;
        CUtexObject     evalData;
        CUarray         deviceMbsdfData;
        float           maxAlbedo;
    };

    struct BsdfData
    {
        BsdfSamplingPartData* reflectionBsdf = nullptr;
        BsdfSamplingPartData* transmissionBsdf = nullptr;
        bool hasReflectionBsdf = false;
        bool hasTransmissionBsdf = false;
    };

    struct LightProfileData
    {
        CUarray             lightProfileArray;
        CUtexObject         evalData;
        float*              cdfData;
        math::vec2ui        angularResolution;     // angular resolution of the grid
        math::vec2f         invAngularResolution; // inverse angular resolution of the grid
        math::vec2f         thetaPhiStart;        // start of the grid
        math::vec2f         thetaPhiDelta;        // angular step size
        math::vec2f         thetaPhiInvDelta;    // inverse step size
        // 4 byte aligned
        float               candelaMultiplier;     // factor to rescale the normalized data
        float               totalPower;
    };

    struct TextureHandler : mi::neuraylib::Texture_handler_base
    {
        // All indexed by the mdl indices
        TextureData** textures;
        uint32_t numTextures;
        BsdfData** bsdfs;
        uint32_t numBsdfs;
        LightProfileData** lightProfiles;
        uint32_t numLightProfiles;
    };

    struct MaterialData
    {
        int                         materialWorkQueue;
        char*                       argBlock;
        DeviceShaderConfiguration*  materialConfiguration;
        TextureHandler*             textureHandler;
    };

    struct AliasData {
        unsigned int alias;
        float q;
        float pdf;
    };

    struct EnvLightAttributesData
    {
        TextureData* texture;
        //float  invIntegral;
        //float* cdfU;
        //float* cdfV;
        AliasData* aliasMap;
        math::affine3f    transformation;
        math::affine3f    invTransformation;
        float scaleLuminosity;
    };

    struct MeshLightAttributesData
    {
        vtxID           instanceId; //To retrieve the transform
        GeometryData*   geometryData;
        MaterialData*   materialId;

        float* cdfArea;
        uint32_t* actualTriangleIndices;
        float           totalArea;
        int             size;
    };

    struct LightData
    {
        LightType   type;
        CUdeviceptr attributes;
        bool use = false;
    };

    struct InstanceData
    {
        struct SlotIds
        {
            MaterialData*   material;
            LightData*      meshLight;
        };
        vtxID						instanceId;
        GeometryData*               geometryData;
        SlotIds*                    materialSlots;
        int                         numberOfSlots;
        math::affine3f              transform;
        bool                        hasEmission;
        bool                        hasOpacity;
    };

    struct gBufferHistory
    {
#define historySize 10

        struct IdData
        {
            vtxID id = 0;
            int count = 0;
        };
        vtxID mostFrequent = 0;
        IdData idCounts[historySize];
        int nSamples = 0;

        __forceinline__ __device__ void recordId(const vtxID newId)
        {
            if (nSamples == historySize)
            {
                return;
            }
            nSamples++;

            for (int i = 0; i < historySize; ++i)
            {
	            if (idCounts[i].id == newId)
	            {
	            	idCounts[i].count++;
                    break;
				}
                if (idCounts[i].count == 0)
                {
	                idCounts[i].id = newId;
					idCounts[i].count = 1;
                    break;
				}
			}

            if(mostFrequent == newId)
            {
                return;
            }

            int maxIndex = 0;
            for (int i = 0; i< historySize; ++i)
            {
	            if (idCounts[i].count > idCounts[maxIndex].count)
	            {
	            	maxIndex = i;
				}
			}
            mostFrequent = idCounts[maxIndex].id;
        }

        __forceinline__ __device__ void reset()
        {
	        for (int i = 0; i < historySize; ++i)
	        {
		        idCounts[i].id = 0;
		        idCounts[i].count = 0;
	        }
			mostFrequent = 0;
			nSamples = 0;
        }
    };


    struct FrameBufferData
    {
        //math::vec3f* toneMappedRadiance;

        ///////////////////////////////////////////
        /////////////// Passes ////////////////////
        ///////////////////////////////////////////
        math::vec3f* radianceAccumulator;
        math::vec3f* albedoAccumulator;
        math::vec3f* normalAccumulator;

        math::vec3f* tmRadiance;
        math::vec3f* hdriRadiance;
        math::vec3f* normalNormalized;
        math::vec3f* albedoNormalized;

        math::vec3f* trueNormal;
        math::vec3f* tangent;
        math::vec3f* orientation;
        math::vec3f* uv;
        math::vec3f* debugColor1;

        math::vec3f* fireflyPass;
        int*         samples;
        NoiseData*   noiseBuffer;
        CUdeviceptr  outputBuffer{};
        math::vec2ui frameSize;

        gBufferHistory* gBufferHistory;
        float* gBuffer;
	};

    


    //struct NeuralNetworkDeviceSettings
    //{
    //    bool                 useNetwork;
    //    network::NetworkType type;
    //    int                  inferenceStart;
    //    bool                 clearOnInferenceStart;
    //    bool                 doInference;
    //    float                samplingFraction = 0.5f;
    //};

    //struct RendererDeviceSettings
    //{
    //    int                 iteration;
    //    int                 maxBounces;
    //    bool                accumulate;
    //    SamplingTechnique   samplingTechnique;
	//	DisplayBuffer       displayBuffer;
    //    float               minClamp;
    //    float               maxClamp;
    //
    //    bool                adaptiveSampling;
    //    int                 minAdaptiveSamples;
    //    int                 minPixelSamples;
    //    int                 maxPixelSamples;
    //    float               noiseCutOff;
    //
    //    bool                useLongPathKernel;
    //    float               longPathPercentage;
    //    int                 maxTraceQueueSize;
    //
    //    bool                     enableDenoiser;
    //    bool                     removeFirefly;
	//	bool                     useRussianRoulette;
	//	int                      parallelShade;
    //
	//};

    struct Counters {
        int* traceQueueCounter = nullptr;
        int* shadeQueueCounter = nullptr;
        int* escapedQueueCounter = nullptr;
        int* accumulationQueueCounter = nullptr;
        int* shadowQueueCounter = nullptr;
    };

    struct QueueSizes {
        int traceQueueCounter = 0;
        int shadeQueueCounter = 0;
        int escapedQueueCounter = 0;
        int accumulationQueueCounter = 0;
        int shadowQueueCounter = 0;
    };

    struct OnDeviceSettings
    {
		RendererSettings renderer;
        WavefrontSettings wavefront;
        network::config::NetworkSettings neural;
    };

    struct QueuesData
    {
        WorkQueueSOA<TraceWorkItem>* radianceTraceQueue = nullptr;
        WorkQueueSOA<RayWorkItem>* shadeQueue = nullptr;
        WorkQueueSOA<ShadowWorkItem>* shadowQueue = nullptr;
        WorkQueueSOA<EscapedWorkItem>* escapedQueue = nullptr;
        WorkQueueSOA<AccumulationWorkItem>* accumulationQueue = nullptr;
        Counters*                           queueCounters = nullptr;
    };

	struct LaunchParams
    {
		int*                    frameID;
		int                     nextPixel = 0;
		FrameBufferData         frameBuffer;
		CameraData              cameraData;
		//SbtProgramIdx*          programs;
		OptixTraversableHandle  topObject = 0;
		InstanceData**          instances;
		LightData*              envLight = nullptr;
		LightData**             lights;
		int                     numberOfLights;

        QueuesData              queues;

		NetworkInterface* networkInterface = nullptr;

        OnDeviceSettings settings;
		OptixAabb              aabb;
        DebugData*             debugData;
	};

    enum TypeRay
    {
        TYPE_RAY_RADIANCE = 0,
        NUM_RAY_TYPES
    };

    struct LensRay
    {
        math::vec3f org;
        math::vec3f dir;
    };

} // ::osc

#endif
