#pragma once
#ifndef MDL_STRUCT_H
#define MDL_STRUCT_H
#include "Device/DevicePrograms/RayData.h"
#include "Device/DevicePrograms/HitProperties.h"

namespace vtx::mdl
{

    typedef mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE>   BsdfEvaluateData;
    typedef mi::neuraylib::Bsdf_sample_data                                 BsdfSampleData;
    typedef mi::neuraylib::Bsdf_auxiliary_data<mi::neuraylib::DF_HSM_NONE>  BsdfAuxiliaryData;
    typedef mi::neuraylib::Edf_evaluate_data<mi::neuraylib::DF_HSM_NONE>    EdfEvaluateData;

    typedef mi::neuraylib::Bsdf_event_type BsdfEventType;

    __forceinline__ __device__ const char* bsdfEventTypeName(const BsdfEventType& eventType) {
        switch (eventType) {
        case mi::neuraylib::BSDF_EVENT_ABSORB: return "BSDF_EVENT_ABSORB";
        case mi::neuraylib::BSDF_EVENT_DIFFUSE: return "BSDF_EVENT_DIFFUSE";
        case mi::neuraylib::BSDF_EVENT_GLOSSY: return "BSDF_EVENT_GLOSSY";
        case mi::neuraylib::BSDF_EVENT_SPECULAR: return "BSDF_EVENT_SPECULAR";
        case mi::neuraylib::BSDF_EVENT_REFLECTION: return "BSDF_EVENT_REFLECTION";
        case mi::neuraylib::BSDF_EVENT_TRANSMISSION: return "BSDF_EVENT_TRANSMISSION";
        case mi::neuraylib::BSDF_EVENT_DIFFUSE_REFLECTION: return "BSDF_EVENT_DIFFUSE_REFLECTION";
        case mi::neuraylib::BSDF_EVENT_DIFFUSE_TRANSMISSION: return "BSDF_EVENT_DIFFUSE_TRANSMISSION";
        case mi::neuraylib::BSDF_EVENT_GLOSSY_REFLECTION: return "BSDF_EVENT_GLOSSY_REFLECTION";
        case mi::neuraylib::BSDF_EVENT_GLOSSY_TRANSMISSION: return "BSDF_EVENT_GLOSSY_TRANSMISSION";
        case mi::neuraylib::BSDF_EVENT_SPECULAR_REFLECTION: return "BSDF_EVENT_SPECULAR_REFLECTION";
        case mi::neuraylib::BSDF_EVENT_SPECULAR_TRANSMISSION: return "BSDF_EVENT_SPECULAR_TRANSMISSION";
        case mi::neuraylib::BSDF_EVENT_FORCE_32_BIT: return "BSDF_EVENT_FORCE_32_BIT";
        case (mi::neuraylib::BSDF_EVENT_DIFFUSE| mi::neuraylib::BSDF_EVENT_GLOSSY): return "BSDF_EVENT_DIFFUSE_GLOSSY";
        default: return "UNKNOWN_BSDF_EVENT_TYPE";
        }
    }

    struct MdlData
    {
        mi::neuraylib::Shading_state_material   state;
        mi::neuraylib::Resource_data            resourceData;
        char* argBlock;
        bool                                    isFrontFace;
        bool                                    isThinWalled;
        float                                   opacity;
    };

    struct EdfResult
    {
        math::vec3f intensity{0.0f};
        int         mode;
        float       cos;
        math::vec3f edf;
        float       pdf;
        bool        isValid = false;
    };

    struct BsdfEvalResult
    {
        math::vec3f diffuse;
        math::vec3f glossy;
        float       pdf;
        float       bsdfPdf;
        bool        isValid = false;
		math::vec3f bsdf;
        BsdfEventType eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
    };

    struct BsdfSampleResult
    {
        math::vec3f   nextDirection;
        float         pdf;
        float         bsdfPdf;
        math::vec3f   bsdfOverPdf;
        math::vec3f   bsdf; //(bsdf * cosTheta)
        BsdfEventType eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
        bool          isValid = false;
        bool          isComputed = false;
        BsdfEvalResult bsdfEvaluation;

        __device__ void print(const char* message="")
        {
            printf("%s"
                   "isValid: %d\n"
				   "isComputed %d\n"
                   "Next Direction: %f %f %f\n"
                   "Bsdf Over Pdf: %f %f %f\n"
                   "pdf: %f\n"
                   "eventType: %d\n\n",
                   message,
                   isValid,
                   isComputed,
                   nextDirection.x, nextDirection.y, nextDirection.z,
                   bsdfOverPdf.x, bsdfOverPdf.y, bsdfOverPdf.z,
                   pdf,
                   eventType);
        }
    };

    struct BsdfAuxResult
    {
        math::vec3f   albedo;
        math::vec3f   normal;
        bool          isValid = false;
    };

    struct MaterialEvaluation
    {
        BsdfEvalResult   bsdfEvaluation;
        BsdfSampleResult bsdfSample;
        EdfResult        edf;
        BsdfAuxResult    aux;
        math::vec3f      ior;
        float            opacity;
        bool             isThinWalled;
        // We export these as well to not compute them twice
        BsdfEvalResult   neuralBsdfEvaluation;

	};

    struct MdlRequest
    {
		bool           bsdfEvaluation = false;
		bool           bsdfSample     = false;
		bool           auxiliary      = false;
		bool           edf            = false;
		bool           opacity        = false;
		bool           ior            = false;
		math::vec3f    outgoingDirection;
		math::vec3f    toSampledLight;
		math::vec3f    surroundingIor;
		unsigned*      seed;
		HitProperties* hitProperties;
		bool           evalOnNeuralSampling = false;
		math::vec3f    toNeuralSample;
	};
}
#endif
