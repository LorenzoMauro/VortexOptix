#pragma once
#ifndef MDL_WRAPPER_DEVICE_H
#define MDL_WRAPPER_DEVICE_H
#include "Core/Math.h"
#include "Device/DevicePrograms/RayData.h"
#include "InlineMdlDeclarations.h"
#include "Utils.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"

#ifdef DEBUG
#define clangAssert(x) \
	if(!(x)){\
		printf("Assertion failed: %s, File: %s, Line: %d\n", #x, __FILE__, __LINE__);\
		asm("trap;");\
	}
#else
#define clangAssert(x)
#endif

namespace vtx::mdl {

    __forceinline__ __device__ BsdfEvalResult evaluateBsdf(const MdlData* mdlData, const math::vec3f& surroundingIor, const math::vec3f& outgoingDirection, const math::vec3f& incomingDirection)
    {
        BsdfEvaluateData evalData;

        evalData.k1 = outgoingDirection;
        evalData.k2 = incomingDirection;
        evalData.bsdf_diffuse = math::vec3f(0.0f);
        evalData.bsdf_glossy = math::vec3f(0.0f);

        // If the hit is either on the surface or a thin-walled material,
        // the ray is inside the surrounding material and the material ior is on the other side.
        if (mdlData->isFrontFace || mdlData->isThinWalled)
        {
            evalData.ior1 = { 1.0f, 1.0f, 1.0f }; // From surrounding medium ior
            evalData.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // to material ior.
        }
        else
        {
            // When hitting the backface of a non-thin-walled material, 
            // the ray is inside the current material and the surrounding material is on the other side.
            // The material's IOR is the current top-of-stack. We need the one further down!
            evalData.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // From material ior 
            evalData.ior2 = { 1.0f, 1.0f, 1.0f }; // From surrounding medium ior
        }

        if (mdlData->isFrontFace || !mdlData->isThinWalled || true)
        {
            frontBsdf_evaluate(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }
        else
        {
            backBsdf_evaluate(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }

        BsdfEvalResult result;
        result.isValid = true;
        result.pdf = evalData.pdf;
        result.bsdfPdf = evalData.pdf;
        result.diffuse = evalData.bsdf_diffuse;
        result.glossy = evalData.bsdf_glossy;
        result.bsdf = result.diffuse + result.glossy;
        result.eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
        bool isDiffuse = !math::isZero(result.diffuse);
        bool isGlossy = !math::isZero(result.glossy);
        if (!math::isZero(result.diffuse))
        {
            result.eventType = mi::neuraylib::BSDF_EVENT_DIFFUSE;
        }
        if (!math::isZero(result.glossy))
        {
            result.eventType = (mi::neuraylib::Bsdf_event_type)( result.eventType | mi::neuraylib::BSDF_EVENT_GLOSSY);
        }
        // evidently the evaluation is able to pick almost delta functions for the pdf if the light sample exactly matches the reflection direction
        // the pdf becomes extremely large as well as the glossy component
        bool isDelta = (result.pdf > 1000000.0f && length(result.glossy) > 1000000.0f);// || length(result.bsdf) > 1.0f; // not sure about the bsdf condition but in theory for the conservation of energy it should be true
        if (isDelta)
        {
	        result.eventType = mi::neuraylib::BSDF_EVENT_SPECULAR;
		}
        if(result.eventType == mi::neuraylib::BSDF_EVENT_ABSORB)
        {
	        result.isValid = false;
		}
        return result;
    }

	__forceinline__ __device__ void correctBsdfSample(BsdfSampleResult& bsdfSample, const math::vec3f& normal)
    {
        if (bsdfSample.eventType == mi::neuraylib::BSDF_EVENT_ABSORB)
        {
            return;
        }
        if (bsdfSample.eventType != 0 && (math::isZero(bsdfSample.bsdfOverPdf) || math::isNan(bsdfSample.bsdfOverPdf) || utl::isNan(bsdfSample.nextDirection)))
        {
            // TODO: there might be a bug somewhere since sometimes the bsdfOverPdf is 0.0f but the event type is not ABSORB
            // not sure what is going on here
            bsdfSample.isValid = false;
            bsdfSample.eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
            return;
        }

        const float cosTheta = math::dot(bsdfSample.nextDirection, normal);
        const math::vec3f bsdf = bsdfSample.bsdfOverPdf * (bsdfSample.pdf / cosTheta);
        const float bsdfNorm = math::length(bsdf);

        //change Type to specular if bsdfNorm > 1.0f
        if (bsdfNorm > 1.0f || (bsdfSample.pdf == 0.0f && !math::isZero(bsdfSample.bsdfOverPdf)))
        {
            switch (bsdfSample.eventType)
            {
            case mi::neuraylib::BSDF_EVENT_DIFFUSE_REFLECTION:
            {
                bsdfSample.eventType = mi::neuraylib::BSDF_EVENT_SPECULAR_REFLECTION;
            }
            break;
            case mi::neuraylib::BSDF_EVENT_DIFFUSE_TRANSMISSION:
            {
                bsdfSample.eventType = mi::neuraylib::BSDF_EVENT_SPECULAR_TRANSMISSION;
            }
            break;
            case mi::neuraylib::BSDF_EVENT_GLOSSY_REFLECTION:
            {
                bsdfSample.eventType = mi::neuraylib::BSDF_EVENT_SPECULAR_REFLECTION;
            }
            break;
            case mi::neuraylib::BSDF_EVENT_GLOSSY_TRANSMISSION:
            {
                bsdfSample.eventType = mi::neuraylib::BSDF_EVENT_SPECULAR_TRANSMISSION;
            }
            break;
            }

        }
    }

    // Importance sample the BSDF. 
    __forceinline__ __device__ BsdfSampleResult sampleBsdf(const MdlData* mdlData, const math::vec3f& surroundingIor, const math::vec3f& outgoingDirection, unsigned& seed)
    {
        BsdfSampleData data;
        // If the hit is either on the surface or a thin-walled material,
        // the ray is inside the surrounding material and the material ior is on the other side.
        // When hitting the backface of a non-thin-walled material, 
        // the ray is inside the current material and the surrounding material is on the other side.
        // The material's IOR is the current top-of-stack. We need the one further down!
        data.k1 = outgoingDirection; // == -optixGetWorldRayDirection()
        data.xi = rng4(seed);
        data.pdf = 1.0f;

        if (mdlData->isFrontFace || mdlData->isThinWalled)
        {
            data.ior1 = {1.0f, 1.0f, 1.0f};
            data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
        }
        else
        {
            data.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
            data.ior2 = {1.0f, 1.0f, 1.0f};
        }

        if (mdlData->isFrontFace || !mdlData->isThinWalled || true)
        {
            frontBsdf_sample(&data, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }
        else
        {
            backBsdf_sample(&data, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }

        BsdfSampleResult result;
        result.isValid = true;
        result.isComputed = true;
        result.nextDirection = data.k2;
        result.nextDirection = math::normalize(result.nextDirection);
        result.bsdfOverPdf = data.bsdf_over_pdf;
        result.eventType = data.event_type;
        result.pdf = data.pdf;
        result.bsdfPdf = data.pdf;

        correctBsdfSample(result, mdlData->state.normal);
        if (!(result.eventType & mi::neuraylib::BSDF_EVENT_SPECULAR) && (result.eventType != mi::neuraylib::BSDF_EVENT_ABSORB)) {
            result.bsdf = result.bsdfOverPdf * result.bsdfPdf;
        }
        else if (result.eventType & mi::neuraylib::BSDF_EVENT_SPECULAR)
        {
            result.bsdf = result.bsdfOverPdf;
		}

        result.bsdfEvaluation = evaluateBsdf(mdlData, surroundingIor, outgoingDirection, result.nextDirection);

        return result;
    }


    __forceinline__ __device__ BsdfAuxResult auxiliaryBsdf(const MdlData* mdlData, const math::vec3f& surroundingIor, const math::vec3f& outgoingDirection)
    {
        BsdfAuxiliaryData auxData;

        auxData.k1 = outgoingDirection;

        // If the hit is either on the surface or a thin-walled material,
        // the ray is inside the surrounding material and the material ior is on the other side.
        if (mdlData->isFrontFace || mdlData->isThinWalled)
        {
            auxData.ior1 = surroundingIor; // From surrounding medium ior
            auxData.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // to material ior.
        }
        else
        {
            // When hitting the backface of a non-thin-walled material, 
            // the ray is inside the current material and the surrounding material is on the other side.
            // The material's IOR is the current top-of-stack. We need the one further down!
            auxData.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // From material ior 
            auxData.ior2 = surroundingIor; // From surrounding medium ior
        }


        if (mdlData->isFrontFace || !mdlData->isThinWalled)
        {
            frontBsdf_auxiliary(&auxData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }
        else
        {
            backBsdf_auxiliary(&auxData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }

        BsdfAuxResult result;
        result.isValid = true;
        result.albedo = auxData.albedo;
        result.normal = auxData.normal;
        return result;
    }

    __forceinline__ __device__ EdfResult evaluateEmission(const MdlData* mdlData, const math::vec3f& outgoingDirection)
    {
        EdfEvaluateData evalData;
        EdfResult result;

        float3 emissionIntensity;
        emissionIntensity.x = 0.0f;
        emissionIntensity.y = 0.0f;
        emissionIntensity.z = 0.0f;

        // Emission 
        if (mdlData->isFrontFace || !mdlData->isThinWalled)
        {
            frontEdfIntensity(&emissionIntensity, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);

            result.intensity = emissionIntensity;
            //printMath("Front Intensity", emissionIntensity);

            if (result.intensity == math::vec3f(0.0f)) {
                result.isValid = false;
                return result;
            }

            evalData.k1 = outgoingDirection; // input: outgoing direction (-ray.direction)
            frontEdfMode(&result.mode, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
            frontEdf_evaluate(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }
        else
        {
            backEdfIntensity(&emissionIntensity, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
            result.intensity = emissionIntensity;


            if (result.intensity == math::vec3f(0.0f)) {
                result.isValid = false;
                return result;
            }
            evalData.k1 = outgoingDirection; // input: outgoing direction (-ray.direction)
            backEdfMode(&result.mode, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
            backEdf_evaluate(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }

        // Check if the hit geometry contains any emission.
        result.isValid = true;
        result.cos = evalData.cos;
        result.edf = evalData.edf;
        result.pdf = evalData.pdf;
        return result;
    }
    __forceinline__ __device__ void printTransforms(float4* otw, float4* wto)
    {
    	printf(
            "Object to World:\n"
            "%f %f %f %f\n"
            "%f %f %f %f\n"
            "%f %f %f %f\n"
            "World to Object:\n"
            "%f %f %f %f\n"
            "%f %f %f %f\n"
            "%f %f %f %f\n"
		, otw[0].x, otw[0].y, otw[0].z, otw[0].w
		, otw[1].x, otw[1].y, otw[1].z, otw[1].w
		, otw[2].x, otw[2].y, otw[2].z, otw[2].w
		, wto[0].x, wto[0].y, wto[0].z, wto[0].w
		, wto[1].x, wto[1].y, wto[1].z, wto[1].w
		, wto[2].x, wto[2].y, wto[2].z, wto[2].w
		);
	}
    __forceinline__ __device__ MdlData mdlInit(const MdlRequest* request)
    {
        float4 oTwF4[3];
        float4 wToF4[3];
        float3 textureCoordinates[2];
        float3 textureBitangents[2];
        float3 textureTangents[2];
        float4 textureResults[16]; //TODO add macro
        MdlData mdlData{};

        const math::affine3f& objectToWorld = *request->hitProperties->oTw;
		const auto            lInv          = objectToWorld.l.inverse();
        const math::affine3f& worldToObject = math::affine3f(lInv, -lInv * objectToWorld.p);
        objectToWorld.toFloat4(oTwF4);
        worldToObject.toFloat4(wToF4);

        //printTransforms(oTwF4, wToF4);

        textureCoordinates[0] = request->hitProperties->uv;
        textureTangents[0] = request->hitProperties->tangent;
        textureBitangents[0] = request->hitProperties->bitangent;

        textureCoordinates[1] = textureCoordinates[0];
        textureTangents[1] = textureTangents[0];
        textureBitangents[1] = textureBitangents[0];

        mdlData.state.normal = request->hitProperties->shadingNormal;
        mdlData.state.geom_normal = request->hitProperties->trueNormal;
        mdlData.state.position = request->hitProperties->position;
        mdlData.state.animation_time = 0.0f;
        mdlData.state.text_coords = textureCoordinates;

    	mdlData.state.tangent_u = (mi::neuraylib::tct_float3*)textureTangents;
        mdlData.state.tangent_v = (mi::neuraylib::tct_float3*)textureBitangents;

        mdlData.state.text_results = (mi::neuraylib::tct_float4*)textureResults;
        mdlData.state.ro_data_segment = nullptr;
        mdlData.state.world_to_object = (mi::neuraylib::tct_float4*)wToF4;
        mdlData.state.object_to_world = (mi::neuraylib::tct_float4*)oTwF4;
        mdlData.state.object_id = request->hitProperties->instanceId;
        mdlData.state.meters_per_scene_unit = 1.0f;

        mdlData.resourceData.shared_data = nullptr;
        mdlData.resourceData.texture_handler = request->hitProperties->textureHandler;
        mdlData.argBlock = request->hitProperties->argBlock;
        mdlData.isFrontFace = request->hitProperties->isFrontFace;

        clangAssert(!math::isNan(request->hitProperties->shadingNormal));
        clangAssert(!math::isZero(request->hitProperties->shadingNormal));
        clangAssert(!math::isInf(request->hitProperties->shadingNormal));

        clangAssert(!math::isNan(request->hitProperties->trueNormal));
        clangAssert(!math::isZero(request->hitProperties->trueNormal));
        clangAssert(!math::isInf(request->hitProperties->trueNormal));

        clangAssert(!math::isNan(request->hitProperties->position));
        clangAssert(!math::isInf(request->hitProperties->position));
        clangAssert(!math::isNan(request->hitProperties->uv));
        clangAssert(!math::isInf(request->hitProperties->uv));
        clangAssert(!math::isNan(request->hitProperties->bitangent));
        clangAssert(!math::isZero(request->hitProperties->bitangent));
        clangAssert(!math::isInf(request->hitProperties->bitangent));

        clangAssert(!math::isNan(request->hitProperties->tangent));
        clangAssert(!math::isZero(request->hitProperties->tangent));
        clangAssert(!math::isInf(request->hitProperties->tangent));

        clangAssert(!math::isNan(request->outgoingDirection));
        clangAssert(!math::isZero(request->outgoingDirection));
        clangAssert(!math::isInf(request->outgoingDirection));

        clangAssert(mdlData.resourceData.texture_handler != nullptr);
        clangAssert(mdlData.argBlock != nullptr);
        init(&mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        
        return mdlData;
    }
}

#endif