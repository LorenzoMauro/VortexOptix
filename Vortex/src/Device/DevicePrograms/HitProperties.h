#ifndef HIT_PROPERTIES_H
#define HIT_PROPERTIES_H
#pragma once
#include "CudaDebugHelper.h"
#include "LaunchParams.h"
#include "Utils.h"
#include "Core/Math.h"

namespace vtx
{
	struct HitProperties
    {
        bool isInit = false;
        unsigned instanceId;
        unsigned triangleId;
        math::vec3f position;
        math::vec3f baricenter;
        math::vec3f trueNormal;
        math::vec3f shadingNormal;
        math::vec3f uv;
        math::vec3f bitangent;
        math::vec3f tangent;
        const math::affine3f* oTw;
        bool isFrontFace;

        // Material Related data
        bool hasMaterial = false;
        int programCall;
        bool hasEdf = false;
        bool hasOpacity = false;
        char* argBlock = nullptr;
        TextureHandler* textureHandler = nullptr;
        LightData* lightData = nullptr;

        __forceinline__ __device__ void init(const unsigned& _instanceId, const unsigned& _triangleId, const math::vec3f& _baricenter, const math::vec3f& _position)
        {
            instanceId = _instanceId;
            triangleId = _triangleId;
            baricenter = _baricenter;
            position = _position;
            isInit = true;
        }

        // Can be used after defining instanceId and triangleId
        __forceinline__ __device__ void getTriangleVertices(
            const InstanceData& instance,
            const graph::VertexAttributes*& v0,
            const graph::VertexAttributes*& v1,
            const graph::VertexAttributes*& v2
        )
        {
            const GeometryData* geometry = instance.geometryData;
            const unsigned* indices = &geometry->indicesData[triangleId*3];
            v0 = &geometry->vertexAttributeData[indices[0]];
            v1 = &geometry->vertexAttributeData[indices[1]];
            v2 = &geometry->vertexAttributeData[indices[2]];
        }

        // Can be used after defining otw and baricentric Coordinates
        __forceinline__ __device__ void computeSurfaceProperties(
            const graph::VertexAttributes* v0,
            const graph::VertexAttributes* v1,
            const graph::VertexAttributes* v2,
            float rotateTangent
        )
        {
            trueNormal = cross(v1->position - v0->position, v2->position - v0->position);
            shadingNormal   = v0->normal    * baricenter.x + v1->normal     * baricenter.y + v2->normal     * baricenter.z;
            bitangent       = v0->bitangent * baricenter.x + v1->bitangent  * baricenter.y + v2->bitangent  * baricenter.z;
            tangent         = v0->tangent   * baricenter.x + v1->tangent    * baricenter.y + v2->tangent    * baricenter.z;


            if (dot(trueNormal, shadingNormal) < 0.0f) // make sure that shading and geometry normal agree on sideness
            {
                trueNormal = -trueNormal;
            }

            trueNormal      = math::normalize(math::transformNormal3F(*oTw, math::normalize(trueNormal)));
            shadingNormal   = math::normalize(math::transformNormal3F(*oTw, math::normalize(shadingNormal)));
            tangent         = math::normalize(math::transformVector3F(*oTw, math::normalize(tangent)));
            bitangent       = math::normalize(math::transformVector3F(*oTw, math::normalize(bitangent)));

            const float rotRad = rotateTangent * 3.14159265358979323846f / 180.0f;
            const float cosRot = cos(rotRad);
            const float sinRot = sin(rotRad);
            tangent = math::normalize(math::vec3f(
				cosRot * tangent.x + sinRot * bitangent.x,
				cosRot * tangent.y + sinRot * bitangent.y,
				cosRot * tangent.z + sinRot * bitangent.z
			));
            bitangent = math::normalize(cross(shadingNormal, tangent));

            if(false)
	        {
		        math::vec3f referenceVector(0.0f, 1.0f, 0.0f);

            	// Check if the normal is not the 'up' vector or too close to it
            	if (fabs(math::dot(shadingNormal, referenceVector)) > 0.999f) {
            		referenceVector = math::vec3f(1.0f, 0.0f, 0.0f);
            	}
            	// radial Z coordinates
            	tangent = math::normalize(cross(shadingNormal, referenceVector));
            	bitangent = math::normalize(cross(tangent, shadingNormal));
	        }

            //bitangent = math::normalize(cross(shadingNormal, tangent));
            //tangent = math::normalize(cross(bitangent, shadingNormal));

        	if (utl::isNan(bitangent))
            {
                printf(
                    "\nNan Bitangent\n"
                    "v0 : %f %f %f\n"
                    "v1 : %f %f %f\n"
                    "v2 : %f %f %f\n"
                    "Baricenter : %f %f %f\n"
                    "True Normal : %f %f %f\n"
                    "Shading Normal : %f %f %f\n"
                    "Tangent : %f %f %f\n"
                    ,
                    v0->position.x, v0->position.y, v0->position.z,
                    v1->position.x, v1->position.y, v1->position.z,
                    v2->position.x, v2->position.y, v2->position.z,
                    baricenter.x, baricenter.y, baricenter.z,
                    shadingNormal.x, shadingNormal.y, shadingNormal.z,
                    trueNormal.x, trueNormal.y, trueNormal.z,
                    tangent.x, tangent.y, tangent.z
                );
            }

            uv = v0->texCoord * baricenter.x + v1->texCoord * baricenter.y + v2->texCoord * baricenter.z;
        }

        __forceinline__ __device__ void calculateForMeshLightSampling(const LaunchParams* params)
        {
            const InstanceData& instance = *params->instances[instanceId];
            oTw = &(instance.transform);
            const graph::VertexAttributes* v0 = nullptr;
            const graph::VertexAttributes* v1 = nullptr;
            const graph::VertexAttributes* v2 = nullptr;
            getTriangleVertices(instance, v0, v1, v2);
            computeSurfaceProperties(v0, v1, v2,params->settings.renderer.tangentRotation);
            position = v0->position * baricenter.x + v1->position * baricenter.y + v2->position * baricenter.z;
            position = math::transformPoint3F(*oTw, position);
        }

        __forceinline__ __device__ void calculate(const LaunchParams* params, const math::vec3f& outgoingDirection)
        {
            const InstanceData& instance = *params->instances[instanceId];
            oTw = &(instance.transform);
            const graph::VertexAttributes* v0 = nullptr;
            const graph::VertexAttributes* v1 = nullptr;
            const graph::VertexAttributes* v2 = nullptr;
            getTriangleVertices(instance, v0, v1, v2);
            computeSurfaceProperties(v0, v1, v2, params->settings.renderer.tangentRotation);
            isFrontFace = 0.0f <= dot(-outgoingDirection, trueNormal); // Explicitly include edge-on cases as frontface condition!
            determineMaterialInfo(instance);
        }
        __forceinline__ __device__ void calculateForInferenceQuery(const LaunchParams* params)
        {
            const InstanceData& instance = *params->instances[instanceId];
            const graph::VertexAttributes* v0 = nullptr;
            const graph::VertexAttributes* v1 = nullptr;
            const graph::VertexAttributes* v2 = nullptr;
            getTriangleVertices(instance, v0, v1, v2);
            oTw = &(instance.transform);
            const math::vec3f nsO = math::normalize(v0->normal * baricenter.x + v1->normal * baricenter.y + v2->normal * baricenter.z);
            shadingNormal = math::normalize(math::transformNormal3F(*oTw, nsO));
            determineMaterialInfo(instance);

        }
        __forceinline__ __device__ void determineMaterialInfo(const InstanceData& instance)
        {
            const unsigned& materialSlotIndex = instance.geometryData->faceAttributeData[triangleId].materialSlotId;
            const InstanceData::SlotIds slotIds = instance.materialSlots[materialSlotIndex];
            if (slotIds.material != nullptr)
            {
                textureHandler = slotIds.material->textureHandler;
                argBlock = slotIds.material->argBlock;

#ifdef ARCHITECTURE_OPTIX
                programCall = slotIds.material->materialConfiguration->idxCallEvaluateMaterialWavefront;
#else
                programCall = slotIds.material->materialConfiguration->idxCallEvaluateMaterialWavefrontCuda;
#endif

                if (slotIds.meshLight != nullptr)
                {
                    hasEdf = true;
                    lightData = instance.materialSlots[materialSlotIndex].meshLight;
                }
                if (instance.hasOpacity)
                {
                    hasOpacity = true;
                }
                hasMaterial = true;
            }
            else
            {
                hasMaterial = false;
            }
        }
    };
}

#endif